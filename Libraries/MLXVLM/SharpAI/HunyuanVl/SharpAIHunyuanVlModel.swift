// SharpAI fork overlay: hunyuan_vl text decoder, vision tower and patch merger. Not part of
// upstream mlx-swift-lm.
//
// Ported from transformers `models/hunyuan_vl/modular_hunyuan_vl.py` (HF main). Layout is NHWC
// for the convolutions, which is the MLX convention. The creator in
// SharpAIHunyuanVlRegistration.swift stays gated: this type is not a LanguageModel and has no
// KV cache wiring for generation.
//
// UNVERIFIED: checkpoint key names and Conv2d weight layout (PyTorch OIHW vs MLX OHWI) are not
// mapped yet. Loading a real checkpoint needs a sanitize pass for both.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Rotary (HF HunYuanVLRotaryEmbedding and recomposition_frequencies)

enum SharpAIHunyuanVlRotary {

    /// `inv_freq[k] = 1 / base^(2k/d)`. With alpha > 0 the base is scaled as in HF:
    /// `base = theta * alpha^(d / (d - 2))`.
    static func inverseFrequencies(headDim: Int, theta: Float, alpha: Float?) -> [Float] {
        var base = theta
        if let alpha, alpha > 0, headDim > 2 {
            base = theta * powf(alpha, Float(headDim) / Float(headDim - 2))
        }
        return (0 ..< headDim / 2).map { k in
            1.0 / powf(base, Float(2 * k) / Float(headDim))
        }
    }

    /// Cos and sin tables `[B, L, d]` for per-axis positions `[A, B, L]`, recomposed as in HF:
    /// the half-width table is doubled, then chunk `i` (width `2 * section[i]`) is taken from axis
    /// `i % A`.
    static func cosSin(
        positions: MLXArray, inverseFrequencies: [Float], section: [Int]
    ) -> (cos: MLXArray, sin: MLXArray) {
        let invFreq = MLXArray(inverseFrequencies)
        let angles = positions.asType(.float32)[.ellipsis, .newAxis] * invFreq  // [A, B, L, d/2]
        let full = concatenated([angles, angles], axis: -1)  // [A, B, L, d]
        return (
            recompose(MLX.cos(full), section: section),
            recompose(MLX.sin(full), section: section)
        )
    }

    static func recompose(_ table: MLXArray, section: [Int]) -> MLXArray {
        var parts: [MLXArray] = []
        var start = 0
        for (i, width) in section.enumerated() where width > 0 {
            let axis = table[i % section.count]  // [B, L, d]
            parts.append(axis[.ellipsis, start ..< start + 2 * width])
            start += 2 * width
        }
        return concatenated(parts, axis: -1)
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let d = x.dim(-1)
        let first = x[.ellipsis, 0 ..< d / 2]
        let second = x[.ellipsis, d / 2 ..< d]
        return concatenated([-second, first], axis: -1)
    }

    /// `x * cos + rotate_half(x) * sin` for `x` `[B, H, L, d]` and tables `[B, L, d]`.
    static func apply(_ x: MLXArray, cosTable: MLXArray, sinTable: MLXArray) -> MLXArray {
        let c = cosTable.expandedDimensions(axis: 1).asType(x.dtype)
        let s = sinTable.expandedDimensions(axis: 1).asType(x.dtype)
        return x * c + rotateHalf(x) * s
    }
}

// MARK: - Text decoder

final class SharpAIHunyuanVlAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let inverseFrequencies: [Float]
    let section: [Int]

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "query_layernorm") var queryLayerNorm: RMSNorm
    @ModuleInfo(key: "key_layernorm") var keyLayerNorm: RMSNorm

    init(_ config: SharpAIHunyuanVlTextDimensions) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        inverseFrequencies = SharpAIHunyuanVlRotary.inverseFrequencies(
            headDim: headDim, theta: config.ropeTheta, alpha: config.ropeScaling?.alpha)
        section = config.ropeScaling?.xdropeSection ?? [headDim / 2]

        _qProj.wrappedValue = Linear(
            config.hiddenSize, heads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(
            heads * headDim, config.hiddenSize, bias: config.attentionBias)
        _queryLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _keyLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var q = qProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        let (cosTable, sinTable) = SharpAIHunyuanVlRotary.cosSin(
            positions: positions, inverseFrequencies: inverseFrequencies, section: section)
        q = SharpAIHunyuanVlRotary.apply(q, cosTable: cosTable, sinTable: sinTable)
        k = SharpAIHunyuanVlRotary.apply(k, cosTable: cosTable, sinTable: sinTable)
        // HF applies the per-head Q/K RMSNorm after RoPE.
        q = queryLayerNorm(q)
        k = keyLayerNorm(k)

        let out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(out)
    }
}

final class SharpAIHunyuanVlMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hidden: Int, intermediate: Int) {
        _gate.wrappedValue = Linear(hidden, intermediate, bias: false)
        _up.wrappedValue = Linear(hidden, intermediate, bias: false)
        _down.wrappedValue = Linear(intermediate, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAIHunyuanVlDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIHunyuanVlAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIHunyuanVlMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIHunyuanVlTextDimensions) {
        _attention.wrappedValue = SharpAIHunyuanVlAttention(config)
        _mlp.wrappedValue = SharpAIHunyuanVlMLP(
            hidden: config.hiddenSize, intermediate: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), positions: positions, mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// Text backbone (HF `HunYuanVLTextModel`). Parameter names match the HF module tree.
public final class SharpAIHunyuanVlTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIHunyuanVlDecoderLayer]
    let norm: RMSNorm
    let axisCount: Int

    public init(_ config: SharpAIHunyuanVlTextDimensions) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIHunyuanVlDecoderLayer(config)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.axisCount = config.ropeScaling?.xdropeSection?.count ?? 1
    }

    /// Text-only positions: every M-RoPE axis carries the same index, as in HF `get_rope_index`.
    /// Returns `[A, B, L]`.
    public func defaultPositions(batch: Int, length: Int, offset: Int) -> MLXArray {
        var flat: [Float] = []
        flat.reserveCapacity(axisCount * batch * length)
        for _ in 0 ..< axisCount * batch {
            for t in 0 ..< length {
                flat.append(Float(offset + t))
            }
        }
        return MLXArray(flat, [axisCount, batch, length])
    }

    /// Hidden states `[B, L, hidden]` after the final norm.
    public func callAsFunction(
        _ inputs: MLXArray, positions: MLXArray? = nil, cache: [KVCache]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputs)
        let positionIds =
            positions
            ?? defaultPositions(
                batch: h.dim(0), length: h.dim(1), offset: cache?.first?.offset ?? 0)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, positions: positionIds, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

// MARK: - Vision tower

final class SharpAIHunyuanVlVisionAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ v: SharpAIHunyuanVlVisionDimensions) {
        heads = v.attentionHeads
        headDim = v.hiddenSize / v.attentionHeads
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(v.hiddenSize, v.hiddenSize, bias: true)
        _kProj.wrappedValue = Linear(v.hiddenSize, v.hiddenSize, bias: true)
        _vProj.wrappedValue = Linear(v.hiddenSize, v.hiddenSize, bias: true)
        _oProj.wrappedValue = Linear(v.hiddenSize, v.hiddenSize, bias: true)
    }

    /// Full bidirectional attention over one image's patches, `[1, N, D]`. No RoPE.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let q = qProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

final class SharpAIHunyuanVlVisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(hidden: Int, intermediate: Int) {
        _fc1.wrappedValue = Linear(hidden, intermediate, bias: true)
        _fc2.wrappedValue = Linear(intermediate, hidden, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(MLXNN.gelu(fc1(x)))
    }
}

final class SharpAIHunyuanVlVisionBlock: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIHunyuanVlVisionAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SharpAIHunyuanVlVisionMLP

    init(_ v: SharpAIHunyuanVlVisionDimensions) {
        _layerNorm1.wrappedValue = LayerNorm(dimensions: v.hiddenSize, eps: v.rmsNormEps)
        _selfAttn.wrappedValue = SharpAIHunyuanVlVisionAttention(v)
        _layerNorm2.wrappedValue = LayerNorm(dimensions: v.hiddenSize, eps: v.rmsNormEps)
        _mlp.wrappedValue = SharpAIHunyuanVlVisionMLP(
            hidden: v.hiddenSize, intermediate: v.intermediateSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + selfAttn(layerNorm1(x))
        h = h + mlp(layerNorm2(h))
        return h
    }
}

final class SharpAIHunyuanVlPatchEmbedding: Module {
    let channels: Int
    let patchSize: Int
    let hidden: Int
    /// Side of the learned position grid (`max_image_size / patch_size`).
    let edge: Int

    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(_ v: SharpAIHunyuanVlVisionDimensions) {
        channels = v.numChannels
        patchSize = v.patchSize
        hidden = v.hiddenSize
        edge = v.maxImageSize / v.patchSize
        _patchEmbedding.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: hidden,
            kernelSize: .init(patchSize), stride: .init(patchSize))
        // Row 0 is the CLS slot. Rows 1... are the learned grid.
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: edge * edge + 1, dimensions: hidden)
    }

    /// Weights `[output, input]` for bilinear resize with align_corners=False.
    static func bilinearWeights(output: Int, input: Int) -> MLXArray {
        var flat = [Float](repeating: 0, count: output * input)
        let scale = Float(input) / Float(output)
        for o in 0 ..< output {
            let src = max((Float(o) + 0.5) * scale - 0.5, 0)
            let i0 = min(Int(src), input - 1)
            let i1 = min(i0 + 1, input - 1)
            let lambda = src - Float(i0)
            flat[o * input + i0] += 1 - lambda
            flat[o * input + i1] += lambda
        }
        return MLXArray(flat, [output, input])
    }

    /// Learned grid resized to `h x w`, `[h * w, D]`.
    func interpolatedPositions(h: Int, w: Int) -> MLXArray {
        let grid = positionEmbedding.weight[1 ..< 1 + edge * edge]  // [E*E, D]
        let g = grid.reshaped(edge, edge, hidden).transposed(2, 0, 1)  // [D, E, E]
        let rows = Self.bilinearWeights(output: h, input: edge)  // [h, E]
        let cols = Self.bilinearWeights(output: w, input: edge)  // [w, E]
        let out = MLX.matmul(MLX.matmul(rows, g), cols.transposed())  // [D, h, w]
        return out.transposed(1, 2, 0).reshaped(h * w, hidden)
    }

    /// `pixelValues` `[N, C * p * p]` with N = h * w (one image). Returns `[1, N, D]`.
    func callAsFunction(_ pixelValues: MLXArray, h: Int, w: Int) -> MLXArray {
        let n = pixelValues.dim(0)
        let nhwc = pixelValues.reshaped(n, channels, patchSize, patchSize).transposed(0, 2, 3, 1)
        let patches = patchEmbedding(nhwc).reshaped(n, hidden)
        return (patches + interpolatedPositions(h: h, w: w)).expandedDimensions(axis: 0)
    }
}

/// Spatial merge and projection to the text width (HF `HunYuanVLVisionPatchMerger`).
final class SharpAIHunyuanVlPatchMerger: Module {
    let hidden: Int

    @ModuleInfo(key: "before_rms") var beforeRms: RMSNorm
    @ModuleInfo(key: "proj_conv") var projConv: Conv2d
    @ModuleInfo(key: "proj_out") var projOut: Conv2d
    @ModuleInfo(key: "mlp") var mlp: Linear
    @ModuleInfo(key: "after_rms") var afterRms: RMSNorm
    @ParameterInfo(key: "image_newline") var imageNewline: MLXArray
    @ParameterInfo(key: "image_begin") var imageBegin: MLXArray
    @ParameterInfo(key: "image_end") var imageEnd: MLXArray
    @ParameterInfo(key: "image_sep") var imageSep: MLXArray

    init(_ v: SharpAIHunyuanVlVisionDimensions, textHiddenSize: Int) {
        let d = v.hiddenSize
        let m = v.spatialMergeSize
        hidden = d
        _beforeRms.wrappedValue = RMSNorm(dimensions: d, eps: v.rmsNormEps)
        _projConv.wrappedValue = Conv2d(
            inputChannels: d, outputChannels: 2 * d, kernelSize: .init(m), stride: .init(m))
        _projOut.wrappedValue = Conv2d(
            inputChannels: 2 * d, outputChannels: 4 * d, kernelSize: .init(1))
        _mlp.wrappedValue = Linear(4 * d, textHiddenSize)
        _afterRms.wrappedValue = RMSNorm(dimensions: textHiddenSize, eps: v.rmsNormEps)
        _imageNewline.wrappedValue = MLXArray.zeros([4 * d])
        _imageBegin.wrappedValue = MLXArray.zeros([textHiddenSize])
        _imageEnd.wrappedValue = MLXArray.zeros([textHiddenSize])
        _imageSep.wrappedValue = MLXArray.zeros([textHiddenSize])
    }

    /// `hiddenStates` `[1, h * w, D]`. Returns `[1, h' * (w' + 1) + 2, T]` with h' = h / m,
    /// w' = w / m. The newline column is appended per row. Begin and end tokens wrap the sequence.
    func callAsFunction(_ hiddenStates: MLXArray, h: Int, w: Int) -> MLXArray {
        var x = beforeRms(hiddenStates).reshaped(1, h, w, hidden)  // NHWC
        x = MLXNN.gelu(projConv(x))
        x = projOut(x)  // [1, h', w', 4D]
        let (hp, wp, c) = (x.dim(1), x.dim(2), x.dim(3))
        let newline =
            MLXArray.zeros([1, hp, 1, c]).asType(x.dtype)
            + imageNewline.reshaped(1, 1, 1, c).asType(x.dtype)
        x = concatenated([x, newline], axis: 2).reshaped(1, hp * (wp + 1), c)
        x = mlp(x)
        let t = x.dim(-1)
        let begin = imageBegin.reshaped(1, 1, t).asType(x.dtype)
        let end = imageEnd.reshaped(1, 1, t).asType(x.dtype)
        x = concatenated([begin, x, end], axis: 1)
        return afterRms(x)
    }
}

/// Vision tower (HF `HunYuanVLVisionTransformer`). Parameter names match the HF module tree.
public final class SharpAIHunyuanVlVisionTransformer: Module {
    @ModuleInfo(key: "embeddings") var embeddings: SharpAIHunyuanVlPatchEmbedding
    let layers: [SharpAIHunyuanVlVisionBlock]
    @ModuleInfo(key: "patch_merger") var patchMerger: SharpAIHunyuanVlPatchMerger
    let spatialMergeSize: Int

    public init(_ v: SharpAIHunyuanVlVisionDimensions, textHiddenSize: Int) {
        _embeddings.wrappedValue = SharpAIHunyuanVlPatchEmbedding(v)
        self.layers = (0 ..< v.depth).map { _ in SharpAIHunyuanVlVisionBlock(v) }
        _patchMerger.wrappedValue = SharpAIHunyuanVlPatchMerger(v, textHiddenSize: textHiddenSize)
        self.spatialMergeSize = v.spatialMergeSize
    }

    /// `pixelValues` `[sum(h_i * w_i), C * p * p]`, images stacked in order. `grids` holds one
    /// `(h, w)` patch grid per image, in patches. Each side must be divisible by the spatial merge
    /// size. Returns `[1, sum(h'_i * (w'_i + 1) + 2), T]`.
    public func callAsFunction(_ pixelValues: MLXArray, grids: [(h: Int, w: Int)]) -> MLXArray {
        precondition(!grids.isEmpty, "hunyuan_vl vision needs at least one image grid")
        var outputs: [MLXArray] = []
        var start = 0
        for grid in grids {
            precondition(
                grid.h % spatialMergeSize == 0 && grid.w % spatialMergeSize == 0,
                "hunyuan_vl grid \(grid) must be divisible by spatial_merge_size")
            let count = grid.h * grid.w
            var h = embeddings(pixelValues[start ..< start + count], h: grid.h, w: grid.w)
            start += count
            for layer in layers {
                h = layer(h)
            }
            outputs.append(patchMerger(h, h: grid.h, w: grid.w))
        }
        return concatenated(outputs, axis: 1)
    }
}

// MARK: - Full model

/// Text backbone, optional vision tower and LM head (HF `HunYuanVLForConditionalGeneration`,
/// without image scatter into the token embeddings). Text-only logits go through `callAsFunction`.
public final class SharpAIHunyuanVlModel: Module {
    @ModuleInfo(key: "model") var model: SharpAIHunyuanVlTextModel
    @ModuleInfo(key: "vision_tower") var visionTower: SharpAIHunyuanVlVisionTransformer?
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let vocabularySize: Int

    public init(_ configuration: SharpAIHunyuanVlConfiguration) {
        let text = configuration.textDimensions
        vocabularySize = text.vocabSize
        _model.wrappedValue = SharpAIHunyuanVlTextModel(text)
        if let vision = configuration.visionDimensions {
            _visionTower.wrappedValue = SharpAIHunyuanVlVisionTransformer(
                vision, textHiddenSize: text.hiddenSize)
        }
        if !text.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(text.hiddenSize, text.vocabSize, bias: false)
        }
    }

    /// Text logits `[B, L, vocab]`.
    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        let hidden = model(inputs)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    /// Image features `[1, sum(h'(w'+1)+2), T]`, or nil when the config has no vision tower.
    public func imageFeatures(_ pixelValues: MLXArray, grids: [(h: Int, w: Int)]) -> MLXArray? {
        guard let visionTower else { return nil }
        return visionTower(pixelValues, grids: grids)
    }
}
