// SharpAI fork overlay: llava_onevision2 model (onevision_encoder vision tower, 2x2 patch merger,
// Qwen3-style text decoder). Not part of upstream mlx-swift-lm.
//
// Ported from lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct, modeling_llava_onevision2.py:
// VisionRotaryEmbedding (4:6:6 T:H:W split, cat([f, f]) table with split-half rotate_half), OneVisionEncoderEmbeddings,
// OneVisionEncoderAttention / EncoderLayer, LlavaOnevision2VisionPatchMerger, LlavaOnevision2Model
// (masked_scatter of image features into the placeholder tokens) and the causal LM head.
// Checkpoint names are produced by SharpAILlavaOnevision2Sanitizer and were checked against the
// repo's model.safetensors.index.json (696 keys).
//
// Scope, stated plainly:
// - Images only. Video input throws; the reference aliases video frames onto the image path, and
//   callers should pass frames as images with their grid_thw.
// - RoPE positions are built from grid_thw in the processor's 2x2 block layout, which matches
//   `build_patch_positions` in video_processing_llava_onevision2.py (the reference `patch_positions`).
//   The merger groups consecutive patches, so the patch tensor must be in this same order.
//   Frame indices are dense (0..T-1); the processor's optional `frame_indices` is not accepted.
// - Attention inside the vision tower is block-diagonal per frame window (fixed_t =
//   frame_windows_size), as the reference cu_seqlens path does. The reference also sends each window
//   through the same RoPE grid, which this file does as well.
// - Not implemented: use_head (pooling probe), use_patch_position_encoding, tied embeddings, batch > 1.
// - Prefill is unchunked: the whole prompt runs in one forward, because image placeholders must be
//   spliced before the text decoder sees them.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Vision rotary embedding

/// Per-patch rotary angles for the 3D (t, h, w) grid. The half head-dimension is split 4:6:6
/// across T, H and W, and the half-width table is repeated (`cat([f, f])`) before `cos` and `sin`,
/// as in the reference `apply_rotary_pos_emb`.
enum SharpAILlavaOnevision2VisionRope {

    static func cosSin(
        positions: [(t: Int, h: Int, w: Int)], headDim: Int, base: Float
    ) -> (cos: MLXArray, sin: MLXArray) {
        let half = headDim / 2
        let unit = half / 16
        let tSize = 4 * unit
        let hSize = 6 * unit
        let wSize = 6 * unit

        func inverseFrequencies(_ size: Int) -> [Float] {
            (0 ..< size).map { 1.0 / powf(base, Float($0) / Float(size)) }
        }
        let invT = inverseFrequencies(tSize)
        let invH = inverseFrequencies(hSize)
        let invW = inverseFrequencies(wSize)

        var values = [Float]()
        values.reserveCapacity(positions.count * headDim)
        for p in positions {
            var freq = [Float]()
            freq.reserveCapacity(half)
            for j in 0 ..< tSize { freq.append(Float(p.t) * invT[j]) }
            for j in 0 ..< hSize { freq.append(Float(p.h) * invH[j]) }
            for j in 0 ..< wSize { freq.append(Float(p.w) * invW[j]) }
            values += freq
            values += freq
        }

        let angles = MLXArray(values, [positions.count, headDim])
        return (MLX.cos(angles), MLX.sin(angles))
    }

    /// Split-half `rotate_half`, the form that matches the `cat([f, f])` table: `(x1, x2)` becomes
    /// `(-x2, x1)` over the two halves of the last axis. Applied to `[B, H, L, D]`.
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(3) / 2
        let first = x[.ellipsis, 0 ..< half]
        let second = x[.ellipsis, half...]
        return concatenated([-second, first], axis: -1)
    }
}

// MARK: - Vision tower

/// `model.visual.embeddings`: a bias-free patch projection. The checkpoint stores a Conv2d weight;
/// `SharpAILlavaOnevision2Sanitizer` flattens it to `[out, C * p * p]`, which matches the
/// `(C, p, p)` order of each input patch.
final class SharpAILlavaOnevision2VisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear

    init(channels: Int, patchSize: Int, hidden: Int) {
        _patchEmbedding.wrappedValue = Linear(
            channels * patchSize * patchSize, hidden, bias: false)
    }

    func callAsFunction(_ patches: MLXArray) -> MLXArray {
        patchEmbedding(patches)
    }
}

final class SharpAILlavaOnevision2VisionAttention: Module {
    let heads: Int
    let scale: Float

    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear

    init(hidden: Int, heads: Int) {
        self.heads = heads
        self.scale = pow(Float(hidden / heads), -0.5)
        _qkv.wrappedValue = Linear(hidden, 3 * hidden, bias: true)
        _proj.wrappedValue = Linear(hidden, hidden, bias: true)
    }

    /// `x` is `[1, L, hidden]`. `cos` and `sin` are `[1, 1, L, D]`. `mask` is a block mask for
    /// the frame windows.
    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        // (B, L, 3, H, D): the reference reshape order, with q, k, v on the third axis.
        let packed = qkv(x).reshaped(batch, length, 3, heads, -1)
        var q = packed[0..., 0..., 0].transposed(0, 2, 1, 3)
        var k = packed[0..., 0..., 1].transposed(0, 2, 1, 3)
        let v = packed[0..., 0..., 2].transposed(0, 2, 1, 3)

        q = q * cos + SharpAILlavaOnevision2VisionRope.rotateHalf(q) * sin
        k = k * cos + SharpAILlavaOnevision2VisionRope.rotateHalf(k) * sin

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)
        return proj(out)
    }
}

final class SharpAILlavaOnevision2VisionMLP: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(hidden: Int, intermediate: Int) {
        _fc1.wrappedValue = Linear(hidden, intermediate, bias: true)
        _fc2.wrappedValue = Linear(intermediate, hidden, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(MLXNN.gelu(fc1(x)))
    }
}

final class SharpAILlavaOnevision2VisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAILlavaOnevision2VisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo var mlp: SharpAILlavaOnevision2VisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: SharpAILlavaOnevision2VisionConfiguration) {
        _attention.wrappedValue = SharpAILlavaOnevision2VisionAttention(
            hidden: config.hiddenSize, heads: config.attentionHeads)
        _layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        _mlp.wrappedValue = SharpAILlavaOnevision2VisionMLP(
            hidden: config.hiddenSize, intermediate: config.intermediateSize)
        _layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        var h = x + attention(layerNorm1(x), cos: cos, sin: sin, mask: mask)
        h = h + mlp(layerNorm2(h))
        return h
    }
}

/// `model.visual.encoder`: the stack of pre-norm layers. The array key path is `layers.N`.
final class SharpAILlavaOnevision2VisionEncoder: Module {
    let layers: [SharpAILlavaOnevision2VisionEncoderLayer]

    init(_ config: SharpAILlavaOnevision2VisionConfiguration) {
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAILlavaOnevision2VisionEncoderLayer(config)
        }
    }

    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, cos: cos, sin: sin, mask: mask)
        }
        return h
    }
}

/// `model.visual.merger`: LayerNorm on each patch, then 2x2 patches are concatenated and projected
/// through Linear, GELU, Linear. The checkpoint names of the two linears are `mlp.0` and `mlp.2`,
/// which the sanitizer renames to `fc1` and `fc2`.
final class SharpAILlavaOnevision2PatchMerger: Module {
    let mergedDim: Int
    @ModuleInfo(key: "ln_q") var lnQ: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear

    init(contextDim: Int, outDim: Int, spatialMergeSize: Int, eps: Float) {
        mergedDim = contextDim * spatialMergeSize * spatialMergeSize
        _lnQ.wrappedValue = LayerNorm(dimensions: contextDim, eps: eps)
        _fc1.wrappedValue = Linear(mergedDim, mergedDim, bias: true)
        _fc2.wrappedValue = Linear(mergedDim, outDim, bias: true)
    }

    /// `x` is `[N, contextDim]` in 2x2 block order. Returns `[N / merge², outDim]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let grouped = lnQ(x).reshaped(-1, mergedDim)
        return fc2(MLXNN.gelu(fc1(grouped)))
    }
}

/// `model.visual`: patch embedding, pre-norm, the encoder, and the merger.
final class SharpAILlavaOnevision2VisionModel: Module {
    let spatialMergeSize: Int
    let frameWindowSize: Int
    let headDim: Int
    let ropeTheta: Float

    @ModuleInfo var embeddings: SharpAILlavaOnevision2VisionEmbeddings
    @ModuleInfo(key: "layernorm_pre") var layernormPre: LayerNorm
    @ModuleInfo var encoder: SharpAILlavaOnevision2VisionEncoder
    @ModuleInfo var merger: SharpAILlavaOnevision2PatchMerger

    init(_ config: SharpAILlavaOnevision2VisionConfiguration) {
        spatialMergeSize = config.spatialMergeSize
        frameWindowSize = config.frameWindowSize
        headDim = config.hiddenSize / config.attentionHeads
        ropeTheta = config.effectiveRopeTheta

        _embeddings.wrappedValue = SharpAILlavaOnevision2VisionEmbeddings(
            channels: config.numChannels, patchSize: config.patchSize, hidden: config.hiddenSize)
        _layernormPre.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        _encoder.wrappedValue = SharpAILlavaOnevision2VisionEncoder(config)
        _merger.wrappedValue = SharpAILlavaOnevision2PatchMerger(
            contextDim: config.hiddenSize,
            outDim: config.outHiddenSize,
            spatialMergeSize: config.spatialMergeSize,
            eps: config.layerNormEps)
    }

    /// `(t, h, w)` for every patch in the processor's 2x2 block layout (`build_patch_positions`):
    /// per frame, the merge blocks in row-major order, and within each block its patches in row-major order.
    static func gridPositions(_ grid: [THW], spatialMergeSize m: Int) -> [(t: Int, h: Int, w: Int)] {
        var positions = [(t: Int, h: Int, w: Int)]()
        for g in grid {
            for ti in 0 ..< g.t {
                for bh in 0 ..< (g.h / m) {
                    for bw in 0 ..< (g.w / m) {
                        for i in 0 ..< m {
                            for j in 0 ..< m {
                                positions.append((t: ti, h: bh * m + i, w: bw * m + j))
                            }
                        }
                    }
                }
            }
        }
        return positions
    }

    /// Patch counts per frame window, as the reference `_build_cu_seqlens` splits them.
    static func windowLengths(_ grid: [THW], fixedT: Int) -> [Int] {
        var lengths = [Int]()
        for g in grid {
            let perFrame = g.h * g.w
            if fixedT > 0, g.t > fixedT {
                for _ in 0 ..< (g.t / fixedT) { lengths.append(fixedT * perFrame) }
                let remainder = g.t % fixedT
                if remainder > 0 { lengths.append(remainder * perFrame) }
            } else {
                lengths.append(g.t * perFrame)
            }
        }
        return lengths
    }

    /// Block-diagonal mask over the patch sequence, or `.none` when there is one window.
    static func windowMask(lengths: [Int], total: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard lengths.count > 1 else { return .none }
        var flags = [Bool](repeating: false, count: total * total)
        var start = 0
        for length in lengths {
            for i in start ..< (start + length) {
                for j in start ..< (start + length) {
                    flags[i * total + j] = true
                }
            }
            start += length
        }
        return .array(MLXArray(flags, [total, total]))
    }

    /// `pixels` is `[N, C * p * p]`. Returns the merged features `[sum(t*h*w) / merge², outHidden]`.
    func callAsFunction(_ pixels: MLXArray, grid: [THW]) throws -> MLXArray {
        let total = grid.reduce(0) { $0 + $1.t * $1.h * $1.w }
        guard pixels.dim(0) == total else {
            throw SharpAILlavaOnevision2Error.patchCountMismatch(
                patches: pixels.dim(0), grid: total)
        }
        for g in grid where g.h % spatialMergeSize != 0 || g.w % spatialMergeSize != 0 {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: image grid \(g.h)x\(g.w) is not divisible by spatial_merge_size")
        }

        var h = layernormPre(embeddings(pixels))
        let (cos, sin) = SharpAILlavaOnevision2VisionRope.cosSin(
            positions: Self.gridPositions(grid, spatialMergeSize: spatialMergeSize),
            headDim: headDim, base: ropeTheta)
        let cos4 = cos.reshaped(1, 1, total, headDim)
        let sin4 = sin.reshaped(1, 1, total, headDim)
        let mask = Self.windowMask(
            lengths: Self.windowLengths(grid, fixedT: frameWindowSize), total: total)

        h = encoder(h.expandedDimensions(axis: 0), cos: cos4, sin: sin4, mask: mask)[0]
        return merger(h)
    }
}

// MARK: - Text decoder (Qwen3 layout, RoPE theta from config)

final class SharpAILlavaOnevision2TextAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ config: SharpAILlavaOnevision2TextConfiguration) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)

        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        rope = initializeRope(
            dims: headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: nil)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x).reshaped(batch, length, heads, -1))
            .transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(batch, length, kvHeads, -1))
            .transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)

        return oProj(output)
    }
}

final class SharpAILlavaOnevision2TextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: SharpAILlavaOnevision2TextConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAILlavaOnevision2TextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAILlavaOnevision2TextAttention
    @ModuleInfo var mlp: SharpAILlavaOnevision2TextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAILlavaOnevision2TextConfiguration) {
        _attention.wrappedValue = SharpAILlavaOnevision2TextAttention(config)
        _mlp.wrappedValue = SharpAILlavaOnevision2TextMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Composite model

/// The `model.` prefix of the sanitized checkpoint: text embeddings, decoder layers, final norm and
/// the vision tower (`model.visual.*`).
final class SharpAILlavaOnevision2Inner: Module {
    @ModuleInfo var visual: SharpAILlavaOnevision2VisionModel
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAILlavaOnevision2TextDecoderLayer]
    let norm: RMSNorm

    init(_ configuration: SharpAILlavaOnevision2Configuration) {
        let text = configuration.textConfiguration
        _visual.wrappedValue = SharpAILlavaOnevision2VisionModel(configuration.visionConfiguration)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: text.vocabularySize, dimensions: text.hiddenSize)
        self.layers = (0 ..< text.hiddenLayers).map { _ in
            SharpAILlavaOnevision2TextDecoderLayer(text)
        }
        self.norm = RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
    }

    /// Runs the decoder over already-embedded inputs `[B, L, hidden]`.
    func callAsFunction(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

/// Conditional-generation model. Parameter paths match the sanitized checkpoint names.
public final class SharpAILlavaOnevision2Model: Module, LanguageModel, KVCacheDimensionProvider {
    public let kvHeads: [Int]
    let configuration: SharpAILlavaOnevision2Configuration

    @ModuleInfo var model: SharpAILlavaOnevision2Inner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ configuration: SharpAILlavaOnevision2Configuration) {
        let text = configuration.textConfiguration
        self.kvHeads = Array(repeating: text.kvHeads, count: text.hiddenLayers)
        self.configuration = configuration

        _model.wrappedValue = SharpAILlavaOnevision2Inner(configuration)
        _lmHead.wrappedValue = Linear(text.hiddenSize, text.vocabularySize, bias: false)
    }

    /// Replaces each run of image placeholder tokens with the next run of merged image features,
    /// in order. This is the `masked_scatter` of the reference `LlavaOnevision2Model.forward`.
    func spliceImageFeatures(
        _ image: LMInput.ProcessedImage, tokens: MLXArray, embeddings: MLXArray
    ) throws -> MLXArray {
        guard let grid = image.frames, !grid.isEmpty else {
            throw SharpAILlavaOnevision2Error.missingImageGrid
        }
        let dtype = embeddings.dtype
        let pixels = image.pixels.reshaped(
            -1,
            configuration.visionConfiguration.numChannels
                * configuration.visionConfiguration.patchSize
                * configuration.visionConfiguration.patchSize
        ).asType(dtype)
        let features = try model.visual(pixels, grid: grid).asType(dtype)

        let ids = tokens[0].asType(.int32).asArray(Int32.self)
        let imageId = Int32(configuration.imageTokenId)
        let placeholders = ids.reduce(0) { $0 + ($1 == imageId ? 1 : 0) }
        guard placeholders == features.dim(0) else {
            throw SharpAILlavaOnevision2Error.imageTokenMismatch(
                placeholders: placeholders, features: features.dim(0))
        }

        var pieces = [MLXArray]()
        var featureCursor = 0
        var i = 0
        while i < ids.count {
            var j = i
            if ids[i] == imageId {
                while j < ids.count, ids[j] == imageId { j += 1 }
                pieces.append(features[featureCursor ..< (featureCursor + (j - i))])
                featureCursor += j - i
            } else {
                while j < ids.count, ids[j] != imageId { j += 1 }
                pieces.append(embeddings[0, i ..< j])
            }
            i = j
        }
        return concatenated(pieces, axis: 0).expandedDimensions(axis: 0)
    }

    /// Prefill the prompt into `cache` and return logits for the whole prompt.
    ///
    /// Image input is spliced into the placeholder tokens before the decoder runs. Video throws.
    /// Batch size must be 1.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.video == nil, input.audio == nil else {
            throw SharpAILlavaOnevision2Error.videoNotImplemented
        }
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        guard tokens.dim(0) == 1 else {
            throw SharpAILlavaOnevision2Error.batchSizeNotSupported(tokens.dim(0))
        }
        let totalTokens = tokens.dim(1)
        guard totalTokens > 0 else {
            throw SharpAILlavaOnevision2Error.emptyPrompt
        }

        var embeddings = model.embedTokens(tokens)
        if let image = input.image {
            embeddings = try spliceImageFeatures(image, tokens: tokens, embeddings: embeddings)
        }

        let logits = lmHead(model(embeddings: embeddings, cache: cache))
        prefill.progress?(totalTokens, totalTokens)
        return .logits(LMOutput(logits: logits))
    }

    /// Text-only logits for token ids `[B, L]`.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(embeddings: model.embedTokens(inputs), cache: cache))
    }

    /// Maps checkpoint names to parameter paths and rejects duplicates.
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAILlavaOnevision2Sanitizer.sanitize(weights)
    }
}
