// SharpAI fork overlay: PaddleOCR-VL text decoder (ERNIE 4.5) with multimodal RoPE.
// Not part of upstream mlx-swift-lm.
//
// The upstream Ernie45 model applies 1-D RoPE and takes no per-axis position ids, so this
// decoder is written here. It takes embeddings and (3, batch, seq) position ids.
// The fixture sets rope_is_neox_style true, so the rotation is rotate-half, not interleaved.
// UNVERIFIED: the section split over the half-dim frequencies. Check it against HF logits.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Three-axis RoPE. Each section of the half-dim frequencies reads its position from one axis.
final class SharpAIPaddleocrVlMRoPE {
    let invFreq: MLXArray
    let mropeSection: [Int]

    init(headDim: Int, base: Float, mropeSection: [Int]) {
        self.mropeSection = mropeSection
        let p = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.int64).asType(.float32)
            / Float(headDim)
        self.invFreq = 1.0 / pow(base, p)
    }

    /// cos and sin of shape (batch, seq, headDim), cast to `dtype`.
    func cosSin(positionIds: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        // (3, batch, seq, headDim / 2)
        let positions = positionIds.asType(.float32).expandedDimensions(axis: -1)
        let freqs = positions * invFreq

        var sections = [MLXArray]()
        var start = 0
        for (index, width) in mropeSection.enumerated() {
            sections.append(freqs[index % 3, 0..., 0..., start ..< start + width])
            start += width
        }
        let merged = concatenated(sections, axis: -1)
        let emb = concatenated([merged, merged], axis: -1)
        return (MLX.cos(emb).asType(dtype), MLX.sin(emb).asType(dtype))
    }
}

private func sharpAIPaddleocrVlRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, 0 ..< half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

final class SharpAIPaddleocrVlTextAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIPaddleocrVlTextConfiguration) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)

        let dim = config.hiddenSize
        self._qProj.wrappedValue = Linear(dim, heads * headDim, bias: config.useBias)
        self._kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.useBias)
        self._vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.useBias)
        self._oProj.wrappedValue = Linear(heads * headDim, dim, bias: config.useBias)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)

        var queries = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

        // (batch, seq, headDim) -> (batch, 1, seq, headDim) for the head axis.
        let cosHeads = cos.expandedDimensions(axis: 1)
        let sinHeads = sin.expandedDimensions(axis: 1)
        queries = queries * cosHeads + sharpAIPaddleocrVlRotateHalf(queries) * sinHeads
        keys = keys * cosHeads + sharpAIPaddleocrVlRotateHalf(keys) * sinHeads

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

final class SharpAIPaddleocrVlTextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: SharpAIPaddleocrVlTextConfiguration) {
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.useBias)
        self._upProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.useBias)
        self._downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.useBias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class SharpAIPaddleocrVlTextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIPaddleocrVlTextAttention
    @ModuleInfo var mlp: SharpAIPaddleocrVlTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: SharpAIPaddleocrVlTextConfiguration) {
        self._selfAttn.wrappedValue = SharpAIPaddleocrVlTextAttention(config)
        self.mlp = SharpAIPaddleocrVlTextMLP(config)
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x), mask: mask, cache: cache, cos: cos, sin: sin)
        return h + mlp(postAttentionLayernorm(h))
    }
}

final class SharpAIPaddleocrVlTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [SharpAIPaddleocrVlTextDecoderLayer]
    let norm: RMSNorm
    let rotary: SharpAIPaddleocrVlMRoPE

    init(_ config: SharpAIPaddleocrVlTextConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIPaddleocrVlTextDecoderLayer(config)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.rotary = SharpAIPaddleocrVlMRoPE(
            headDim: config.headDim, base: config.ropeTheta,
            mropeSection: config.mropeSection)
        super.init()
    }

    /// `embeddings` is (batch, seq, hidden). `positionIds` is (3, batch, seq).
    func forward(_ embeddings: MLXArray, positionIds: MLXArray, cache: [KVCache]?) -> MLXArray {
        let (cos, sin) = rotary.cosSin(positionIds: positionIds, dtype: embeddings.dtype)
        let mask = createAttentionMask(h: embeddings, cache: cache?.first)

        var h = embeddings
        for (index, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[index], cos: cos, sin: sin)
        }
        return norm(h)
    }
}

final class SharpAIPaddleocrVlLanguageModel: Module, KVCacheDimensionProvider {
    @ModuleInfo var model: SharpAIPaddleocrVlTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let kvHeads: [Int]

    init(_ config: SharpAIPaddleocrVlTextConfiguration) {
        self.model = SharpAIPaddleocrVlTextModel(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        super.init()
    }

    func forward(_ embeddings: MLXArray, positionIds: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model.forward(embeddings, positionIds: positionIds, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }
}
