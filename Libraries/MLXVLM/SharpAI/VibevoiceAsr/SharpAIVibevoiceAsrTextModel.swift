// SharpAI fork overlay: vibevoice_asr text decoder (Qwen2 layout). Not part of upstream mlx-swift-lm.
//
// Written here, not reused, because the upstream Qwen2 types do not accept input embeddings.
// The module keys match Qwen2ForCausalLM: model.embed_tokens, model.layers.N, model.norm, lm_head.
// The decoder takes embeddings, so audio features can replace the placeholder rows first.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIVibevoiceAsrTextAttention: Module {

    let heads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPELayer

    init(_ config: SharpAIVibevoiceAsrTextConfiguration) {
        let headDim = config.headDim
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        self.heads = heads
        self.kvHeads = kvHeads
        self.scale = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: true)
        self._kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: true)
        self._vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: true)
        self._oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)

        self.rope = initializeRope(
            dims: headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)

        var queries = qProj(x).reshaped(batch, length, heads, -1).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(batch, length, kvHeads, -1).transposed(0, 2, 1, 3)
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

final class SharpAIVibevoiceAsrTextMLP: Module {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        self._up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAIVibevoiceAsrTextLayer: Module {

    @ModuleInfo(key: "self_attn") var attention: SharpAIVibevoiceAsrTextAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIVibevoiceAsrTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIVibevoiceAsrTextConfiguration) {
        self._attention.wrappedValue = SharpAIVibevoiceAsrTextAttention(config)
        self._mlp.wrappedValue = SharpAIVibevoiceAsrTextMLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// The `model.` part of Qwen2ForCausalLM.
final class SharpAIVibevoiceAsrTextModel: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [SharpAIVibevoiceAsrTextLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SharpAIVibevoiceAsrTextConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            SharpAIVibevoiceAsrTextLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

/// `language_model`: the Qwen2 model plus its output head.
final class SharpAIVibevoiceAsrCausalLM: Module {

    @ModuleInfo(key: "model") var model: SharpAIVibevoiceAsrTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIVibevoiceAsrTextConfiguration) {
        self._model.wrappedValue = SharpAIVibevoiceAsrTextModel(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    func logits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }
}
