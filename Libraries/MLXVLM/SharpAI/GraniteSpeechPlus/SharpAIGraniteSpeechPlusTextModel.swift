// SharpAI fork overlay: granite_speech_plus text decoder. Not part of upstream mlx-swift-lm.
//
// The GraniteModel in MLXLLM takes token ids only. Audio features must enter as embeddings, so the
// decoder is rebuilt here with an embedding entry point. Its math follows the transformers Granite
// code: attention scaled by attention_multiplier, residual branches scaled by residual_multiplier,
// embeddings scaled by embedding_multiplier, logits divided by logits_scaling.
// Verified against: https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite/modeling_granite.py
// (GraniteAttention.scaling, GraniteDecoderLayer residual, GraniteModel embedding multiplier,
// GraniteForCausalLM logits scaling).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIGraniteSpeechPlusAttention: Module {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIGraniteSpeechPlusTextConfiguration) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        let headDim = config.hiddenSize / config.attentionHeads
        self.headDim = headDim
        self.scale = config.attentionMultiplier
        let bias = config.attentionBias
        self._qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * headDim, bias: bias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, config.kvHeads * headDim, bias: bias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, config.kvHeads * headDim, bias: bias)
        self._oProj.wrappedValue = Linear(
            config.attentionHeads * headDim, config.hiddenSize, bias: bias)
        self.rope = initializeRope(
            dims: headDim, base: config.ropeTheta, traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)
        let out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

/// Gated MLP: down(silu(gate(x)) * up(x)).
final class SharpAIGraniteSpeechPlusMLP: Module {

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: SharpAIGraniteSpeechPlusTextConfiguration) {
        let bias = config.mlpBias
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: bias)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: bias)
        self._downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class SharpAIGraniteSpeechPlusDecoderLayer: Module {

    let residualMultiplier: Float

    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIGraniteSpeechPlusAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIGraniteSpeechPlusMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIGraniteSpeechPlusTextConfiguration) {
        self.residualMultiplier = config.residualMultiplier
        self._selfAttn.wrappedValue = SharpAIGraniteSpeechPlusAttention(config)
        self._mlp.wrappedValue = SharpAIGraniteSpeechPlusMLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache) * residualMultiplier
        h = h + mlp(postAttentionLayerNorm(h)) * residualMultiplier
        return h
    }
}

/// The `model.` part of the text stack: embeddings, decoder layers, final norm.
final class SharpAIGraniteSpeechPlusTextBackbone: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [SharpAIGraniteSpeechPlusDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SharpAIGraniteSpeechPlusTextConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            SharpAIGraniteSpeechPlusDecoderLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
}

/// The text stack under the `language_model.` prefix: `model.*` plus an optional `lm_head`.
/// With tied embeddings there is no `lm_head`, and the embedding matrix is reused.
final class SharpAIGraniteSpeechPlusLanguageModel: Module {

    let embeddingMultiplier: Float
    let logitsScaling: Float

    @ModuleInfo(key: "model") var model: SharpAIGraniteSpeechPlusTextBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIGraniteSpeechPlusTextConfiguration) {
        self.embeddingMultiplier = config.embeddingMultiplier
        self.logitsScaling = config.logitsScaling
        self._model.wrappedValue = SharpAIGraniteSpeechPlusTextBackbone(config)
        if config.tieWordEmbeddings {
            self._lmHead.wrappedValue = nil
        } else {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    /// Runs the decoder on input embeddings (audio already merged in) and returns logits.
    /// The embedding multiplier is applied here, to the merged embeddings, as HF does.
    func logits(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings * embeddingMultiplier
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (index, layer) in model.layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[index])
        }
        h = model.norm(h)
        let raw: MLXArray
        if let head = lmHead {
            raw = head(h)
        } else {
            raw = model.embedTokens.asLinear(h)
        }
        return raw / logitsScaling
    }
}
