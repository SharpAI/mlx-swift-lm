//
//  SharpAIJais2Model.swift
//  MLXLLM
//
//  Jais 2 model body (Jais2ForCausalLM). SharpAI fork overlay. Not part of upstream.
//
//  Port of Hugging Face transformers `modular_jais2.py` (main branch):
//    - Jais2Attention is LlamaAttention: q/k/v/o with bias=attention_bias, GQA,
//      full-width RoPE (rotate_half), no QK-norm.
//    - Jais2MLP is NemotronMLP: up_proj -> relu^2 -> down_proj, no gate, bias=mlp_bias.
//    - Jais2DecoderLayer uses nn.LayerNorm(eps=layer_norm_eps) for both norms,
//      pre-norm residuals.
//    - Jais2Model.norm is a final LayerNorm.
//    - lm_head is untied unless tie_word_embeddings is true.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Squared ReLU, transformers `ReLUSquaredActivation` ("relu2").
private func sharpAIJais2ReLUSquared(_ x: MLXArray) -> MLXArray {
    square(relu(x))
}

final class SharpAIJais2Attention: Module {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ config: SharpAIJais2Configuration) {
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.headDim

        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(
            config.hiddenSize, heads * headDim, bias: config.attentionBias)
        self._wk.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        self._wv.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        self._wo.wrappedValue = Linear(
            heads * headDim, config.hiddenSize, bias: config.attentionBias)

        self.rope = initializeRope(
            dims: headDim, base: config.ropeTheta, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

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
        .reshaped(B, L, -1)

        return wo(output)
    }
}

final class SharpAIJais2MLP: Module, UnaryLayer {

    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: SharpAIJais2Configuration) {
        self._up.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self._down.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(sharpAIJais2ReLUSquared(up(x)))
    }
}

final class SharpAIJais2DecoderLayer: Module {

    @ModuleInfo(key: "self_attn") var attention: SharpAIJais2Attention
    @ModuleInfo(key: "mlp") var mlp: SharpAIJais2MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm

    init(_ config: SharpAIJais2Configuration) {
        self._attention.wrappedValue = SharpAIJais2Attention(config)
        self._mlp.wrappedValue = SharpAIJais2MLP(config)
        self._inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._postAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

final class SharpAIJais2ModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [SharpAIJais2DecoderLayer]
    let norm: LayerNorm

    init(_ config: SharpAIJais2Configuration) {
        precondition(config.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIJais2DecoderLayer(config)
        }
        self.norm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

/// Causal LM for `jais2`.
final class SharpAIJais2Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {

    let vocabularySize: Int
    let kvHeads: [Int]

    let model: SharpAIJais2ModelInner

    let configuration: SharpAIJais2Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIJais2Configuration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SharpAIJais2ModelInner(config)
        self.configuration = config
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIJais2Weights.sanitize(
            weights: weights, tieWordEmbeddings: configuration.tieWordEmbeddings)
    }

    var loraLayers: [Module] {
        model.layers
    }
}
