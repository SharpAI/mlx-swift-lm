// SharpAI fork overlay: llama_bidirec model. Not part of upstream mlx-swift-lm.
//
// Same module layout and weight names as LlamaModel (HF Llama), with two differences:
//   - Attention is non-causal. Every position attends to the whole sequence, with no mask.
//   - No KV cache. The forward pass runs over the full input in one call.
//
// There is no LM head. callAsFunction returns the final hidden states, shape [B, L, hidden].
//
// UNVERIFIED: the pooling used for embeddings (mean, CLS or last token), and any score head for
// the sequence-classification architecture, are not in this repo. They are not implemented here.
// Padding is not masked, so batched inputs with padding attend to pad tokens.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAILlamaBidirecAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    init(_ config: SharpAILlamaBidirecConfiguration) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.resolvedHeadDimensions
        scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(
            config.hiddenSize, heads * headDim, bias: config.attentionBias)
        self._wk.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        self._wv.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        self._wo.wrappedValue = Linear(
            heads * headDim, config.hiddenSize, bias: config.attentionBias)

        self.rope = initializeRope(
            dims: headDim, base: config.ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, offset: nil)
        keys = applyRotaryPosition(rope, to: keys, offset: nil)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: nil,
            scale: scale,
            mask: MLXFast.ScaledDotProductAttentionMaskMode.none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

final class SharpAILlamaBidirecMLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: SharpAILlamaBidirecConfiguration) {
        self._gate.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self._down.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
        self._up.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAILlamaBidirecLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAILlamaBidirecAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAILlamaBidirecMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAILlamaBidirecConfiguration) {
        self._attention.wrappedValue = SharpAILlamaBidirecAttention(config)
        self._mlp.wrappedValue = SharpAILlamaBidirecMLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(inputLayerNorm(x))
        return h + mlp(postAttentionLayerNorm(h))
    }
}

final class SharpAILlamaBidirecInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [SharpAILlamaBidirecLayer]
    let norm: RMSNorm

    init(_ config: SharpAILlamaBidirecConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAILlamaBidirecLayer(config)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        var h = embedTokens(inputs)
        for layer in layers {
            h = layer(h)
        }
        return norm(h)
    }
}

/// Bidirectional Llama encoder for `llama_bidirec`.
final class SharpAILlamaBidirecModel: Module, LLMModel, LoRAModel {

    @ModuleInfo(key: "model") var model: SharpAILlamaBidirecInner

    let configuration: SharpAILlamaBidirecConfiguration

    init(_ configuration: SharpAILlamaBidirecConfiguration) {
        self.configuration = configuration
        self._model.wrappedValue = SharpAILlamaBidirecInner(configuration)
    }

    /// Final hidden states for the whole input, shape [B, L, hidden]. No causal mask and no cache.
    func hiddenStates(_ inputs: MLXArray) -> MLXArray {
        model(inputs)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        hiddenStates(inputs)
    }

    /// The encoder has no cache, so each call sees the whole input.
    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        []
    }

    /// Prefill is not chunked. Chunks would hide later tokens from earlier positions.
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        throw SharpAILlamaBidirecError.generationUnsupported
    }

    func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        SharpAILlamaBidirecSanitizer.sanitize(weights)
    }

    var loraLayers: [Module] {
        model.layers
    }
}
