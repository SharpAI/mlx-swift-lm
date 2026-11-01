// SharpAI fork overlay: jina_embeddings_v5 nano encoder (EuroBERT backbone). Not part of upstream.
//
// Source: jinaai/jina-embeddings-v5-text-nano modeling_eurobert.py (EuroBertModel) and config.json.
//   - Pre-norm decoder-style blocks: h = x + Attn(input_layernorm(x)); h = h + MLP(post_norm(h)).
//   - Attention is bidirectional (is_causal = False). Every position attends to the whole
//     sequence, with no causal triangle. Only padding is masked in the reference, and this
//     overlay takes no padding mask.
//   - RoPE on Q and K only (rotate_half style, traditional = false), scaling 1 / head_dim^-0.5.
//   - Gated SiLU MLP (gate_proj, up_proj, down_proj). RMSNorm without bias.
//
// There is no LM head in the encoder. callAsFunction returns the final hidden states, shape
// [B, L, hidden]. Embedding pooling (last token, then L2 normalization in the reference encode)
// is not implemented here. Like the Llama bidirectional overlay, padding is not masked.
// No KV cache: each call runs over the whole input.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIJinaEmbeddingsV5NanoAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    init(_ config: SharpAIJinaEmbeddingsV5NanoConfiguration) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
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
            traditional: false,
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

final class SharpAIJinaEmbeddingsV5NanoMLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: SharpAIJinaEmbeddingsV5NanoConfiguration) {
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

final class SharpAIJinaEmbeddingsV5NanoLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIJinaEmbeddingsV5NanoAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIJinaEmbeddingsV5NanoMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIJinaEmbeddingsV5NanoConfiguration) {
        self._attention.wrappedValue = SharpAIJinaEmbeddingsV5NanoAttention(config)
        self._mlp.wrappedValue = SharpAIJinaEmbeddingsV5NanoMLP(config)
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

final class SharpAIJinaEmbeddingsV5NanoInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [SharpAIJinaEmbeddingsV5NanoLayer]
    let norm: RMSNorm

    init(_ config: SharpAIJinaEmbeddingsV5NanoConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIJinaEmbeddingsV5NanoLayer(config)
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

/// Bidirectional EuroBERT encoder for the jina_embeddings_v5 nano variant.
final class SharpAIJinaEmbeddingsV5NanoModel: Module, LLMModel, LoRAModel {

    @ModuleInfo(key: "model") var model: SharpAIJinaEmbeddingsV5NanoInner

    let configuration: SharpAIJinaEmbeddingsV5NanoConfiguration

    init(_ configuration: SharpAIJinaEmbeddingsV5NanoConfiguration) {
        self.configuration = configuration
        self._model.wrappedValue = SharpAIJinaEmbeddingsV5NanoInner(configuration)
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
        throw SharpAIJinaEmbeddingsV5NanoError.generationUnsupported
    }

    func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        SharpAIJinaEmbeddingsV5NanoSanitizer.sanitize(weights)
    }

    var loraLayers: [Module] {
        model.layers
    }
}

/// Key mapping for the nano checkpoint. Keys without the leading `model.` get it added, and keys
/// with no module here are dropped: rotary buffers, lm_head, and adapter tensors (merged by
/// SharpAIJinaEmbeddingsV5Adapter before load).
enum SharpAIJinaEmbeddingsV5NanoSanitizer {

    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            guard !isDroppedKey(key) else { continue }
            sanitized[mappedKey(key)] = value
        }
        return sanitized
    }

    static func mappedKey(_ key: String) -> String {
        if key.hasPrefix("model.") {
            return key
        }
        if key.hasPrefix("embed_tokens.") || key.hasPrefix("layers.") || key.hasPrefix("norm.") {
            return "model." + key
        }
        return key
    }

    static func isDroppedKey(_ key: String) -> Bool {
        if key.contains("rotary_emb.inv_freq") {
            return true
        }
        return key.hasPrefix("lm_head.") || key.hasPrefix("model.lm_head.")
            || SharpAIJinaEmbeddingsV5Sanitize.isAdapterKey(key)
    }
}
