//
//  SharpAIStep1Model.swift
//  MLXVLM
//
//  Text model for `step1` (Step1ForCausalLM). Ported from stepfun-ai/Step-Audio-EditX
//  modeling_step1.py. Not part of upstream.
//  Layers: RMSNorm, grouped-query attention (num_attention_groups KV heads), SwiGLU MLP.
//  There is no rotary embedding. Position comes only from the sqrt ALiBi bias
//  (build_alibi_cache), which also applies the causal mask.
//  The reference flash-attention branch needs torch.ops.Optimus and is not ported. The
//  ALiBi branch is the one the reference runs without that kernel.
//  The audio tokenizer and audio tower are not part of this model.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// ALiBi slopes and the sqrt position bias, as in `build_alibi_cache`.
public enum SharpAIStep1ALiBi {

    /// Per-head slopes. With n the largest power of two not above `heads`, the first n heads
    /// use 2^(-8k/n) for k = 1...n. Any extra heads use 2^(-4(2j+1)/n) for j = 0...(heads-n-1).
    public static func slopes(heads: Int) -> [Float] {
        precondition(heads > 0, "heads must be positive")
        let n = 1 << Int(floor(log2(Double(heads))))
        let m0 = pow(2.0, -8.0 / Double(n))
        var values = (1 ... n).map { pow(m0, Double($0)) }
        if n < heads {
            let m1 = pow(2.0, -4.0 / Double(n))
            values += stride(from: 1, to: 2 * (heads - n) + 1, by: 2).map { pow(m1, Double($0)) }
        }
        return values.map { Float($0) }
    }

    /// Additive attention bias of shape [1, heads, queryLength, keyLength] for the last
    /// `queryLength` positions of a `keyLength` sequence. Entry (i, j) is -slope * sqrt(i - j)
    /// for j <= i, and -inf for j > i.
    public static func bias(
        queryLength: Int, keyLength: Int, heads: Int, dtype: DType
    ) -> MLXArray {
        precondition(
            queryLength > 0 && keyLength >= queryLength,
            "need 0 < queryLength <= keyLength")
        let queryPositions = MLXArray(
            (keyLength - queryLength ..< keyLength).map { Float($0) }
        ).reshaped([queryLength, 1])
        let keyPositions = MLXArray((0 ..< keyLength).map { Float($0) }).reshaped([1, keyLength])
        let distance = queryPositions - keyPositions
        let slopeValues = MLXArray(slopes(heads: heads)).reshaped([heads, 1, 1])
        let penalty =
            MLX.sqrt(MLX.maximum(distance, MLXArray(Float(0)))) * slopeValues
        let visible = distance .>= MLXArray(Float(0))
        let bias = MLX.where(
            visible, MLXArray(Float(0)) - penalty, MLXArray(-Float.infinity))
        return bias.reshaped([1, heads, queryLength, keyLength]).asType(dtype)
    }
}

final class SharpAIStep1Attention: Module {
    let heads: Int
    let groups: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ configuration: SharpAIStep1Configuration) {
        let hidden = configuration.hiddenSize
        let headCount = configuration.attentionHeads
        let groupCount = configuration.kvHeads
        let dim = configuration.headDim
        self.heads = headCount
        self.groups = groupCount
        self.headDim = dim
        self.scale = pow(Float(dim), -0.5)

        self._qProj.wrappedValue = Linear(hidden, headCount * dim, bias: false)
        self._kProj.wrappedValue = Linear(hidden, groupCount * dim, bias: false)
        self._vProj.wrappedValue = Linear(hidden, groupCount * dim, bias: false)
        self._oProj.wrappedValue = Linear(headCount * dim, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)

        let queries = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(batch, length, groups, headDim).transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped(batch, length, groups, headDim).transposed(0, 2, 1, 3)

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // Keys cover every cached position, so the queries are the last `length` of them.
        let bias = SharpAIStep1ALiBi.bias(
            queryLength: length, keyLength: keys.dim(2), heads: heads, dtype: queries.dtype)

        let repeats = heads / groups
        if repeats > 1 {
            keys = MLX.repeated(keys, count: repeats, axis: 1)
            values = MLX.repeated(values, count: repeats, axis: 1)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: nil,
            scale: scale,
            mask: .array(bias))

        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class SharpAIStep1MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAIStep1DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIStep1Attention
    @ModuleInfo(key: "mlp") var mlp: SharpAIStep1MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ configuration: SharpAIStep1Configuration) {
        self._selfAttn.wrappedValue = SharpAIStep1Attention(configuration)
        self._mlp.wrappedValue = SharpAIStep1MLP(
            hiddenSize: configuration.hiddenSize,
            intermediateSize: configuration.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

final class SharpAIStep1ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [SharpAIStep1DecoderLayer]
    let norm: RMSNorm

    init(_ configuration: SharpAIStep1Configuration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize, dimensions: configuration.hiddenSize)
        self.layers = (0 ..< configuration.hiddenLayers).map { _ in
            SharpAIStep1DecoderLayer(configuration)
        }
        self.norm = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            h = layer(h, cache: cache?[i])
        }
        return norm(h)
    }
}

/// Causal LM for `step1`. Hub key layout: model.*, and lm_head.* when embeddings are untied.
final class SharpAIStep1Model: Module, LanguageModel, KVCacheDimensionProvider {
    let configuration: SharpAIStep1Configuration
    let kvHeads: [Int]
    let model: SharpAIStep1ModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ configuration: SharpAIStep1Configuration) {
        self.configuration = configuration
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        self.model = SharpAIStep1ModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    /// Evaluates the prompt into the cache in chunks, leaving the last position for the
    /// `TokenIterator`'s first forward. Prompts that fit in one chunk go to the iterator whole.
    /// Mirrors the default `LLMModel.prepare`, minus its MLXLLM-only progress hook: progress
    /// is reported through `prefill.progress` by `forEachChunk`.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let stepSize = prefill.resolvedStepSize()
        let y = input.text
        let total = y.tokens.size

        guard total > stepSize else { return .tokens(y) }

        var processed = 0
        try withPreparedCache(cache, lengths: y.sequenceLengths) {
            processed = try prefill.forEachChunk(
                total: total, reserving: prefill.chunking == .remainder ? stepSize : 1
            ) { range in
                let chunk = y[.newAxis, range]
                _ = callAsFunction(chunk.tokens, cache: cache.isEmpty ? nil : cache)
                asyncEval(cache)
            }
            if processed > 0 {
                eval(cache)
            }
        }

        return .tokens(y[processed...])
    }

    func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        SharpAIStep1Sanitize.sanitize(weights: weights)
    }
}
