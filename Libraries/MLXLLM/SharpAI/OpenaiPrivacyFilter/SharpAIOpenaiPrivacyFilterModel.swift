// SharpAI fork overlay: openai_privacy_filter token classifier. Not part of upstream mlx-swift-lm.
//
// A gpt-oss style encoder, following transformers modeling_openai_privacy_filter.py:
// - pre-norm RMSNorm layers, no KV cache, every layer runs the same bidirectional band
//   |i - j| <= sliding_window with padding keys removed;
// - attention with one learned sink logit per head, dropped after the softmax;
// - interleaved (GPT-J style) RoPE with YaRN frequencies;
// - top-k MoE with a clamped SwiGLU; gate and up are the two halves of gate_up_proj;
// - a score Linear over the BIO-tagged labels, so the model returns per-token logits.
//
// Module keys match the HF names. sanitize only renames the expert tensors.
//
// UNVERIFIED: this has not been compiled or run. The bool mask passed together with sinks to
// MLXFast.scaledDotProductAttention is inferred from the repo's sink path and the Modernbert
// bool mask. Confirm it against a real run before trusting outputs.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum SharpAIOpenaiPrivacyFilterMasks {

    /// Bool mask [1, 1, L, L]. True where |i - j| <= window.
    static func slidingWindow(length: Int, window: Int) -> MLXArray {
        let positions = MLXArray((0 ..< length).map { Int32($0) })
        let distance = (positions.reshaped(length, 1) - positions.reshaped(1, length)).abs()
        return (distance .<= Int32(window)).reshaped(1, 1, length, length)
    }
}

/// Indices of the top k logits along the last axis.
private func sharpAIPrivacyFilterTopKIndices(_ logits: MLXArray, k: Int) -> MLXArray {
    let partitioned = argPartition(logits, kth: -k, axis: -1)
    return partitioned[.ellipsis, (-k)...]
}

final class SharpAIOpenaiPrivacyFilterAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    /// YaRN attention factor times head_dim^-0.25. Applied to q and k, so logits scale by head_dim^-0.5.
    let qkScale: Float
    /// Underscore prefix keeps this out of the parameter tree, so verify: .all does not expect a checkpoint key.
    let _ropeFreqs: MLXArray

    @ParameterInfo(key: "sinks") var sinks: MLXArray
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(
        _ config: SharpAIOpenaiPrivacyFilterConfiguration, ropeFreqs: MLXArray,
        attentionFactor: Float
    ) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDim = config.headDimensions
        self.qkScale = attentionFactor * pow(Float(config.headDimensions), -0.25)
        self._ropeFreqs = ropeFreqs

        self._sinks.wrappedValue = MLXArray.zeros([config.attentionHeads])
        self._qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDimensions,
            bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDimensions,
            bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDimensions,
            bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDimensions, config.hiddenSize,
            bias: config.attentionBias)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var q = qProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        q = MLXFast.RoPE(
            q, dimensions: headDim, traditional: true, base: nil, scale: 1.0, offset: 0,
            freqs: _ropeFreqs) * qkScale
        k = MLXFast.RoPE(
            k, dimensions: headDim, traditional: true, base: nil, scale: 1.0, offset: 0,
            freqs: _ropeFreqs) * qkScale

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: mask, sinks: sinks)

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

/// Fused gate_up and down expert projections with the reference clamped SwiGLU.
final class SharpAIOpenaiPrivacyFilterExperts: Module {
    let alpha: Float = 1.702
    let limit: Float = 7.0

    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    init(_ config: SharpAIOpenaiPrivacyFilterConfiguration) {
        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: config.hiddenSize, outputDims: 2 * config.intermediateSize,
            numExperts: config.localExperts, bias: true)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: config.intermediateSize, outputDims: config.hiddenSize,
            numExperts: config.localExperts, bias: true)
    }

    /// x: [B, L, hidden], indices: [B, L, k]. Returns [B, L, k, hidden].
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
        let gateUp = gateUpProj(expanded, indices).squeezed(axis: -2)

        // UNVERIFIED: contiguous halves are assumed. GPT-OSS in this repo uses interleaved channels (stride 2). Check against the checkpoint.
        let halves = gateUp.split(parts: 2, axis: -1)
        let gate = clip(halves[0], max: MLXArray(limit))
        let up = clip(halves[1], min: MLXArray(-limit), max: MLXArray(limit))
        let glu = gate * sigmoid(alpha * gate)
        let activated = (up + 1) * glu

        return downProj(MLX.expandedDimensions(activated, axis: -2), indices)
            .squeezed(axis: -2)
    }
}

final class SharpAIOpenaiPrivacyFilterMLP: Module {
    let topK: Int

    @ModuleInfo(key: "router") var router: Linear
    @ModuleInfo(key: "experts") var experts: SharpAIOpenaiPrivacyFilterExperts

    init(_ config: SharpAIOpenaiPrivacyFilterConfiguration) {
        self.topK = config.expertsPerToken
        self._router.wrappedValue = Linear(
            config.hiddenSize, config.localExperts, bias: true)
        self._experts.wrappedValue = SharpAIOpenaiPrivacyFilterExperts(config)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // The router runs in fp32, as in the reference.
        let logits = router(x.asType(.float32))
        let indices = sharpAIPrivacyFilterTopKIndices(logits, k: topK)
        let values = takeAlong(logits, indices, axis: -1)

        // The reference divides the top-k softmax by top_k and multiplies the sum back by top_k.
        // The two factors cancel, so the weights are the plain softmax over the top-k logits.
        let weights = softmax(values, axis: -1, precise: true)
        let expertOutputs = experts(x, indices)

        return (expertOutputs * expandedDimensions(weights, axis: -1))
            .sum(axis: -2)
            .asType(x.dtype)
    }
}

final class SharpAIOpenaiPrivacyFilterLayer: Module {

    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIOpenaiPrivacyFilterAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIOpenaiPrivacyFilterMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(
        _ config: SharpAIOpenaiPrivacyFilterConfiguration, ropeFreqs: MLXArray,
        attentionFactor: Float
    ) {
        self._selfAttn.wrappedValue = SharpAIOpenaiPrivacyFilterAttention(
            config, ropeFreqs: ropeFreqs, attentionFactor: attentionFactor)
        self._mlp.wrappedValue = SharpAIOpenaiPrivacyFilterMLP(config)
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x), mask: mask)
        return h + mlp(postAttentionLayernorm(h))
    }
}

final class SharpAIOpenaiPrivacyFilterInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [SharpAIOpenaiPrivacyFilterLayer]
    let slidingWindow: Int

    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SharpAIOpenaiPrivacyFilterConfiguration) {
        let rope = SharpAIOpenaiPrivacyFilterRope.yarn(config)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIOpenaiPrivacyFilterLayer(
                config, ropeFreqs: rope.freqs, attentionFactor: rope.attentionFactor)
        }
        self.slidingWindow = config.slidingWindow
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// `padding` is an optional Bool mask [B, 1, 1, L]. True marks a real token.
    func callAsFunction(_ inputs: MLXArray, padding: MLXArray?) -> MLXArray {
        var h = embedTokens(inputs)
        let length = h.dim(1)

        let band = SharpAIOpenaiPrivacyFilterMasks.slidingWindow(
            length: length, window: slidingWindow)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let padding {
            mask = .array(band & padding)
        } else {
            mask = .array(band)
        }

        for layer in layers {
            h = layer(h, mask: mask)
        }
        return norm(h)
    }
}

/// Bidirectional sliding-window token classifier for `openai_privacy_filter`.
final class SharpAIOpenaiPrivacyFilterModel: Module, LLMModel, LoRAModel {

    @ModuleInfo(key: "model") var model: SharpAIOpenaiPrivacyFilterInner
    @ModuleInfo(key: "score") var score: Linear

    let configuration: SharpAIOpenaiPrivacyFilterConfiguration

    init(_ configuration: SharpAIOpenaiPrivacyFilterConfiguration) {
        self.configuration = configuration
        self._model.wrappedValue = SharpAIOpenaiPrivacyFilterInner(configuration)
        self._score.wrappedValue = Linear(
            configuration.hiddenSize, configuration.numLabels, bias: true)
    }

    /// Per-token label logits, shape [B, L, numLabels]. `attentionMask` is [B, L] with 1 for real
    /// tokens and 0 for padding.
    func tokenLogits(_ inputs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let padding = attentionMask.map { mask in
            mask.asType(.bool).reshaped(inputs.dim(0), 1, 1, inputs.dim(1))
        }
        return score(model(inputs, padding: padding))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        tokenLogits(inputs)
    }

    /// The encoder has no cache. Each call sees the whole input.
    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        []
    }

    /// Token classification has no autoregressive decode, so prepare refuses.
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        throw SharpAIOpenaiPrivacyFilterError.tokenClassifierOnly
    }

    func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        SharpAIOpenaiPrivacyFilterSanitizer.sanitize(weights)
    }

    var loraLayers: [Module] {
        model.layers
    }
}
