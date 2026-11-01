//
//  SharpAIStep5Model.swift
//  MLXLLM
//
//  Text model for `step5` (Step5ForCausalLM). Not part of upstream mlx-swift-lm.
//  Ported from stepfun-ai/Step-5-Preview-BF16 inference/model.py (sha256 330a1668...), the
//  single-device, unfused path. Every layer here is a sliding-window layer, because the
//  reference's dense path only covers sliding layers. Full-attention (DSA sparse) layers are
//  refused by validateSupportedArchitecture before this type is built.
//
//  Per layer, as in the reference:
//    - attention: QK zero-centred RMSNorm, then partial NEOX RoPE over the leading
//      rotary span, causal sliding window (q - k < window), per-head sigmoid gate applied
//      before o_proj.
//    - FFN: dense clamped SwiGLU, or sigmoid MoE over stacked experts plus a shared expert.
//      The clamp is silu(gate) <= limit and up in [-limit, limit].
//
//  Known differences from the reference, which must not be read as parity:
//    - The residual stream runs in the model dtype, not the reference's fp32 residual.
//    - The reference's fp8/TP/EP/Triton kernels are not used. This is plain MLX.
//    - No numeric oracle was run. The test checks shape, finiteness and causality only.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Zero-centred RMSNorm: `x * rsqrt(mean(x^2)) * (1 + weight)`. The checkpoint centres
/// its weights on zero, so the `1 +` is required.
class SharpAIStep5RMSNorm: Module {
    let eps: Float

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1.0 + weight, eps: eps)
    }
}

/// Clamped SwiGLU MLP. Used for dense layers and for each MoE layer's shared expert.
class SharpAIStep5MLP: Module, UnaryLayer {
    let limit: Float

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int, limit: Float) {
        self.limit = limit
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // silu(gate) clamped above only; the lower bound is already -0.28 from silu.
        let activated = clip(silu(gateProj(x)), max: limit)
        let bounded = clip(upProj(x), min: -limit, max: limit)
        return downProj(activated * bounded)
    }
}

/// Self-attention for one sliding-window layer.
class SharpAIStep5Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIStep5RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIStep5RMSNorm

    init(_ config: SharpAIStep5Configuration, layer: Int) {
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.headDim

        self.numHeads = heads
        self.numKVHeads = kvHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)

        if config.useHeadWiseAttnGate {
            _gProj.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        } else {
            _gProj.wrappedValue = nil
        }

        _qNorm.wrappedValue = SharpAIStep5RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = SharpAIStep5RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        // Partial, NEOX-paired rotary over the leading span (traditional: false).
        // The reference's llama3 scaling is only accepted when its factor is 1.0, which is an
        // identity, so no scaling config is passed here.
        self.rope = initializeRope(
            dims: config.rotaryDimensions(layer: layer),
            base: config.ropeBase(layer: layer),
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(B, L, numHeads, headDim)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim)
        let values = vProj(x).reshaped(B, L, numKVHeads, headDim)

        // Norm before RoPE, as in the reference.
        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys).transposed(0, 2, 1, 3)
        let valuesHeadFirst = values.transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: valuesHeadFirst,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)

        // Per-head sigmoid gate, applied before o_proj.
        if let gProj {
            let gate = sigmoid(gProj(x)).asType(output.dtype)
            output = output * gate[.ellipsis, .newAxis]
        }

        return oProj(output.reshaped(B, L, -1))
    }
}

/// Sigmoid router in FP32. Scores are unbiased sigmoid values. The router bias only changes
/// which experts are selected.
class SharpAIStep5MoE: Module {
    let topK: Int
    let scalingFactor: Float
    let renormalize: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "router_bias") var routerBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ config: SharpAIStep5Configuration, layer: Int) {
        topK = config.moeTopK
        scalingFactor = config.moeRouterScalingFactor
        renormalize = config.normExpertWeight

        _gate.wrappedValue = Linear(config.hiddenSize, config.moeNumExperts, bias: false)
        _routerBias.wrappedValue = zeros([config.moeNumExperts])

        let limit = config.swigluLimit(layer: layer)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.moeNumExperts,
            activation: { x in clip(silu(x), max: limit) },
            activationProduct: { g, u in
                clip(silu(g), max: limit) * clip(u, min: -limit, max: limit)
            }
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = x.asType(.float32).matmul(gate.weight.asType(.float32).T)
        let scores = sigmoid(logits)
        let selection = scores + routerBias

        let inds = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        if renormalize {
            // The reference adds a 1e-20 guard here to keep the division finite.
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        weights = (weights * scalingFactor).asType(x.dtype)

        return (switchMLP(x, inds) * weights[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

/// One decoder layer: attention, then a dense MLP or MoE plus shared expert.
class SharpAIStep5DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIStep5Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: SharpAIStep5RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: SharpAIStep5RMSNorm
    @ModuleInfo(key: "mlp") var mlp: SharpAIStep5MLP?
    @ModuleInfo(key: "moe") var moe: SharpAIStep5MoE?
    @ModuleInfo(key: "share_expert") var shareExpert: SharpAIStep5MLP?

    let slidingWindow: Int?

    init(_ config: SharpAIStep5Configuration, layer: Int) {
        _attention.wrappedValue = SharpAIStep5Attention(config, layer: layer)
        _inputLayerNorm.wrappedValue = SharpAIStep5RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = SharpAIStep5RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.isMoELayer(layer) {
            _moe.wrappedValue = SharpAIStep5MoE(config, layer: layer)
            _shareExpert.wrappedValue = SharpAIStep5MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.shareExpertDim,
                limit: config.sharedSwigluLimit(layer: layer))
            _mlp.wrappedValue = nil
        } else {
            _mlp.wrappedValue = SharpAIStep5MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                limit: config.swigluLimit(layer: layer))
            _moe.wrappedValue = nil
            _shareExpert.wrappedValue = nil
        }

        self.slidingWindow = config.slidingWindowFor(layer)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        let normed = postAttentionLayerNorm(h)
        if let moe, let shareExpert {
            return h + moe(normed) + shareExpert(normed)
        }
        guard let mlp else {
            preconditionFailure("step5 layer has neither a dense MLP nor a MoE block")
        }
        return h + mlp(normed)
    }
}

class SharpAIStep5ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIStep5DecoderLayer]
    let norm: SharpAIStep5RMSNorm

    init(_ config: SharpAIStep5Configuration) {
        precondition(config.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAIStep5DecoderLayer(config, layer: $0)
        }
        self.norm = SharpAIStep5RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            // Each layer builds its own mask, using its own window.
            let layerCache = cache?[i]
            let mask = createAttentionMask(h: h, cache: layerCache, windowSize: layer.slidingWindow)
            h = layer(h, mask: mask, cache: layerCache)
        }
        return norm(h)
    }
}

class SharpAIStep5Model: Module, LLMModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    let kvHeads: [Int]
    let model: SharpAIStep5ModelInner
    let configuration: SharpAIStep5Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIStep5Configuration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SharpAIStep5ModelInner(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        } else {
            _lmHead.wrappedValue = nil
        }
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIStep5Sanitize.sanitize(weights: weights, configuration: configuration)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layers.map { (layer) throws -> KVCache in
            if let window = layer.slidingWindow {
                return try makeSlidingWindowKVCache(parameters: parameters, window: window)
            }
            return try makeAttentionKVCache(parameters: parameters)
        }
    }
}

extension SharpAIStep5Model: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
