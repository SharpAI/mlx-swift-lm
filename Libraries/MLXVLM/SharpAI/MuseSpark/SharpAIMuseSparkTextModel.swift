// SharpAI fork overlay: muse_spark text-only causal LM. Not part of upstream mlx-swift-lm.
//
// Layer math follows transformers modeling_muse_spark.py: MuseSparkTextModel,
// MuseSparkTextDecoderLayer, MuseSparkTextAttention, MuseSparkMoE and the text logit transform
// in MuseSparkForConditionalGeneration (output_multiplier, then tanh soft cap).
//
// Not registered with the factory. SharpAIMuseSparkOverlay still throws, because the vision and
// audio towers and the checkpoint key mapping are not written. This type is the text backbone only.
//
// The MoE evaluates every expert densely and zeroes unselected ones. The result matches the
// reference per-expert loop, but compute scales with num_local_experts, not num_experts_per_tok.
// No KV-cache factory is provided; callers pass nil or their own [KVCache].

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// x / sqrt(mean(x^2) + eps) over the last axis.
private func sharpAIMuseSparkNormalize(_ x: MLXArray, eps: Float) -> MLXArray {
    x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps)
}

/// Scale-free RMS norm computed in float32 (reference MuseSparkRMSNorm with_scale=False).
func sharpAIMuseSparkScaleFreeNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    sharpAIMuseSparkNormalize(x.asType(.float32), eps: eps).asType(x.dtype)
}

/// Norm times (1 + weight), weight initialized to zero (reference MuseSparkTextCenteredRMSNorm).
class SharpAIMuseSparkCenteredRMSNorm: Module {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = sharpAIMuseSparkNormalize(x.asType(.float32), eps: eps)
        return (normed * (weight.asType(.float32) + 1)).asType(x.dtype)
    }
}

/// Scale first, round to the input dtype, then normalize (reference MuseSparkTextInputScaledRMSNorm).
class SharpAIMuseSparkInputScaledRMSNorm: Module {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scaled = (x.asType(.float32) * (weight.asType(.float32) + 1)).asType(x.dtype)
        return sharpAIMuseSparkNormalize(scaled.asType(.float32), eps: eps).asType(x.dtype)
    }
}

/// Dense SiLU-gated MLP for layers whose mlp_layer_types entry is "dense".
class SharpAIMuseSparkMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Sigmoid router. The correction bias shifts selection only; weights use the unbiased scores
/// (reference MuseSparkTopkRouter).
class SharpAIMuseSparkRouter: Module {
    let topK: Int
    let outputMultiplier: Float
    let routedScalingFactor: Float
    let normEps: Float = 1e-15

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: SharpAIMuseSparkTextConfiguration) {
        let experts = config.numLocalExperts
        topK = config.numExpertsPerTok
        outputMultiplier = config.outputMultiplier
        routedScalingFactor = config.routedScalingFactor
        _weight.wrappedValue =
            MLXRandom.normal([experts, config.hiddenSize]) * config.initializerRange
        _eScoreCorrectionBias.wrappedValue = zeros([experts])
        super.init()
    }

    /// `x` is [N, hidden]. Returns expert indices [N, topK] and weights [N, topK].
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = x.asType(.float32).matmul(weight.T.asType(.float32)) * outputMultiplier
        let scores = sigmoid(logits)
        let choice = scores + eScoreCorrectionBias
        let inds = argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + normEps)
        return (inds, weights * routedScalingFactor)
    }
}

/// Scatter top-k weights into a dense [N, numExperts] matrix, zero for unselected experts.
private func sharpAIMuseSparkDenseRouting(
    indices: MLXArray, weights: MLXArray, numExperts: Int
) -> MLXArray {
    let expertIds = MLXArray((0 ..< numExperts).map { Int32($0) })
    let selected = indices.asType(.int32).expandedDimensions(axis: -1) .== expertIds
    return (selected.asType(weights.dtype) * weights.expandedDimensions(axis: -1)).sum(axis: 1)
}

/// Stacked routed experts (reference MuseSparkExperts). Evaluated densely over all experts.
class SharpAIMuseSparkExperts: Module {
    let numExperts: Int
    let intermediateDim: Int

    @ParameterInfo(key: "gate_up_proj") var gateUpProj: MLXArray
    @ParameterInfo(key: "down_proj") var downProj: MLXArray
    @ModuleInfo(key: "post_expert_norm") var postExpertNorm: SharpAIMuseSparkInputScaledRMSNorm

    init(_ config: SharpAIMuseSparkTextConfiguration) {
        let experts = config.numLocalExperts
        let intermediate = config.moeIntermediateSize
        let moeHidden = config.moeHiddenSize
        numExperts = experts
        intermediateDim = intermediate
        _gateUpProj.wrappedValue =
            MLXRandom.normal([experts, 2 * intermediate, moeHidden]) * config.initializerRange
        _downProj.wrappedValue =
            MLXRandom.normal([experts, moeHidden, intermediate]) * config.initializerRange
        _postExpertNorm.wrappedValue = SharpAIMuseSparkInputScaledRMSNorm(
            dimensions: moeHidden, eps: config.postNormEps)
        super.init()
    }

    /// `x` is [N, moeHidden]; `routing` is [N, numExperts]. Returns [N, moeHidden].
    func callAsFunction(_ x: MLXArray, routing: MLXArray) -> MLXArray {
        // [1, N, H] x [E, H, 2I] -> [E, N, 2I]; the first I columns are the gate, the rest the up.
        let fused = x.expandedDimensions(axis: 0).matmul(gateUpProj.transposed(0, 2, 1))
        let gate = fused[.ellipsis, ..<intermediateDim]
        let up = fused[.ellipsis, intermediateDim...]
        let hidden = silu(gate) * up
        // [E, N, I] x [E, I, H] -> [E, N, H]
        let down = hidden.matmul(downProj.transposed(0, 2, 1))
        let perExpert = postExpertNorm(down)
        // Weight each expert's output for each token, then sum over experts.
        let weights = routing.transposed(1, 0).expandedDimensions(axis: -1)
        return (perExpert * weights).sum(axis: 0)
    }
}

/// Sparse MoE block (reference MuseSparkMoE). The router reads the full-width hidden state;
/// the experts run in the reduced moe_hidden space.
class SharpAIMuseSparkMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: SharpAIMuseSparkRouter
    @ModuleInfo(key: "pre_expert_proj") var preExpertProj: Linear
    @ModuleInfo(key: "pre_expert_norm") var preExpertNorm: SharpAIMuseSparkCenteredRMSNorm
    @ModuleInfo(key: "experts") var experts: SharpAIMuseSparkExperts
    @ModuleInfo(key: "post_expert_proj") var postExpertProj: Linear

    init(_ config: SharpAIMuseSparkTextConfiguration) {
        _gate.wrappedValue = SharpAIMuseSparkRouter(config)
        _preExpertProj.wrappedValue = Linear(config.hiddenSize, config.moeHiddenSize, bias: false)
        _preExpertNorm.wrappedValue = SharpAIMuseSparkCenteredRMSNorm(
            dimensions: config.moeHiddenSize, eps: config.rmsNormEps)
        _experts.wrappedValue = SharpAIMuseSparkExperts(config)
        _postExpertProj.wrappedValue = Linear(config.moeHiddenSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let flat = x.reshaped(-1, shape[shape.count - 1])
        let (inds, weights) = gate(flat)
        let routing = sharpAIMuseSparkDenseRouting(
            indices: inds, weights: weights, numExperts: experts.numExperts)
        let hidden = experts(preExpertNorm(preExpertProj(flat)), routing: routing)
        return postExpertProj(hidden).reshaped(shape)
    }
}

/// Learned mix of the residual and the sublayer output (reference MuseSparkResidualGate):
/// alpha * residual + beta * hidden, with beta = sigmoid(w) and w = gate / temperature.
class SharpAIMuseSparkResidualGate: Module {
    let temperature: Float
    let eps: Float = 1e-3

    @ParameterInfo(key: "gate") var gate: MLXArray

    init(_ config: SharpAIMuseSparkTextConfiguration) {
        temperature = config.residualGateTemperature
        _gate.wrappedValue = MLXRandom.normal([config.hiddenSize]) * config.initializerRange
        super.init()
    }

    func callAsFunction(_ residual: MLXArray, _ hidden: MLXArray) -> MLXArray {
        let w = gate / temperature
        let beta = sigmoid(w)
        let alpha = sqrt(maximum(sigmoid(-w) * (beta + 1), MLXArray(eps)))
        return (alpha * residual + beta * hidden).asType(residual.dtype)
    }
}

/// GQA attention with scale-free qk-norm, a qk scale factor, a per-head scale-free output norm
/// and a sigmoid output gate (reference MuseSparkTextAttention). NoPE layers skip RoPE.
class SharpAIMuseSparkAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let qkScaleFactor: Float
    let normEps: Float
    let rope: RoPE?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    init(_ config: SharpAIMuseSparkTextConfiguration, layerIndex: Int) {
        let hidden = config.hiddenSize
        let heads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let dim = config.headDim

        numHeads = heads
        numKVHeads = kvHeads
        headDim = dim
        scale = pow(Float(dim), -0.5)
        qkScaleFactor = config.qkScaleFactor
        normEps = config.rmsNormEps
        rope =
            config.layerRopeTheta[layerIndex] != 0
            ? RoPE(dimensions: dim, traditional: false, base: config.ropeTheta) : nil

        _qProj.wrappedValue = Linear(hidden, heads * dim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(hidden, kvHeads * dim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(hidden, kvHeads * dim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(heads * dim, hidden, bias: config.attentionBias)
        _gateProj.wrappedValue = Linear(hidden, heads * dim, bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        q = sharpAIMuseSparkScaleFreeNorm(q, eps: normEps) * qkScaleFactor
        k = sharpAIMuseSparkScaleFreeNorm(k, eps: normEps)
        if let rope {
            q = applyRotaryPosition(rope, to: q, cache: cache)
            k = applyRotaryPosition(rope, to: k, cache: cache)
        }

        let attended = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
        let perHead = sharpAIMuseSparkScaleFreeNorm(attended, eps: normEps)
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, numHeads * headDim)
        return oProj(perHead * sigmoid(gateProj(x)))
    }
}

/// One decoder layer (reference MuseSparkTextDecoderLayer): pre-norm attention and FFN, each
/// closed by a learned residual gate.
class SharpAIMuseSparkDecoderLayer: Module {
    let postNormEps: Float
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMuseSparkAttention
    let mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: SharpAIMuseSparkCenteredRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm:
        SharpAIMuseSparkCenteredRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm:
        SharpAIMuseSparkInputScaledRMSNorm
    @ModuleInfo(key: "post_attention_residual_gate") var postAttentionResidualGate:
        SharpAIMuseSparkResidualGate
    @ModuleInfo(key: "post_feedforward_residual_gate") var postFeedforwardResidualGate:
        SharpAIMuseSparkResidualGate

    init(_ config: SharpAIMuseSparkTextConfiguration, layerIndex: Int) {
        postNormEps = config.postNormEps
        _selfAttn.wrappedValue = SharpAIMuseSparkAttention(config, layerIndex: layerIndex)
        if config.mlpLayerTypes[layerIndex] == "sparse" {
            mlp = SharpAIMuseSparkMoE(config)
        } else {
            mlp = SharpAIMuseSparkMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
        _inputLayernorm.wrappedValue = SharpAIMuseSparkCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayernorm.wrappedValue = SharpAIMuseSparkCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayernorm.wrappedValue = SharpAIMuseSparkInputScaledRMSNorm(
            dimensions: config.hiddenSize, eps: config.postNormEps)
        _postAttentionResidualGate.wrappedValue = SharpAIMuseSparkResidualGate(config)
        _postFeedforwardResidualGate.wrappedValue = SharpAIMuseSparkResidualGate(config)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let attention = sharpAIMuseSparkScaleFreeNorm(
            selfAttn(inputLayernorm(x), mask: mask, cache: cache), eps: postNormEps)
        let mid = postAttentionResidualGate(x, attention)
        let feed = postFeedforwardLayernorm(mlp(preFeedforwardLayernorm(mid)))
        return postFeedforwardResidualGate(mid, feed)
    }
}

/// Embedding, decoder stack and final norm (reference MuseSparkTextModel).
class SharpAIMuseSparkTextBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIMuseSparkDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let layerTypes: [String]
    let slidingWindow: Int
    let embedNormEps: Float

    init(_ config: SharpAIMuseSparkTextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = (0 ..< config.numHiddenLayers).map {
            SharpAIMuseSparkDecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        layerTypes = config.layerTypes
        slidingWindow = config.slidingWindow
        embedNormEps = config.rmsNormEps
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Embedding lookup followed by a scale-free norm (reference MuseSparkTextNormedEmbedding).
        var h = sharpAIMuseSparkScaleFreeNorm(embedTokens(inputs), eps: embedNormEps)

        // Masks come from the first layer of each kind. A missing kind falls back to index 0,
        // and its mask is never used.
        let fullIndex = layerTypes.firstIndex(of: "full_attention") ?? 0
        let swaIndex = layerTypes.firstIndex(of: "sliding_attention") ?? 0
        let fullMask = createAttentionMask(h: h, cache: cache?[fullIndex])
        let swaMask = createAttentionMask(
            h: h, cache: cache?[swaIndex], windowSize: slidingWindow)

        for (i, layer) in layers.enumerated() {
            let mask = layerTypes[i] == "sliding_attention" ? swaMask : fullMask
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

/// Text-only muse_spark causal LM: backbone, lm_head, then the logit transform
/// T * tanh(logits * output_multiplier / T) from the reference.
class SharpAIMuseSparkTextForCausalLM: Module {
    let outputMultiplier: Float
    let finalLogitSoftcapping: Float

    @ModuleInfo(key: "model") var model: SharpAIMuseSparkTextBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ config: SharpAIMuseSparkTextConfiguration) {
        outputMultiplier = config.outputMultiplier
        finalLogitSoftcapping = config.finalLogitSoftcapping
        _model.wrappedValue = SharpAIMuseSparkTextBackbone(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        let logits = lmHead(hidden) * outputMultiplier
        return tanh(logits / finalLogitSoftcapping) * finalLogitSoftcapping
    }
}
