// SharpAI fork overlay: muse_glimmer_text layers. Not part of upstream mlx-swift-lm.
//
// Text tower of the Muse-Glimmer VLM, ported from Libraries/MLXVLM/Models/MuseGlimmer.swift.
// Keep the arithmetic and module keys in step with that file. The eager ops here match
// the reference path; the VLM's compiled fusions are skipped.
//
// Non-standard pieces:
//   - Centered RMSNorm scales by `1 + weight`. The block norms use two eps values.
//   - Full-attention layers have no RoPE (theta 0). Sliding layers use the base theta.
//   - Attention output is gated: `output * sigmoid(gate_proj(x))`.
//   - Queries are scaled by `qk_scale_factor` in fp32 before RoPE, on top of `1/sqrt(d)`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// RMS norm with no learnable scale. Holds no `MLXArray`, so it adds no checkpoint keys.
class SharpAIMuseGlimmerTextNoScaleRMSNorm: Module, UnaryLayer {
    let eps: Float

    init(eps: Float) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

/// RMS norm whose checkpoint scale is centered at zero, so the effective scale is `1 + weight`.
class SharpAIMuseGlimmerTextCenteredRMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        scaledNormalize(x)
    }

    /// `residual + self(x)`, used for the post-attention and post-MLP steps.
    func addedTo(_ residual: MLXArray, _ x: MLXArray) -> MLXArray {
        residual + scaledNormalize(x)
    }

    private func scaledNormalize(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let f = x.asType(.float32)
        let variance = mean(f * f, axis: -1, keepDims: true)
        var out = f * rsqrt(variance + eps)
        out = out * (1.0 + weight.asType(.float32))
        return out.asType(dtype)
    }
}

class SharpAIMuseGlimmerTextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: SharpAIMuseGlimmerTextConfiguration) {
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(MLXNN.silu(gateProj(x)) * upProj(x))
    }
}

class SharpAIMuseGlimmerTextAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    /// Output gate. Same checkpoint name as `mlp.gate_proj`, but a different module.
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    @ModuleInfo(key: "qk_norm") var qkNorm: SharpAIMuseGlimmerTextNoScaleRMSNorm
    @ModuleInfo var rope: RoPELayer

    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float
    let qkScaleFactor: Float

    /// False on NoPE (full-attention) layers.
    let useRope: Bool

    init(_ config: SharpAIMuseGlimmerTextConfiguration, layerIndex: Int) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDimensions = config.headDimensions
        self.scale = pow(Float(config.headDimensions), -0.5)
        self.qkScaleFactor = config.qkScaleFactor

        let theta = config.layerRopeTheta[layerIndex]
        self.useRope = theta != 0

        let dim = config.hiddenSize
        let queryDim = config.attentionHeads * config.headDimensions
        let kvDim = config.kvHeads * config.headDimensions

        self._queryProj.wrappedValue = Linear(dim, queryDim, bias: config.attentionBias)
        self._keyProj.wrappedValue = Linear(dim, kvDim, bias: config.attentionBias)
        self._valueProj.wrappedValue = Linear(dim, kvDim, bias: config.attentionBias)
        self._outputProj.wrappedValue = Linear(queryDim, dim, bias: config.attentionBias)
        self._gateProj.wrappedValue = Linear(dim, queryDim, bias: false)
        self._qkNorm.wrappedValue = SharpAIMuseGlimmerTextNoScaleRMSNorm(eps: config.rmsNormEps)

        let effectiveTheta = self.useRope ? theta : config.ropeTheta
        self._rope.wrappedValue = initializeRope(
            dims: config.headDimensions,
            base: effectiveTheta,
            traditional: false,
            scalingConfig: [
                "rope_type": .string("default"),
                "rope_theta": .float(effectiveTheta),
            ],
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))

        var queries = queryProj(x).reshaped(batch, length, heads, headDimensions)
        var keys = keyProj(x).reshaped(batch, length, kvHeads, headDimensions)
        var values = valueProj(x).reshaped(batch, length, kvHeads, headDimensions)

        queries = qkNorm(queries)
        queries = (queries.asType(.float32) * qkScaleFactor)
            .asType(queries.dtype).transposed(0, 2, 1, 3)
        keys = qkNorm(keys).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        if useRope {
            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
        }

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        output = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)

        // The gate reads the layer input, not the attention output.
        output = output * sigmoid(gateProj(x))
        return outputProj(output)
    }
}

class SharpAIMuseGlimmerTextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: SharpAIMuseGlimmerTextAttention
    @ModuleInfo var mlp: SharpAIMuseGlimmerTextMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: SharpAIMuseGlimmerTextCenteredRMSNorm
    @ModuleInfo(key: "post_attention_layernorm")
    var postAttentionLayerNorm: SharpAIMuseGlimmerTextCenteredRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm")
    var preFeedforwardLayerNorm: SharpAIMuseGlimmerTextCenteredRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm")
    var postFeedforwardLayerNorm: SharpAIMuseGlimmerTextCenteredRMSNorm

    let isSliding: Bool

    init(_ config: SharpAIMuseGlimmerTextConfiguration, layerIndex: Int) {
        self._selfAttention.wrappedValue = SharpAIMuseGlimmerTextAttention(
            config, layerIndex: layerIndex)
        self._mlp.wrappedValue = SharpAIMuseGlimmerTextMLP(config)

        // Pre-norms use rms_norm_eps. Post-norms use the much smaller post_norm_eps.
        self._inputLayerNorm.wrappedValue = SharpAIMuseGlimmerTextCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = SharpAIMuseGlimmerTextCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.postNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = SharpAIMuseGlimmerTextCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = SharpAIMuseGlimmerTextCenteredRMSNorm(
            dimensions: config.hiddenSize, eps: config.postNormEps)

        self.isSliding = config.layerTypes[layerIndex] == "sliding_attention"
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var h = postAttentionLayerNorm.addedTo(
            x, selfAttention(inputLayerNorm(x), mask: mask, cache: cache))
        h = postFeedforwardLayerNorm.addedTo(h, mlp(preFeedforwardLayerNorm(h)))
        return h
    }
}

/// Text tower: embeddings, decoder layers, final norm. Keys match `model.*` in the checkpoint.
class SharpAIMuseGlimmerTextInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    /// Applied to token embeddings before layer 0. Parameterless.
    @ModuleInfo(key: "embed_norm") var embedNorm: SharpAIMuseGlimmerTextNoScaleRMSNorm

    @ModuleInfo var layers: [SharpAIMuseGlimmerTextDecoderLayer]

    /// The final norm is a plain RMSNorm (scale used directly).
    @ModuleInfo var norm: RMSNorm

    let config: SharpAIMuseGlimmerTextConfiguration
    private let fullAttentionIndex: Int
    private let slidingAttentionIndex: Int?

    init(_ config: SharpAIMuseGlimmerTextConfiguration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._embedNorm.wrappedValue = SharpAIMuseGlimmerTextNoScaleRMSNorm(
            eps: config.rmsNormEps)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            SharpAIMuseGlimmerTextDecoderLayer(config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        self.fullAttentionIndex = config.layerTypes.firstIndex(of: "full_attention") ?? 0
        self.slidingAttentionIndex = config.layerTypes.firstIndex(of: "sliding_attention")
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]?) -> MLXArray {
        var h = embedNorm(embedTokens(inputs))

        let layerCache = cache ?? Array(repeating: nil, count: layers.count)

        // Full-layer and sliding-layer masks, each built from the first cache of its kind.
        let fullMask = createAttentionMask(h: h, cache: layerCache[fullAttentionIndex])
        var slidingMask = MLXFast.ScaledDotProductAttentionMaskMode.none
        if let slidingAttentionIndex {
            slidingMask = createAttentionMask(
                h: h, cache: layerCache[slidingAttentionIndex],
                windowSize: config.slidingWindow)
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: layer.isSliding ? slidingMask : fullMask, cache: layerCache[i])
        }
        return norm(h)
    }
}
