// SharpAI fork overlay: step3p5 text backbone used by step3p7 (`text_config`). Not part of upstream mlx-swift-lm.
//
// Port of the text path of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/step3p5.py.
// Module keys match the overlay sanitizer in Libraries/MLXVLM/SharpAI/Step3p7/SharpAIStep3p7Sanitize.swift:
// MoE layers use `moe` (gate, router_bias, switch_mlp, share_expert); dense layers use `mlp`.
// Norm weights are applied as stored. The sanitizer adds 1 to them for vanilla checkpoints, as mlx-lm does.
// Not included: the perception_encoder vision tower and projector, and the multi-token-prediction
// layers (the sanitizer drops them). No `newCache` or LoRA hooks yet.
// UNVERIFIED against a real step3p7 checkpoint. The tiny forward-pass test is the only check so far.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Text-backbone hyperparameters. Field names mirror the `text_config` keys of step3p5.
public struct SharpAIStep3p7TextArguments: Sendable {
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var vocabularySize: Int
    public var attentionHeads: Int
    public var attentionGroups: Int
    public var headDim: Int
    /// Head count for `sliding_attention` layers (`attention_other_setting.num_attention_heads`).
    public var slidingAttentionHeads: Int?
    /// Dense MLP width, used by layers that are not in `moeLayers`.
    public var intermediateSize: Int
    /// One of `full_attention` or `sliding_attention` per layer. Empty means all full attention.
    public var layerTypes: [String]
    public var slidingWindow: Int
    public var ropeTheta: Float
    /// Per-layer `rope_theta` list. Empty means `ropeTheta` for every layer.
    public var ropeThetaPerLayer: [Float]
    /// Per-layer fraction of the head dimension that gets rotary embedding. Empty means 1.0.
    public var partialRotaryFactors: [Float]
    public var ropeScaling: [String: StringOrNumber]?
    /// Layer types that receive `ropeScaling`. Empty means every layer type receives it.
    public var yarnOnlyTypes: [String]
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var useHeadWiseAttnGate: Bool
    /// Layer indices that use the MoE block. All other layers use the dense MLP.
    public var moeLayers: Set<Int>
    public var moeNumExperts: Int
    public var moeTopK: Int
    public var moeIntermediateSize: Int
    public var shareExpertDim: Int
    public var moeRouterScalingFactor: Float
    public var normExpertWeight: Bool
    /// Per-layer clamp for the routed-expert SwiGLU. 0 or missing means no clamp.
    public var swigluLimits: [Float]
    /// Per-layer clamp for the shared-expert (or dense) SwiGLU. 0 or missing means no clamp.
    public var swigluLimitsShared: [Float]

    public init(
        hiddenSize: Int,
        hiddenLayers: Int,
        vocabularySize: Int,
        attentionHeads: Int,
        attentionGroups: Int,
        headDim: Int,
        slidingAttentionHeads: Int? = nil,
        intermediateSize: Int = 0,
        layerTypes: [String] = [],
        slidingWindow: Int = 512,
        ropeTheta: Float = 10000,
        ropeThetaPerLayer: [Float] = [],
        partialRotaryFactors: [Float] = [],
        ropeScaling: [String: StringOrNumber]? = nil,
        yarnOnlyTypes: [String] = [],
        maxPositionEmbeddings: Int = 262144,
        rmsNormEps: Float = 1e-5,
        useHeadWiseAttnGate: Bool = true,
        moeLayers: Set<Int> = [],
        moeNumExperts: Int = 0,
        moeTopK: Int = 1,
        moeIntermediateSize: Int = 0,
        shareExpertDim: Int = 0,
        moeRouterScalingFactor: Float = 1,
        normExpertWeight: Bool = true,
        swigluLimits: [Float] = [],
        swigluLimitsShared: [Float] = []
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.vocabularySize = vocabularySize
        self.attentionHeads = attentionHeads
        self.attentionGroups = attentionGroups
        self.headDim = headDim
        self.slidingAttentionHeads = slidingAttentionHeads
        self.intermediateSize = intermediateSize
        self.layerTypes = layerTypes
        self.slidingWindow = slidingWindow
        self.ropeTheta = ropeTheta
        self.ropeThetaPerLayer = ropeThetaPerLayer
        self.partialRotaryFactors = partialRotaryFactors
        self.ropeScaling = ropeScaling
        self.yarnOnlyTypes = yarnOnlyTypes
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.useHeadWiseAttnGate = useHeadWiseAttnGate
        self.moeLayers = moeLayers
        self.moeNumExperts = moeNumExperts
        self.moeTopK = moeTopK
        self.moeIntermediateSize = moeIntermediateSize
        self.shareExpertDim = shareExpertDim
        self.moeRouterScalingFactor = moeRouterScalingFactor
        self.normExpertWeight = normExpertWeight
        self.swigluLimits = swigluLimits
        self.swigluLimitsShared = swigluLimitsShared
    }
}

/// Reads a per-layer array entry, or returns `fallback` when the array is short or empty.
private func perLayer<T>(_ values: [T], _ index: Int, _ fallback: T) -> T {
    values.indices.contains(index) ? values[index] : fallback
}

private func layerTypeAt(_ args: SharpAIStep3p7TextArguments, _ index: Int) -> String {
    perLayer(args.layerTypes, index, "full_attention")
}

// MARK: - Attention

final class SharpAIStep3p7Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    /// Head-wise output gate. Present only when `use_head_wise_attn_gate` is set.
    @ModuleInfo(key: "g_proj") var gProj: Linear?

    let rope: RoPELayer

    init(_ args: SharpAIStep3p7TextArguments, layerIndex: Int) {
        let layerType = layerTypeAt(args, layerIndex)
        let isSliding = layerType == "sliding_attention"

        let heads = isSliding ? (args.slidingAttentionHeads ?? args.attentionHeads) : args.attentionHeads
        self.numHeads = heads
        self.numKVHeads = args.attentionGroups
        self.headDim = args.headDim
        self.scale = pow(Float(args.headDim), -0.5)

        _qProj.wrappedValue = Linear(args.hiddenSize, heads * args.headDim, bias: false)
        _kProj.wrappedValue = Linear(args.hiddenSize, args.attentionGroups * args.headDim, bias: false)
        _vProj.wrappedValue = Linear(args.hiddenSize, args.attentionGroups * args.headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * args.headDim, args.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        _gProj.wrappedValue = args.useHeadWiseAttnGate
            ? Linear(args.hiddenSize, heads, bias: false) : nil

        let theta = args.ropeThetaPerLayer.isEmpty
            ? args.ropeTheta : perLayer(args.ropeThetaPerLayer, layerIndex, args.ropeTheta)
        let factor = perLayer(args.partialRotaryFactors, layerIndex, 1.0)
        let ropeDims = Int(Float(args.headDim) * factor)
        let usesScaling = args.yarnOnlyTypes.isEmpty || args.yarnOnlyTypes.contains(layerType)
        let scaling = usesScaling ? args.ropeScaling : nil

        self.rope = initializeRope(
            dims: ropeDims,
            base: theta,
            traditional: false,
            scalingConfig: scaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qNorm(qProj(x).reshaped(B, L, numHeads, headDim)).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, L, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        // B, H, L, D -> B, L, H, D so the head-wise gate broadcasts over D.
        output = output.transposed(0, 2, 1, 3)
        if let gProj {
            output = output * sigmoid(gProj(x)).expandedDimensions(axis: -1)
        }
        return oProj(output.reshaped(B, L, -1))
    }
}

// MARK: - MLP

/// SwiGLU MLP. A positive `limit` applies mlx-lm's clamped SwiGLU:
/// `clip(silu(gate), max: limit) * clip(up, -limit, limit)`.
final class SharpAIStep3p7MLP: Module, UnaryLayer {
    let limit: Float?

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int, limit: Float) {
        self.limit = limit > 0 ? limit : nil
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let limit {
            let gate = clip(silu(gateProj(x)), max: limit)
            let up = clip(upProj(x), min: -limit, max: limit)
            return downProj(gate * up)
        }
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE

final class SharpAIStep3p7MoE: Module, UnaryLayer {
    let topK: Int
    let routerScalingFactor: Float
    let normExpertWeight: Bool

    /// Router weight. The checkpoint key is `moe.gate.weight`.
    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "router_bias") var routerBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "share_expert") var shareExpert: SharpAIStep3p7MLP

    init(_ args: SharpAIStep3p7TextArguments, layerIndex: Int) {
        self.topK = args.moeTopK
        self.routerScalingFactor = args.moeRouterScalingFactor
        self.normExpertWeight = args.normExpertWeight

        _gate.wrappedValue = Linear(args.hiddenSize, args.moeNumExperts, bias: false)
        _routerBias.wrappedValue = MLXArray.zeros([args.moeNumExperts])

        let routedLimit = perLayer(args.swigluLimits, layerIndex, 0)
        if routedLimit > 0 {
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.moeIntermediateSize,
                numExperts: args.moeNumExperts,
                activation: { x in clip(silu(x), max: routedLimit) },
                activationProduct: { gate, up in
                    clip(silu(gate), max: routedLimit) * clip(up, min: -routedLimit, max: routedLimit)
                }
            )
        } else {
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.moeIntermediateSize,
                numExperts: args.moeNumExperts
            )
        }

        _shareExpert.wrappedValue = SharpAIStep3p7MLP(
            hiddenSize: args.hiddenSize,
            intermediateSize: args.shareExpertDim,
            limit: perLayer(args.swigluLimitsShared, layerIndex, 0)
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Sigmoid router scores in float32. Selection uses scores plus router_bias, and the
        // weights come from the unbiased scores (moe_gate_select in mlx-lm).
        let scores = sigmoid(gate(x).asType(.float32))
        let corrected = scores + routerBias.asType(.float32)
        let inds = argPartition(-corrected, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]

        var weights = takeAlong(scores, inds, axis: -1)
        if normExpertWeight {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + Float(1e-20))
        }
        weights = weights * routerScalingFactor

        let routed = switchMLP(x, inds)
        let combined = (routed * weights.expandedDimensions(axis: -1))
            .sum(axis: -2)
            .asType(routed.dtype)
        return combined + shareExpert(x)
    }
}

// MARK: - Decoder layer

final class SharpAIStep3p7DecoderLayer: Module {
    let isSliding: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIStep3p7Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    /// Set on dense layers.
    @ModuleInfo(key: "mlp") var denseMLP: SharpAIStep3p7MLP?
    /// Set on MoE layers.
    @ModuleInfo(key: "moe") var moe: SharpAIStep3p7MoE?

    init(_ args: SharpAIStep3p7TextArguments, layerIndex: Int) {
        self.isSliding = layerTypeAt(args, layerIndex) == "sliding_attention"

        _selfAttn.wrappedValue = SharpAIStep3p7Attention(args, layerIndex: layerIndex)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        if args.moeLayers.contains(layerIndex) {
            _moe.wrappedValue = SharpAIStep3p7MoE(args, layerIndex: layerIndex)
        } else {
            _denseMLP.wrappedValue = SharpAIStep3p7MLP(
                hiddenSize: args.hiddenSize,
                intermediateSize: args.intermediateSize,
                limit: perLayer(args.swigluLimitsShared, layerIndex, 0)
            )
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let normed = postAttentionLayerNorm(h)
        if let moe {
            return h + moe(normed)
        }
        return h + denseMLP!(normed)
    }
}

// MARK: - Inner model

final class SharpAIStep3p7Inner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [SharpAIStep3p7DecoderLayer]
    @ModuleInfo var norm: RMSNorm

    let slidingWindow: Int
    /// Index of the first full-attention layer, used to build the causal mask.
    let fullIdx: Int
    /// Index of the first sliding-attention layer, if any.
    let slidingIdx: Int?

    init(_ args: SharpAIStep3p7TextArguments) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        let layerList = (0 ..< args.hiddenLayers).map {
            SharpAIStep3p7DecoderLayer(args, layerIndex: $0)
        }
        self.layers = layerList
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.slidingWindow = args.slidingWindow
        self.fullIdx = layerList.firstIndex(where: { !$0.isSliding }) ?? 0
        self.slidingIdx = layerList.firstIndex(where: { $0.isSliding })
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        let fullMask = createAttentionMask(h: h, cache: cache?[fullIdx])
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        if let slidingIdx {
            slidingMask = createAttentionMask(
                h: h, cache: cache?[slidingIdx], windowSize: slidingWindow)
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: layer.isSliding ? slidingMask : fullMask, cache: cache?[i])
        }
        return norm(h)
    }
}

// MARK: - Model

/// Text-only step3p5 causal LM used as the `text_config` backbone of step3p7.
public class SharpAIStep3p7TextModel: Module {
    public let vocabularySize: Int

    @ModuleInfo var model: SharpAIStep3p7Inner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: SharpAIStep3p7TextArguments) {
        self.vocabularySize = args.vocabularySize
        _model.wrappedValue = SharpAIStep3p7Inner(args)
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }
}
