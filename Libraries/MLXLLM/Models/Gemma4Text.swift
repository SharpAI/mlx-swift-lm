//
//  Gemma4Text.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - KV State

/// Discriminated union that carries either regular (fp16/bf16) or quantized KV tensors through
/// the attention forward pass. Mirrors the equivalent type in MLXVLM/Models/Gemma4.swift.
private enum Gemma4LLMKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    )

    var seqLen: Int {
        switch self {
        case .regular(let keys, _):          return keys.dim(2)
        case .quantized(let keys, _, _, _, _): return keys.0.dim(-2)
        }
    }
}

// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    var modelType: String = "gemma4_text"
    var hiddenSize: Int = 1536
    var numHiddenLayers: Int = 35
    var intermediateSize: Int = 6144
    var numAttentionHeads: Int = 8
    var headDim: Int = 256
    var globalHeadDim: Int = 512
    var globalPartialRotaryFactor: Float = 0.25
    var rmsNormEps: Float = 1e-6
    var vocabSize: Int = 262144
    var vocabSizePerLayerInput: Int = 262144
    var numKeyValueHeads: Int = 1
    var numGlobalKeyValueHeads: Int?
    var numKvSharedLayers: Int = 20
    var hiddenSizePerLayerInput: Int = 256
    var slidingWindow: Int = 512
    var slidingWindowPattern: Int = 5
    var maxPositionEmbeddings: Int = 131072
    var attentionKeqV: Bool = false
    var finalLogitSoftcapping: Float? = 30.0
    var useDoubleWideMlp: Bool = true
    var enableMoEBlock: Bool = false
    var numExperts: Int?
    var topKExperts: Int?
    var moeIntermediateSize: Int?
    var layerTypes: [String] = []
    var tieWordEmbeddings: Bool = true

    // RoPE parameters (nested dict with full_attention/sliding_attention sub-configs)
    var ropeParameters: [String: [String: StringOrNumber]]?

    // Derived properties
    var slidingRopeTheta: Float = 10000.0
    var fullRopeTheta: Float = 1_000_000.0
    var fullPartialRotaryFactor: Float = 1.0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case globalPartialRotaryFactor = "global_partial_rotary_factor"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case numKvSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionKeqV = "attention_k_eq_v"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case enableMoEBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case layerTypes = "layer_types"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_text"
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        self.numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 35
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        self.numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        self.globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        self.globalPartialRotaryFactor =
            try container.decodeIfPresent(Float.self, forKey: .globalPartialRotaryFactor) ?? 0.25
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        self.vocabSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 262144
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 1
        self.numGlobalKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
        self.numKvSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 20
        self.hiddenSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 256
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        self.slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.attentionKeqV =
            try container.decodeIfPresent(Bool.self, forKey: .attentionKeqV) ?? false
        self.finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        self.useDoubleWideMlp =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? true
        self.enableMoEBlock =
            try container.decodeIfPresent(Bool.self, forKey: .enableMoEBlock) ?? false
        self.numExperts =
            try container.decodeIfPresent(Int.self, forKey: .numExperts)
        self.topKExperts =
            try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        if let decoded = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            self.layerTypes = decoded
        } else {
            // Derive layer types from sliding window pattern
            var pattern = [String]()
            for i in 0 ..< slidingWindowPattern {
                pattern.append(
                    i == slidingWindowPattern - 1 ? "full_attention" : "sliding_attention")
            }
            var types = [String]()
            while types.count < numHiddenLayers {
                types.append(contentsOf: pattern)
            }
            self.layerTypes = Array(types.prefix(numHiddenLayers))
        }
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.ropeParameters =
            try container.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters)

        // Extract RoPE parameters from nested config
        if let ropeParams = ropeParameters {
            if let sliding = ropeParams["sliding_attention"] {
                self.slidingRopeTheta = sliding["rope_theta"]?.asFloat() ?? 10000.0
            }
            if let full = ropeParams["full_attention"] {
                self.fullRopeTheta = full["rope_theta"]?.asFloat() ?? 1_000_000.0
                self.fullPartialRotaryFactor =
                    full["partial_rotary_factor"]?.asFloat() ?? 1.0
            }
        }
    }
}

// MARK: - Helper Modules

private class RMSNormNoScale: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

private class ScaledLinear: Module {
    let weight: MLXArray
    let scalar: Float

    init(inFeatures: Int, outFeatures: Int, scalar: Float) {
        self.weight = MLXArray.zeros([outFeatures, inFeatures])
        self.scalar = scalar
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(x, weight.T) * scalar
    }
}

private enum Gemma4PositionOffset {
    case scalar(Int)
    case batch(MLXArray)
}

private func gemma4CapturePositionOffset(from cache: KVCache?) -> Gemma4PositionOffset {
    if let batchCache = cache as? BatchPositionedKVCache {
        // Snapshot the per-sequence offsets before cache.update(...) advances them.
        .batch(batchCache.batchOffset + 0)
    } else {
        .scalar(cache?.offset ?? 0)
    }
}

private func gemma4ApplyRotaryPosition<R: RoPELayer>(
    _ rope: R,
    to x: MLXArray,
    offset: Gemma4PositionOffset
) -> MLXArray {
    switch offset {
    case .scalar(let value):
        rope(x, offset: value)
    case .batch(let values):
        rope(x, offset: values)
    }
}

// MARK: - Attention

private class Gemma4Attention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let effectiveHeadDim: Int
    let nHeads: Int
    let nKvHeads: Int
    let useKeqV: Bool
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?
    @ModuleInfo(key: "v_norm") var vNorm: RMSNormNoScale?

    @ModuleInfo var rope: RoPELayer

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"

        // Full attention uses globalHeadDim, sliding uses headDim
        self.effectiveHeadDim =
            isSliding ? config.headDim : config.globalHeadDim

        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads

        // K-eq-V for full attention layers
        self.useKeqV = config.attentionKeqV && !isSliding
        if useKeqV, let globalKvHeads = config.numGlobalKeyValueHeads {
            self.nKvHeads = globalKvHeads
        } else {
            self.nKvHeads = config.numKeyValueHeads
        }

        self.scale = 1.0

        self._qProj.wrappedValue = Linear(dim, nHeads * effectiveHeadDim, bias: false)
        
        // A layer owns its own K/V if it is NOT a KV-shared layer.
        // In the Gemma 4 architecture, the main model has K/V weights for all layers even if num_kv_shared_layers > 0.
        // However, the assistant model has numHiddenLayers == numKvSharedLayers and NO K/V weights at all.
        let isAssistant = config.numHiddenLayers == config.numKvSharedLayers
        let hasKv = !isAssistant
        
        if hasKv {
            self._kProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
            if !useKeqV {
                self._vProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
            }
            self._kNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
            self._vNorm.wrappedValue = RMSNormNoScale(eps: config.rmsNormEps)
        }
        
        self._oProj.wrappedValue = Linear(nHeads * effectiveHeadDim, dim, bias: false)

        self._qNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)

        // RoPE: sliding uses default, full uses proportional with partial rotation
        if isSliding {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.slidingRopeTheta, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
        } else {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.fullRopeTheta, traditional: false,
                scalingConfig: [
                    "type": .string("proportional"),
                    "partial_rotary_factor": .float(config.fullPartialRotaryFactor),
                ],
                maxPositionEmbeddings: nil)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        positionOffset: Gemma4PositionOffset? = nil
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4PositionOffset) {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, nHeads, effectiveHeadDim)
        queries = qNorm(queries)

        let activePositionOffset = positionOffset ?? gemma4CapturePositionOffset(from: cache)

        var adjustedMask = mask
        let kvState: Gemma4LLMKVState
        if let (sharedK, sharedV) = sharedKV {
            // KV-shared layers use pre-computed KV from an earlier layer
            kvState = .regular(keys: sharedK, values: sharedV)
            
            // For sharedKV, we still need to adjust the mask if cache is shorter than mask
            if case .array(let maskArray) = mask {
                let keysSeqLen = kvState.seqLen
                if maskArray.dim(-1) > keysSeqLen {
                    adjustedMask = .array(maskArray[.ellipsis, 0 ..< keysSeqLen])
                }
            }
            
        } else {
            guard let kProj = kProj, let kNorm = kNorm, let vNorm = vNorm else {
                fatalError("Layer \(layerIdx) is a KV-shared layer but received no sharedKV")
            }
            var k = kProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
            k = kNorm(k)
            k = k.transposed(0, 2, 1, 3)
            k = gemma4ApplyRotaryPosition(rope, to: k, offset: activePositionOffset)

            var v: MLXArray
            if let vProj {
                v = vProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
                v = vNorm(v)
                v = v.transposed(0, 2, 1, 3)
            } else {
                v = vNorm(k)
            }

            if let quantizedCache = cache as? QuantizedKVCacheProtocol {
                let (qKeys, qValues) = quantizedCache.updateQuantized(keys: k, values: v)
                kvState = .quantized(
                    keys: qKeys,
                    values: qValues,
                    groupSize: quantizedCache.groupSize,
                    bits: quantizedCache.bits,
                    mode: quantizedCache.mode
                )
            } else if let cache {
                let (updatedK, updatedV) = cache.update(keys: k, values: v)
                kvState = .regular(keys: updatedK, values: updatedV)
            } else {
                kvState = .regular(keys: k, values: v)
            }
            
            // Adjust mask if cache is shorter than mask
            if case .array(let maskArray) = mask {
                let keysSeqLen = kvState.seqLen
                if maskArray.dim(-1) > keysSeqLen {
                    adjustedMask = .array(maskArray[.ellipsis, 0 ..< keysSeqLen])
                }
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = gemma4ApplyRotaryPosition(rope, to: queries, offset: activePositionOffset)

        let output: MLXArray =
                switch kvState {
                case .regular(let rKeys, let rValues):
                    MLXFast.scaledDotProductAttention(
                        queries: queries,
                        keys: rKeys,
                        values: rValues,
                        scale: scale,
                        mask: adjustedMask ?? .none
                    )
                case .quantized(let qKeys, let qValues, let groupSize, let bits, let mode):
                    quantizedScaledDotProductAttention(
                        queries: queries,
                        quantizedKeys: qKeys,
                        quantizedValues: qValues,
                        scale: scale,
                        mask: adjustedMask ?? .none,
                        groupSize: groupSize,
                        bits: bits,
                        mode: mode
                    )
                }

            // Build the kvPair that will be stored in `intermediates` and potentially
            // consumed as `sharedKV` by later KV-sharing layers.  Those layers expect
            // full-context FP16/BF16 tensors.  For the regular path we already have them;
            // for the quantized path we dequantize the full accumulated cache state.
            let retKVPair: (MLXArray, MLXArray)
            switch kvState {
            case .regular(let rk, let rv):
                retKVPair = (rk, rv)
            case .quantized(let qk, let qv, let groupSize, let bits, _):
                // If the cache has accumulated more than the current step we need the
                // full state, not just the new-token quantized tuples.  Try the protocol
                // accessor first; fall back to dequantizing the just-updated tuples.
                if let fullState = (cache as? QuantizedKVCacheProtocol)?.getQuantizedState() {
                    let fullKeys   = dequantized(fullState.0.0, scales: fullState.0.1,
                                                 biases: fullState.0.2, groupSize: groupSize, bits: bits)
                    let fullValues = dequantized(fullState.1.0, scales: fullState.1.1,
                                                 biases: fullState.1.2, groupSize: groupSize, bits: bits)
                    retKVPair = (fullKeys, fullValues)
                } else {
                    // First decode step (offset==1): no prior context to merge.
                    retKVPair = (dequantized(qk.0, scales: qk.1, biases: qk.2,
                                             groupSize: groupSize, bits: bits),
                                 dequantized(qv.0, scales: qv.1, biases: qv.2,
                                             groupSize: groupSize, bits: bits))
                }
            }

            return (
                oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1)),
                retKVPair,
                activePositionOffset
            )
        }

}

// MARK: - MLP

private class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        let firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        let isKvSharedLayer = layerIdx >= firstKvSharedLayerIdx && firstKvSharedLayerIdx > 0
        let useDoubleWide = config.useDoubleWideMlp && isKvSharedLayer
        let intermediateSize = config.intermediateSize * (useDoubleWide ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE Router

private class Gemma4TextRouter: Module {
    let topKExperts: Int
    let rootSize: Float

    @ModuleInfo(key: "norm") var norm: RMSNormNoScale
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "scale") var scale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(_ config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts, let topKExperts = config.topKExperts else {
            fatalError("Gemma4 MoE router requires numExperts and topKExperts")
        }

        self.topKExperts = topKExperts
        self.rootSize = pow(Float(config.hiddenSize), -0.5)

        self._norm.wrappedValue = RMSNormNoScale(eps: config.rmsNormEps)
        self._proj.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var x = norm(x)
        x = x * MLXArray(rootSize, dtype: x.dtype)
        x = x * scale.asType(x.dtype)

        let expertScores = proj(x)
        let routerProbabilities = MLX.softmax(expertScores, axis: -1, precise: true)

        let topKIndices = MLX.argPartition(-expertScores, kth: topKExperts - 1, axis: -1)[
            .ellipsis, ..<topKExperts,
        ]
        var topKWeights = MLX.takeAlong(routerProbabilities, topKIndices, axis: -1)
        topKWeights = topKWeights / MLX.sum(topKWeights, axis: -1, keepDims: true)
        topKWeights = topKWeights * perExpertScale[topKIndices].asType(topKWeights.dtype)
        return (topKIndices, topKWeights)
    }
}

// MARK: - MoE Experts

private class Gemma4TextExperts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(_ config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts,
            let moeIntermediateSize = config.moeIntermediateSize
        else {
            fatalError("Gemma4 MoE experts require numExperts and moeIntermediateSize")
        }

        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts,
            activation: geluApproximate,
            bias: false
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, topKIndices: MLXArray, topKWeights: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let hidden = x.dim(2)
        let topK = topKIndices.dim(-1)

        let expertOutput = switchGLU(
            x.reshaped(batch * length, hidden),
            topKIndices.reshaped(batch * length, topK)
        )
        let weights = topKWeights.reshaped(batch * length, topK, 1).asType(expertOutput.dtype)
        return (expertOutput * weights).sum(axis: -2).reshaped(batch, length, hidden)
    }
}

// MARK: - Decoder Layer

private class Gemma4DecoderLayer: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "router") var router: Gemma4TextRouter?
    @ModuleInfo(key: "experts") var experts: Gemma4TextExperts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayernorm1: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayernorm2: RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayernorm2: RMSNorm?

    // Per-layer input (PLE) gating
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    // Per-layer scalar
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput

        self._selfAttn.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma4MLP(config, layerIdx: layerIdx)

        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.enableMoEBlock {
            self._router.wrappedValue = Gemma4TextRouter(config)
            self._experts.wrappedValue = Gemma4TextExperts(config)
            self._postFeedforwardLayernorm1.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayernorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayernorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        if hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._layerScalar.wrappedValue = MLXArray.ones([1], dtype: .float16)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        positionOffset: Gemma4PositionOffset? = nil
    ) -> (MLXArray, (MLXArray, MLXArray), Gemma4PositionOffset) {
        let residual = x

        let h = inputLayernorm(x)
        let (attnOut, kvPair, attnPositionOffset) = selfAttn(
            h, mask: mask, cache: cache, sharedKV: sharedKV, positionOffset: positionOffset)
        let postAttn = postAttentionLayernorm(attnOut)
        var out = residual + postAttn

        let residual2 = out
        if let router, let experts,
            let postFeedforwardLayernorm1,
            let postFeedforwardLayernorm2,
            let preFeedforwardLayernorm2
        {
            // MoE: dual dense + sparse feedforward
            var dense = preFeedforwardLayernorm(out)
            dense = mlp(dense)
            dense = postFeedforwardLayernorm1(dense)

            let (topKIndices, topKWeights) = router(out)
            var sparse = preFeedforwardLayernorm2(out)
            sparse = experts(sparse, topKIndices: topKIndices, topKWeights: topKWeights)
            sparse = postFeedforwardLayernorm2(sparse)

            out = dense + sparse
        } else {
            out = preFeedforwardLayernorm(out)
            out = mlp(out)
        }
        out = postFeedforwardLayernorm(out)
        out = residual2 + out

        // PLE gating
        if let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm,
            let perLayerInput
        {
            let residual3 = out
            var g = gate(out)
            g = geluApproximate(g)
            g = g * perLayerInput
            g = proj(g)
            g = norm(g)
            out = residual3 + g
        }

        out = out * layerScalar

        return (out, kvPair, attnPositionOffset)
    }
}

// MARK: - Text Model

private class Gemma4TextModelInner: Module {
    let config: Gemma4TextConfiguration
    let embedScale: Float
    let hiddenSizePerLayerInput: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: RMSNorm

    // Per-layer embeddings (PLE)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: ScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm?

    // KV sharing mapping: for each layer, which earlier layer provides KVs
    let previousKvs: [Int]
    let firstKvSharedLayerIdx: Int
    
    public var lastHiddenState: MLXArray?
    public var hiddenStateBeforeNorm: MLXArray?

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.embedScale = Float(config.hiddenSize).squareRoot()
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            Gemma4DecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // PLE
        if config.hiddenSizePerLayerInput > 0 {
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.numHiddenLayers * config.hiddenSizePerLayerInput)
            self._perLayerModelProjection.wrappedValue = ScaledLinear(
                inFeatures: config.hiddenSize,
                outFeatures: config.numHiddenLayers * config.hiddenSizePerLayerInput,
                scalar: pow(Float(config.hiddenSize), -0.5))
            self._perLayerProjectionNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        }

        // Build KV-sharing map
        self.firstKvSharedLayerIdx = config.numHiddenLayers - config.numKvSharedLayers
        var kvMap = Array(0 ..< config.numHiddenLayers)
        if config.numKvSharedLayers > 0 {
            // Find the last non-shared layer of each type
            var lastByType = [String: Int]()
            for i in 0 ..< firstKvSharedLayerIdx {
                lastByType[config.layerTypes[i]] = i
            }
            // Shared layers reference the last non-shared layer of the same type
            for j in firstKvSharedLayerIdx ..< config.numHiddenLayers {
                if let prev = lastByType[config.layerTypes[j]] {
                    kvMap[j] = prev
                }
            }
        }
        self.previousKvs = kvMap

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let inputEmbeddings = embedTokens(inputs)
        var h = inputEmbeddings * embedScale

        // Compute per-layer inputs (PLE)
        var perLayerInputs: [MLXArray?]
        if hiddenSizePerLayerInput > 0,
            let embedPerLayer = embedTokensPerLayer,
            let modelProj = perLayerModelProjection,
            let projNorm = perLayerProjectionNorm
        {
            // Token-based PLE
            let tokenPLE =
                embedPerLayer(inputs)
                * Float(config.hiddenSizePerLayerInput).squareRoot()

            // [B, L, numLayers * hiddenSizePerLayerInput] -> [B, L, numLayers, hiddenSizePerLayerInput]
            let reshapedTokenPLE = tokenPLE.reshaped(
                tokenPLE.dim(0), tokenPLE.dim(1),
                config.numHiddenLayers, config.hiddenSizePerLayerInput)

            // Model projection PLE
            let modelPLE = modelProj(h).reshaped(
                h.dim(0), h.dim(1),
                config.numHiddenLayers, config.hiddenSizePerLayerInput)
            let normedModelPLE = projNorm(modelPLE)

            // Combine: (model_proj + token_embed) * 2^{-0.5}
            let perLayerInputScale = pow(Float(2.0), -0.5)
            let combined = (normedModelPLE + reshapedTokenPLE) * perLayerInputScale

            perLayerInputs = (0 ..< config.numHiddenLayers).map { i in
                combined[.ellipsis, i, 0...]
            }
        } else {
            perLayerInputs = Array(repeating: nil, count: config.numHiddenLayers)
        }

        // Extend cache array for shared layers (which get nil caches)
        var fullCache: [KVCache?]
        if let cache {
            fullCache = cache.map { Optional($0) }
            while fullCache.count < config.numHiddenLayers {
                fullCache.append(nil)
            }
        } else {
            fullCache = Array(repeating: nil, count: config.numHiddenLayers)
        }

        // Build masks: one per attention type
        var maskByType = [String: MLXFast.ScaledDotProductAttentionMaskMode]()
        for (i, layer) in layers.enumerated() {
            let lt = layer.layerType
            if maskByType[lt] == nil {
                if lt == "sliding_attention" {
                    maskByType[lt] = createAttentionMask(
                        h: h, cache: fullCache[i], windowSize: config.slidingWindow)
                } else {
                    maskByType[lt] = createAttentionMask(h: h, cache: fullCache[i])
                }
            }
        }

        // Forward through layers, tracking intermediate KV pairs for sharing
        var intermediates = [(kv: (MLXArray, MLXArray)?, positionOffset: Gemma4PositionOffset?)](
            repeating: (nil, nil), count: config.numHiddenLayers)

        let isAssistant = (config.numKvSharedLayers == config.numHiddenLayers)
        
        for (idx, layer) in layers.enumerated() {
            var sharedKV: (MLXArray, MLXArray)? = nil
            var sharedPositionOffset: Gemma4PositionOffset? = nil
            
            if isAssistant, let fullCache = cache, fullCache.count > config.numHiddenLayers {
                // Determine which layer of the main model to share KV from
                let mainIdx = layer.layerType == "sliding_attention" ? fullCache.count - 2 : fullCache.count - 1
                let cacheElement = fullCache[mainIdx]
                if let c = cacheElement as? KVCacheSimple, let k = c.keys, let v = c.values {
                    sharedKV = (k, v)
                } else if let c = cacheElement as? RotatingKVCache, let k = c.keys, let v = c.values {
                    sharedKV = (k, v)
                }
            } else {
                let prevIdx = previousKvs[idx]
                sharedKV = intermediates[prevIdx].kv
                sharedPositionOffset = intermediates[prevIdx].positionOffset
            }

            let mask = maskByType[layer.layerType]
            let (out, kvPair, positionOffset) = layer(
                h,
                mask: mask,
                cache: fullCache[idx],
                perLayerInput: perLayerInputs[idx],
                sharedKV: sharedKV,
                positionOffset: sharedPositionOffset
            )
            h = out
            intermediates[idx] = (kvPair, positionOffset)
        }

        self.hiddenStateBeforeNorm = h
        h = norm(h)
        self.lastHiddenState = h
        return h
    }
}

// MARK: - Public Model

public class Gemma4TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public var lastHiddenState: MLXArray? { return model.lastHiddenState }
    public var hiddenStateBeforeNorm: MLXArray? { return model.hiddenStateBeforeNorm }

    fileprivate let config: Gemma4TextConfiguration
    fileprivate let model: Gemma4TextModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        self.model = Gemma4TextModelInner(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        if let cap = config.finalLogitSoftcapping {
            out = tanh(out / cap) * cap
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (k, v) in weights {
            // Skip vision/audio/rotary weights and unsupported MTP keys
            if k.contains("self_attn.rotary_emb")
                || k.contains("input_max")
                || k.contains("input_min")
                || k.contains("output_max")
                || k.contains("output_min")
                || k.hasPrefix("pre_projection")
                || k.hasPrefix("post_projection")
                || k.hasPrefix("masked_embedding")
            {
                continue
            }

            // MoE expert weight remapping: fused HF tensors → SwitchGLU layout
            if k.hasSuffix(".experts.down_proj") {
                sanitized[
                    k.replacingOccurrences(
                        of: ".experts.down_proj",
                        with: ".experts.switch_glu.down_proj.weight"
                    )
                ] = v
                continue
            }
            if k.hasSuffix(".experts.gate_up_proj") {
                let mid = v.dim(-2) / 2
                sanitized[
                    k.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.gate_proj.weight"
                    )
                ] = v[.ellipsis, ..<mid, 0...]
                sanitized[
                    k.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.up_proj.weight"
                    )
                ] = v[.ellipsis, mid..., 0...]
                continue
            }

            sanitized[k] = v
        }
        return sanitized
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let firstKvShared = config.numHiddenLayers - config.numKvSharedLayers

        var caches = [any KVCache]()
        for i in 0 ..< firstKvShared {
            if config.layerTypes[i] == "full_attention" {
                caches.append(StandardKVCache())
            } else {
                caches.append(RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            }
        }
        return caches
    }
}

// MARK: - LoRA

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers.map { $0.selfAttn }
    }
}

// MARK: - Assistant

public class Gemma4AssistantModel: Module, LLMModel, DualModelMTP, MTPPartialRollback, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let config: Gemma4TextConfiguration
    fileprivate let model: Gemma4TextModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    
    public var _preProjectionWeight: MLXArray?
    public var _postProjectionWeight: MLXArray?
    
    public var preProjectionWeight: MLXArray? { _preProjectionWeight }
    public var postProjectionWeight: MLXArray? { _postProjectionWeight }

    // Masked embedder state (centroid-based sparse logit projection)
    var _centroidWeight: MLXArray?       // [num_centroids, hidden] — centroids linear weight
    var _tokenOrdering: MLXArray?        // [vocab_size] int32 — canonical token ordering (ordered->canonical)
    var _invTokenOrdering: MLXArray?     // [vocab_size] int32 — inverse token ordering (canonical->ordered)
    var numCentroids: Int = 2048
    var centroidTopK: Int = 32
    var vocabSizePerCentroid: Int = 128  // vocab_size / num_centroids

    // Reference to the main model so we can call it inside callMTP
    public var mainModelRef: (any BaseLanguageModel)? = nil

    /// Full [B, S, D] backbone hidden state from the most recent callMTP verification pass.
    /// Stored so MTPTokenIterator can extract the hidden state at the accepted position
    /// for partial rollback (re-seeding the MTP head without re-running the main model).
    public var lastBackboneHiddenStateAll: MLXArray? = nil

    /// Number of draft tokens to produce per MTP head call.
    /// depth=2: each pass costs 24% overhead (2 × ~12% per assistant layer pass at 40K).
    /// depth=4: costs 48% overhead — empirically worse due to Metal kernel launch cost per depth.
    public var numMTPDraftTokens: Int = 2

    public init(_ fullConfig: Gemma4Configuration) {
        let config = fullConfig.textConfig
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        self.model = Gemma4TextModelInner(config)
        
        self.numCentroids = fullConfig.numCentroids ?? 2048
        self.centroidTopK = fullConfig.centroidIntermediateTopK ?? 32
        self.vocabSizePerCentroid = config.vocabSize / self.numCentroids
        
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
        super.init()
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights
        if let w = weights["pre_projection.weight"] {
            self._preProjectionWeight = w
            sanitized.removeValue(forKey: "pre_projection.weight")
        }
        if let w = weights["post_projection.weight"] {
            self._postProjectionWeight = w
            sanitized.removeValue(forKey: "post_projection.weight")
        }
        
        // Load masked embedder weights for centroid-based sparse logit projection
        if let w = weights["masked_embedding.centroids.weight"] {
            self._centroidWeight = w
            sanitized.removeValue(forKey: "masked_embedding.centroids.weight")
        }
        if let w = weights["masked_embedding.token_ordering"] {
            self._tokenOrdering = w.asType(.int32)
            // Precompute inverse ordering: inv[canonical_id] = ordered_position
            // This enables O(1) conversion from ordered logits to canonical logits
            self._invTokenOrdering = argSort(w.asType(.int32), axis: 0)
            sanitized.removeValue(forKey: "masked_embedding.token_ordering")
        }
        
        return sanitized
    }

    /// Compute logits using the centroid-based sparse masked embedder.
    /// Matches HF Gemma4AssistantMaskedEmbedder.forward().
    /// - hNormed: [B, 1, hidden=256]
    /// Returns [B, 1, vocab]
    func maskedEmbedderLogits(_ hNormed: MLXArray) -> MLXArray {
        guard let centroidW = _centroidWeight, let tokenOrdering = _tokenOrdering else {
            // Fallback to full projection
            return model.embedTokens.asLinear(hNormed)
        }
        
        let B = hNormed.dim(0)
        let S = hNormed.dim(1)
        let vocabSize = config.vocabSize
        
        // centroid_logits = hNormed @ centroidW.T  → [B, S, num_centroids]
        let centroidLogits = matmul(hNormed, centroidW.T)
        
        // top_k_indices = argTopK(centroid_logits, k=centroidTopK) → [B, S, topK]
        // MLX doesn't have argTopK directly; use argSort descending and take first topK
        let sortedCentroidIdx = argSort(centroidLogits, axis: -1)  // ascending
        let reversedIdx = sortedCentroidIdx[.ellipsis, (sortedCentroidIdx.dim(-1) - centroidTopK)...]
        // reversedIdx is [B, S, topK] — indices of top-K centroids
        
        // token_ordering reshaped: [num_centroids, vocabSizePerCentroid]
        let tokenOrderingReshaped = tokenOrdering.reshaped([numCentroids, vocabSizePerCentroid])
        
        // Gather canonical positions for each selected centroid
        // For each of the topK centroid indices, gather its vocabSizePerCentroid token positions
        // selected_canonical: [B, S, topK, vocabSizePerCentroid]
        let topKFlat = reversedIdx.reshaped([-1])  // [B*S*topK]
        let selectedCanonical = tokenOrderingReshaped[topKFlat]  // [B*S*topK, vocabSizePerCentroid]
        let selectedCanonicalShaped = selectedCanonical.reshaped([B, S, centroidTopK, vocabSizePerCentroid])
        
        // Gather embeddings at those positions: embed_tokens.weight[canonical] → [B*S*topK*K, hidden]
        let embedWeight = model.embedTokens.weight  // [vocab, 256]
        let selectedFlat = selectedCanonicalShaped.reshaped([-1]).asType(.int32)  // [B*S*topK*K]
        let selectedEmbeds = embedWeight[selectedFlat]  // [B*S*topK*K, 256]
        let totalCandidates = centroidTopK * vocabSizePerCentroid
        let selectedEmbedsShaped = selectedEmbeds.reshaped([B, S, totalCandidates, config.hiddenSize])
        
        // dot products: [B, S, 1, hidden] @ [B, S, hidden, topK*K] → [B, S, topK*K]
        let hExpanded = hNormed.expandedDimensions(axis: -2)  // [B, S, 1, hidden]
        let selectedLogits = matmul(hExpanded, selectedEmbedsShaped.transposed(0, 1, 3, 2)).squeezed(axis: -2)
        // selectedLogits: [B, S, topK*K]
        
        // Build output tensor: fill with min - 1.0, scatter selectedLogits to canonical positions
        let minVal = selectedLogits.min(axes: [-1], keepDims: true)  // [B, S, 1]
        var output = broadcast(minVal - 1.0, to: [B, S, vocabSize])  // [B, S, vocab]
        
        // Scatter selectedLogits into output at scatterIdx positions.
        // We use a workaround: create an index array and use scatter-add pattern.
        // selectedLogits: [B, S, topK*K], scatterIdx: [B, S, topK*K] (token indices)
        // For each (b,s,k): output[b, s, scatterIdx[b,s,k]] = selectedLogits[b,s,k]
        // Use mlx scatter via the __setitem__ approach:
        let scatterIdx2D = selectedCanonicalShaped.reshaped([B * S, totalCandidates]).asType(.int32)
        let selectedLogits2D = selectedLogits.reshaped([B * S, totalCandidates])
        let output2D = output.reshaped([B * S, vocabSize])
        let rowIndices = MLXArray.arange(B * S).asType(.int32).reshaped([B * S, 1])
        output2D[rowIndices, scatterIdx2D] = selectedLogits2D
        output = output2D.reshaped([B, S, vocabSize])
        
        return output
    }


    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Fallback for standard autoregressive call, though not used in MTP flow
        let h = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(h)
        }
        return model.embedTokens.asLinear(h)
    }

    /// Override prefill to delegate to the main model, not the assistant layers.
    ///
    /// The inherited LLMModel.prepare runs `self(input, cache, state)` which calls
    /// `callAsFunction` — i.e. the 4-layer assistant transformer.  That writes into
    /// indices [0..3] of the *main model's* 30-layer KVCache, leaving all 30 layers
    /// uninitialized for the main model.  When callMTP subsequently runs the main
    /// model it finds a cold cache, producing garbage logits, so mtpLogits is never
    /// seeded and speculateRound can never produce draft tokens.
    ///
    /// Fix: run the MAIN MODEL's prepare() instead, populating all 30 KV layers correctly.
    /// The assistant model is only invoked during the MTP head phase (callMTP/callMTPHeadOnly).
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        guard let mainModel = mainModelRef as? any LLMModel else {
            // mainModelRef not set yet — fall through to token-by-token (no prefill cache warming)
            return .tokens(input.text)
        }
        return try mainModel.prepare(input, cache: cache, windowSize: windowSize)
    }

    // MARK: - MTP Head Loop (shared by callMTP and callMTPHeadOnly)

    /// Run the iterative MTP head loop.
    /// - Parameters:
    ///   - hLast: [B, 1, backboneDim] — initial backbone hidden state
    ///   - eEmbed: [B, 1, backboneDim] — embedding of the first "next" token
    ///   - posOffset: fixed position offset for assistant RoPE
    ///   - backboneDim: dimension of backbone hidden state
    ///   - cache: main model KV cache (for cross-attention in assistant layers)
    ///   - depth: how many MTP outputs to produce
    /// - Returns: [depth-0 logits, depth-1 logits, ...] each [B, 1, V]
    private func runMTPHead(
        hLast hLastIn: MLXArray,
        eEmbed eEmbedIn: MLXArray,
        posOffset: Gemma4PositionOffset,
        backboneDim: Int,
        cache: [KVCache]?,
        depth: Int
    ) -> [MLXArray] {
        var hLast = hLastIn
        var eEmbed = eEmbedIn
        var results = [MLXArray]()

        for _ in 0 ..< depth {
            let hConcat = concatenated([eEmbed, hLast], axis: -1)
            var hAssistant: MLXArray
            if let w = preProjectionWeight {
                hAssistant = matmul(hConcat, w.T)
            } else {
                hAssistant = hConcat
                if hAssistant.dim(-1) != config.hiddenSize {
                    hAssistant = hAssistant[.ellipsis, ..<config.hiddenSize]
                }
            }

            for i in 0 ..< config.numHiddenLayers {
                let layer = model.layers[i]
                var sharedKV: (MLXArray, MLXArray)? = nil
                if let fullCache = cache {
                    let layerType = model.layers[i].layerType
                    let mainIdx = layerType == "sliding_attention" ? fullCache.count - 2 : fullCache.count - 1
                    if mainIdx >= 0 {
                        // Cap shared-KV cross-attention to the last N backbone positions.
                        // The backbone hLast already encodes the full history; the assistant
                        // only needs local conditioning. Capping to 16 positions reduces
                        // cross-attention bandwidth from O(T) → O(16) at long contexts,
                        // eliminating the 2× slowdown at 40K–100K without hurting short-ctx.
                        let maxSharedKV = 16
                        let cacheElement = fullCache[mainIdx]
                        if let c = cacheElement as? KVCacheSimple, let k = c.keys, let v = c.values {
                            let seqLen = min(c.offset, k.dim(2))
                            let startPos = max(0, seqLen - maxSharedKV)
                            let validK = k[0..., 0..., startPos ..< seqLen, 0...]
                            let validV = v[0..., 0..., startPos ..< seqLen, 0...]
                            sharedKV = (validK, validV)
                        } else if let c = cacheElement as? RotatingKVCache, let k = c.keys, let v = c.values {
                            let seqLen = min(c.offset, k.dim(2))
                            let startPos = max(0, seqLen - maxSharedKV)
                            let validK = k[0..., 0..., startPos ..< seqLen, 0...]
                            let validV = v[0..., 0..., startPos ..< seqLen, 0...]
                            sharedKV = (validK, validV)
                        }
                    }
                }

                let (out, _, _) = layer(hAssistant, mask: nil, cache: nil, perLayerInput: nil, sharedKV: sharedKV, positionOffset: posOffset)
                hAssistant = out
            }

            let hNormed = model.norm(hAssistant)
            let logits: MLXArray
            if _centroidWeight != nil {
                logits = maskedEmbedderLogits(hNormed)
            } else {
                logits = model.embedTokens.asLinear(hNormed)
            }
            results.append(logits)

            if let w = postProjectionWeight {
                hLast = matmul(hNormed, w.T)
            } else {
                hLast = hNormed
                if hLast.dim(-1) != backboneDim {
                    if hLast.dim(-1) > backboneDim {
                        hLast = hLast[.ellipsis, ..<backboneDim]
                    } else {
                        let pad = MLX.zeros([hLast.dim(0), hLast.dim(1), backboneDim - hLast.dim(-1)]).asType(hLast.dtype)
                        hLast = concatenated([hLast, pad], axis: -1)
                    }
                }
            }

            let lastLogits = logits[0..., logits.dim(1)-1, 0...]
            let nextTokenScalar = argMax(lastLogits, axis: -1)
            let nextTokenReshaped = nextTokenScalar.reshaped([1, 1])
            if let g4tm = mainModelRef as? Gemma4TextModel {
                let emb = g4tm.model.embedTokens(nextTokenReshaped)
                eEmbed = emb * MLXArray(g4tm.model.embedScale, dtype: emb.dtype)
            } else if let g4m = mainModelRef as? Gemma4Model {
                let emb = g4m.languageModel.model.embedTokens(nextTokenReshaped)
                eEmbed = emb * MLXArray(g4m.languageModel.model.embedScale, dtype: emb.dtype)
            } else {
                let emb = model.embedTokens(nextTokenReshaped)
                eEmbed = emb * MLXArray(model.embedScale, dtype: emb.dtype)
            }
        }
        return results
    }

    /// Run only the MTP head from a pre-computed backbone hidden state.
    /// Used for partial rollback: after accepting k of N drafts, this re-seeds the
    /// MTP head from h_k (stored from the verification pass) without re-running
    /// the main model. The main model still runs on y in the normal callMTP call;
    /// the draft from callMTPHeadOnly is passed in as the single draft token to verify.
    ///
    /// - Parameters:
    ///   - h: [B, 1, backboneDim] — backbone hidden state at the accepted position
    ///   - nextToken: [B, 1] int32 — the token output after the accepted position (x_{k+1})
    ///   - cache: main model KV cache (post-trim, for cross-attention)
    ///   - posOffset: sequence position of the accepted token
    ///   - mtpDepth: how many draft logits to produce
    /// - Returns: [depth-0 logits, ...] each [B, 1, V] — NO main logits prefix
    public func callMTPHeadOnly(
        _ h: MLXArray,
        nextToken: MLXArray,
        cache: [KVCache]?,
        posOffset: Int,
        mtpDepth: Int
    ) -> [MLXArray] {
        let backboneDim = h.dim(-1)
        let assistantPosOffset = Gemma4PositionOffset.scalar(posOffset)

        var eEmbed: MLXArray
        if let g4tm = mainModelRef as? Gemma4TextModel {
            let emb = g4tm.model.embedTokens(nextToken)
            eEmbed = emb * MLXArray(g4tm.model.embedScale, dtype: emb.dtype)
        } else if let g4m = mainModelRef as? Gemma4Model {
            let emb = g4m.languageModel.model.embedTokens(nextToken)
            eEmbed = emb * MLXArray(g4m.languageModel.model.embedScale, dtype: emb.dtype)
        } else {
            let emb = model.embedTokens(nextToken)
            eEmbed = emb * MLXArray(model.embedScale, dtype: emb.dtype)
        }

        return runMTPHead(
            hLast: h,
            eEmbed: eEmbed,
            posOffset: assistantPosOffset,
            backboneDim: backboneDim,
            cache: cache,
            depth: mtpDepth
        )
    }

    public func callMTP(_ inputs: MLXArray, cache: [KVCache]?, mtpCaches: [[KVCache]]?) -> [MLXArray] {
        guard let mainModel = mainModelRef else {
            fatalError("mainModelRef must be set on Gemma4AssistantModel before calling callMTP")
        }

        let posOffset = cache?.first.map { gemma4CapturePositionOffset(from: $0) }

        guard let llmMain = mainModel as? any LLMModel else {
            fatalError("mainModelRef must be an LLMModel")
        }
        let mainLogits = llmMain(inputs, cache: cache)

        var hBackbone: MLXArray
        if let g4m = mainModel as? Gemma4Model, let lhs = g4m.lastHiddenState {
            hBackbone = lhs
        } else if let g4tm = mainModel as? Gemma4TextModel, let lhs = g4tm.lastHiddenState {
            hBackbone = lhs
        } else {
            fatalError("[MTP] Could not extract normalized hidden state from main model")
        }

        // Store the full [B, S, D] hidden state so MTPTokenIterator can extract
        // the accepted-position's state for partial rollback.
        self.lastBackboneHiddenStateAll = hBackbone

        let backboneDim = hBackbone.dim(-1)
        let seqLen = hBackbone.dim(1)
        let hLast = hBackbone[0..., (seqLen-1)..<seqLen, 0...]

        let inputLen = inputs.dim(1)
        let mainLogitsLast = mainLogits[0..., -1, 0...][.newAxis]
        let predictedToken = argMax(mainLogitsLast, axis: -1)

        var eEmbed: MLXArray
        if let g4tm = mainModel as? Gemma4TextModel {
            eEmbed = g4tm.model.embedTokens(predictedToken)
            eEmbed = eEmbed * MLXArray(g4tm.model.embedScale, dtype: eEmbed.dtype)
        } else if let g4m = mainModel as? Gemma4Model {
            eEmbed = g4m.languageModel.model.embedTokens(predictedToken)
            eEmbed = eEmbed * MLXArray(g4m.languageModel.model.embedScale, dtype: eEmbed.dtype)
        } else {
            eEmbed = model.embedTokens(predictedToken)
            eEmbed = eEmbed * MLXArray(model.embedScale, dtype: eEmbed.dtype)
        }

        let assistantPosOffset: Gemma4PositionOffset
        switch posOffset ?? .scalar(0) {
        case .scalar(let off):
            assistantPosOffset = .scalar(off + inputLen - 1)
        case .batch(let offArr):
            assistantPosOffset = .batch(offArr + inputLen - 1)
        }

        // Use numMTPDraftTokens (default 4) so we generate 4 draft predictions per pass.
        // Previously this was (mtpCaches?.count ?? 0) + 2 = 0 + 2 = 2, meaning only 2 drafts
        // were ever generated despite numMTPTokens=4 in MTPTokenIterator — a 2x deficit.
        let mtpDepth = numMTPDraftTokens

        let headLogits = runMTPHead(
            hLast: hLast,
            eEmbed: eEmbed,
            posOffset: assistantPosOffset,
            backboneDim: backboneDim,
            cache: cache,
            depth: mtpDepth
        )
        return [mainLogits] + headLogits
    }

    public func makeMTPCaches(parameters: GenerateParameters?) -> [[KVCache]] {
        return [] // Assistant does not maintain its own KV cache, it uses the main model's cache
    }

    public var loraLayers: [Module] {
        model.layers.map { $0.selfAttn }
    }
}
