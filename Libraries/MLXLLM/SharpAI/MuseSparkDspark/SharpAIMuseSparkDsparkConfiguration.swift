// SharpAI fork overlay: muse_spark_dspark configuration. Not part of upstream mlx-swift-lm.
//
// Field names and defaults follow vLLM's MuseSparkDSparkConfig and MuseSpark1xTextConfig
// (vllm/transformers_utils/configs/muse_spark.py). UNVERIFIED: the defaults for
// target_hidden_size, num_target_layers and the confidence-head flags, which the source
// does not state. The MoE defaults (is_enabled true) follow MoEArgs in the same file.

import Foundation
import MLXLMCommon

public enum SharpAIMuseSparkDsparkError: LocalizedError {
    case targetNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .targetNotImplemented(let type):
            return
                "Model type \(type) is a DSpark drafter. Its target model (Muse Spark) is not "
                + "implemented, so the drafter cannot load on its own."
        }
    }
}

/// Gain center for a post-norm, as calculate_gain_center in vLLM's muse_spark/common/layers.py.
func museGainCenter(_ type: String, residualId: Int, numLayers: Int) -> Float? {
    let pairedIndex = Float(2 * ((residualId + 1) / 2))
    switch type {
    case "one": return 1
    case "zero": return 0
    case "inverse_residual_index": return 1 / pairedIndex
    case "inverse_residual_index_v2", "inverse_residual_index_p1":
        return 1 / Float(residualId + 1)
    case "inverse_sqrt_residual_index": return 1 / pairedIndex.squareRoot()
    case "inverse_sqrt_layers": return 1 / Float(numLayers).squareRoot()
    default: return nil
    }
}

private struct SharpAIMuseSparkDsparkMoEFields: Decodable {
    let isEnabled: Bool?
    let interleaveMoeLayerStep: Int?
    let nDenseLayersBeforeMoe: Int?

    enum CodingKeys: String, CodingKey {
        case isEnabled = "is_enabled"
        case interleaveMoeLayerStep = "interleave_moe_layer_step"
        case nDenseLayersBeforeMoe = "n_dense_layers_before_moe"
    }
}

private struct SharpAIMuseSparkDsparkMetapFields: Decodable {
    let useMetap: Bool?
    let metapMode: String?
    let baseWidth: Float?
    let routerHasOneInfDim: Bool?

    enum CodingKeys: String, CodingKey {
        case useMetap = "use_metap"
        case metapMode = "metap_mode"
        case baseWidth = "base_width"
        case routerHasOneInfDim = "router_has_one_inf_dim"
    }
}

/// Top-level `config.json` for `model_type: muse_spark_dspark`.
public struct SharpAIMuseSparkDsparkConfiguration: Decodable, Sendable {
    public static let expectedModelType = "muse_spark_dspark"

    public let modelType: String
    public let architectures: [String]
    public let vocabSize: Int
    public let draftVocabSize: Int
    public let hiddenSize: Int
    public let intermediateSizeMlp: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let postNormEps: Float
    public let normAffine: Bool
    public let postNormAffine: Bool
    public let postNormGainCenterType: String
    public let normZeroCenteredGamma: Bool
    public let useQkNorm: Bool
    public let useAttnOGate: Bool
    public let useAttnONorm: Bool
    public let useSqrtAttnScaling: Bool
    /// Effective RoPE switch. `draft_position_mode` wins over `use_rope` when present.
    public let useRope: Bool
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    public let everyNLayersNope: Int?
    /// Per-entry sliding window from `global_attn_cfg`. 0 means full attention.
    public let globalAttnSlidingWindows: [Int]
    /// True when `global_attn_cfg` has a string entry (e.g. "gdn2") or a non-array value.
    /// Validation rejects these, since the drafter cannot run those layers.
    public let globalAttnHasUnsupportedEntries: Bool
    public let attnTemperatureTuningFloorScale: Float?
    public let attnTemperatureTuningQScale: Float?
    public let attnTemperatureTuningLayers: String?
    public let useResidualGate: Bool
    public let residualDtype: String
    public let useFfnLearnableScales: Bool
    public let moeIsEnabled: Bool
    public let moeInterleaveStep: Int
    public let moeDenseLayersBeforeMoe: Int
    public let useMetap: Bool
    public let metapMode: String
    public let metapBaseWidth: Float
    public let metapRouterHasOneInfDim: Bool
    public let outputSoftCapTemp: Float?
    public let targetLayerIds: [Int]
    public let numTargetLayers: Int
    public let targetHiddenSize: Int
    public let markovRank: Int
    public let enableConfidenceHead: Bool
    public let confidenceHeadWithMarkov: Bool
    public let maskTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case vocabSize = "vocab_size"
        case draftVocabSize = "draft_vocab_size"
        case hiddenSize = "hidden_size"
        case ffnDimMultiplier = "ffn_dim_multiplier"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case rmsPostNormEps = "rms_post_norm_eps"
        case normAffine = "norm_affine"
        case postNormAffine = "post_norm_affine"
        case postNormGainCenterType = "post_norm_gain_center_type"
        case normZeroCenteredGamma = "norm_zero_centered_gamma"
        case useQkNorm = "use_qk_norm"
        case useAttnOGate = "use_attn_o_gate"
        case useAttnONorm = "use_attn_o_norm"
        case useSqrtAttnScaling = "use_sqrt_attn_scaling"
        case useRope = "use_rope"
        case draftPositionMode = "draft_position_mode"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case everyNLayersNope = "every_n_layers_nope"
        case globalAttnCfg = "global_attn_cfg"
        case attnTemperatureTuningFloorScale = "attn_temperature_tuning_floor_scale"
        case attnTemperatureTuningQScale = "attn_temperature_tuning_q_scale_constant"
        case attnTemperatureTuningLayers = "attn_temperature_tuning_layers"
        case useResidualGate = "use_residual_gate"
        case residualDtype = "residual_dtype"
        case useFfnLearnableScales = "use_ffn_learnable_scales"
        case moe
        case metap
        case outputSoftCapTemp = "output_soft_cap_temp"
        case targetLayerIds = "target_layer_ids"
        case numTargetLayers = "num_target_layers"
        case targetHiddenSize = "target_hidden_size"
        case markovRank = "markov_rank"
        case enableConfidenceHead = "enable_confidence_head"
        case confidenceHeadWithMarkov = "confidence_head_with_markov"
        case maskTokenId = "mask_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType)
            ?? Self.expectedModelType
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 202048
        draftVocabSize = try c.decodeIfPresent(Int.self, forKey: .draftVocabSize) ?? vocabSize
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 5120
        let ffnDimMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .ffnDimMultiplier) ?? 1.0
        intermediateSizeMlp = Int(ffnDimMultiplier * Float(hiddenSize))
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 48
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 40
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        let rmsPostNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsPostNormEps)
        postNormEps = rmsPostNormEps ?? rmsNormEps
        normAffine = try c.decodeIfPresent(Bool.self, forKey: .normAffine) ?? true
        postNormAffine = try c.decodeIfPresent(Bool.self, forKey: .postNormAffine) ?? true
        postNormGainCenterType =
            try c.decodeIfPresent(String.self, forKey: .postNormGainCenterType) ?? "one"
        normZeroCenteredGamma =
            try c.decodeIfPresent(Bool.self, forKey: .normZeroCenteredGamma) ?? false
        useQkNorm = try c.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
        useAttnOGate = try c.decodeIfPresent(Bool.self, forKey: .useAttnOGate) ?? false
        useAttnONorm = try c.decodeIfPresent(Bool.self, forKey: .useAttnONorm) ?? false
        useSqrtAttnScaling =
            try c.decodeIfPresent(Bool.self, forKey: .useSqrtAttnScaling) ?? false
        let decodedUseRope = try c.decodeIfPresent(Bool.self, forKey: .useRope) ?? true
        if let mode = try c.decodeIfPresent(String.self, forKey: .draftPositionMode) {
            useRope = mode == "rope"
        } else {
            useRope = decodedUseRope
        }
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500000
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        everyNLayersNope = try c.decodeIfPresent(Int.self, forKey: .everyNLayersNope)
        let windowEntries = try? c.decodeIfPresent([StringOrNumber].self, forKey: .globalAttnCfg)
        let globalAttnIsNull = (try? c.decodeNil(forKey: .globalAttnCfg)) ?? true
        globalAttnHasUnsupportedEntries =
            (windowEntries == nil && !globalAttnIsNull)
            || (windowEntries ?? []).contains(where: { (entry: StringOrNumber) -> Bool in
                if case .int = entry { return false }
                return true
            })
        globalAttnSlidingWindows = (windowEntries ?? []).map { (entry: StringOrNumber) -> Int in
            if case .int(let window) = entry, window > 0 { return window }
            return 0
        }
        attnTemperatureTuningFloorScale =
            try c.decodeIfPresent(Float.self, forKey: .attnTemperatureTuningFloorScale)
        attnTemperatureTuningQScale =
            try c.decodeIfPresent(Float.self, forKey: .attnTemperatureTuningQScale)
        attnTemperatureTuningLayers =
            try? c.decodeIfPresent(String.self, forKey: .attnTemperatureTuningLayers)
        useResidualGate = try c.decodeIfPresent(Bool.self, forKey: .useResidualGate) ?? false
        residualDtype = try c.decodeIfPresent(String.self, forKey: .residualDtype) ?? "bf16"
        useFfnLearnableScales =
            try c.decodeIfPresent(Bool.self, forKey: .useFfnLearnableScales) ?? false

        let moe = try c.decodeIfPresent(SharpAIMuseSparkDsparkMoEFields.self, forKey: .moe)
        moeIsEnabled = moe?.isEnabled ?? true
        moeInterleaveStep = moe?.interleaveMoeLayerStep ?? 1
        moeDenseLayersBeforeMoe = moe?.nDenseLayersBeforeMoe ?? 0

        let metap = try c.decodeIfPresent(SharpAIMuseSparkDsparkMetapFields.self, forKey: .metap)
        useMetap = metap?.useMetap ?? false
        metapMode = metap?.metapMode ?? "mup"
        metapBaseWidth = metap?.baseWidth ?? 256
        metapRouterHasOneInfDim = metap?.routerHasOneInfDim ?? true

        outputSoftCapTemp = try c.decodeIfPresent(Float.self, forKey: .outputSoftCapTemp)
        targetLayerIds = try c.decodeIfPresent([Int].self, forKey: .targetLayerIds) ?? []
        numTargetLayers =
            try c.decodeIfPresent(Int.self, forKey: .numTargetLayers) ?? targetLayerIds.count
        targetHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .targetHiddenSize) ?? hiddenSize
        markovRank = try c.decode(Int.self, forKey: .markovRank)
        enableConfidenceHead =
            try c.decodeIfPresent(Bool.self, forKey: .enableConfidenceHead) ?? false
        confidenceHeadWithMarkov =
            try c.decodeIfPresent(Bool.self, forKey: .confidenceHeadWithMarkov) ?? false
        maskTokenId = try c.decodeIfPresent(Int.self, forKey: .maskTokenId)
    }

    /// Whether layer `layer` is an MoE layer. The Swift drafter has no MoE block.
    public func isMoELayer(_ layer: Int) -> Bool {
        moeIsEnabled && moeInterleaveStep > 0 && (layer + 1) % moeInterleaveStep == 0
            && layer >= moeDenseLayersBeforeMoe
    }

    /// Layer index counted from the output end, as vLLM counts it for NoPE and window lookup.
    public func countBackwardLayerId(_ layer: Int) -> Int {
        guard let every = everyNLayersNope else { return layer }
        return layer + (every - numHiddenLayers % every)
    }

    public func layerUsesRope(_ layer: Int) -> Bool {
        guard useRope else { return false }
        guard let every = everyNLayersNope else { return true }
        return (numHiddenLayers - layer - 1) % every != 0
    }

    /// Sliding window for `layer`, or nil for full attention.
    public func slidingWindow(forLayer layer: Int) -> Int? {
        guard !globalAttnSlidingWindows.isEmpty else { return nil }
        let window =
            globalAttnSlidingWindows[countBackwardLayerId(layer) % globalAttnSlidingWindows.count]
        return window > 0 ? window : nil
    }

    /// Scalar multiplier on the output logits, from Metap. 1 when Metap is off.
    public var outputMultiplier: Float {
        guard useMetap, metapRouterHasOneInfDim else { return 1 }
        let widthMultiplier = Float(hiddenSize) / metapBaseWidth
        return metapMode == "mup" ? 1 / widthMultiplier : pow(widthMultiplier, -0.5)
    }

    public func validateModelConfiguration() throws {
        func invalid(_ detail: String) -> ModelFactoryError {
            ModelFactoryError.invalidConfiguration("muse_spark_dspark: " + detail)
        }

        guard modelType == Self.expectedModelType else {
            throw invalid("model_type must be \(Self.expectedModelType), got \(modelType)")
        }
        guard residualDtype == "bf16" || residualDtype == "fp32" else {
            throw invalid("residual_dtype must be bf16 or fp32, got \(residualDtype)")
        }
        guard vocabSize > 0, draftVocabSize > 0, hiddenSize > 0, numHiddenLayers > 0 else {
            throw invalid("vocab_size, draft_vocab_size, hidden_size and layers must be positive")
        }
        guard headDim > 0, numAttentionHeads > 0, numKeyValueHeads > 0,
            numAttentionHeads % numKeyValueHeads == 0
        else {
            throw invalid("attention heads must divide evenly into key/value heads")
        }
        guard museGainCenter(postNormGainCenterType, residualId: 1, numLayers: numHiddenLayers)
            != nil
        else {
            throw invalid("unsupported post_norm_gain_center_type \(postNormGainCenterType)")
        }
        guard markovRank > 0 else {
            throw invalid("markov_rank must be positive")
        }
        guard !targetLayerIds.isEmpty, numTargetLayers == targetLayerIds.count, targetHiddenSize > 0
        else {
            throw invalid("requires non-empty target_layer_ids and a matching num_target_layers")
        }
        if let every = everyNLayersNope, every <= 0 {
            throw invalid("every_n_layers_nope must be positive")
        }
        if let temp = outputSoftCapTemp, temp <= 0 {
            throw invalid("output_soft_cap_temp must be positive")
        }
        if useMetap, metapMode != "mup", metapMode != "sp" {
            throw invalid("unknown metap_mode \(metapMode)")
        }

        var unsupported: [String] = []
        if (0 ..< numHiddenLayers).contains(where: isMoELayer) {
            unsupported.append("MoE layers")
        }
        if globalAttnHasUnsupportedEntries {
            unsupported.append("non-integer global_attn_cfg entries (linear-attention layers)")
        }
        if attnTemperatureTuningFloorScale != nil, attnTemperatureTuningQScale != nil,
            attnTemperatureTuningLayers == "global",
            (0 ..< numHiddenLayers).contains(where: { slidingWindow(forLayer: $0) == nil })
        {
            unsupported.append("attention temperature tuning")
        }
        if !unsupported.isEmpty {
            throw invalid("does not support " + unsupported.joined(separator: ", ") + ".")
        }
    }
}
