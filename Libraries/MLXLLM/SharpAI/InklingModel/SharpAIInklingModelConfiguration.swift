// SharpAI fork overlay: inkling_model configuration. Not part of upstream mlx-swift-lm.
//
// Sources: transformers `models/inkling/configuration_inkling.py` (field defaults and the
// derived layer patterns) and the `text_config` block of thinkingmachines/Inkling config.json.
// The published multimodal checkpoint (`inkling_mm_model`) nests the text settings under
// `text_config`. A flat text config is accepted too. Vision, audio and MTP settings are ignored.

import Foundation

public enum SharpAIInklingModelError: LocalizedError {
    case incrementalDecodeNotImplemented(String)
    case corruptTokenHistoryCache

    public var errorDescription: String? {
        switch self {
        case .corruptTokenHistoryCache:
            return "Corrupt prompt cache: inkling token history must hold one token tensor."
        case .incrementalDecodeNotImplemented(let type):
            return
                "Model type \(type) runs full-sequence forward only. KV-cache and short-conv "
                + "state for incremental decoding is not implemented yet."
        }
    }
}

public struct SharpAIInklingModelConfiguration: Decodable, Sendable {
    public let modelType: String
    public let architectures: [String]
    public let text: SharpAIInklingTextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "inkling_model"
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        if let nested = try c.decodeIfPresent(
            SharpAIInklingTextConfiguration.self, forKey: .textConfig)
        {
            text = nested
        } else {
            text = try SharpAIInklingTextConfiguration(from: decoder)
        }
    }
}

/// Text backbone settings. Defaults are the transformers `InklingTextConfig` defaults.
public struct SharpAIInklingTextConfiguration: Decodable, Sendable {
    public let vocabularySize: Int
    /// Rows of the unembedding that carry real tokens. Logits past it are dropped.
    public let unpaddedVocabularySize: Int?
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let swaAttentionHeads: Int
    public let swaKVHeads: Int
    public let swaHeadDim: Int
    /// Window of the `hybrid_sliding` layers. Also their relative-bias extent.
    public let slidingWindow: Int
    /// Per-head width of the relative states mixed into the position bias.
    public let dRel: Int
    /// Backward distance, in tokens, covered by the bias in full-attention layers.
    public let relExtent: Int
    /// Position from which full-attention logits are log-scaled. `nil` disables the scaling.
    public let logScalingNFloor: Int?
    public let logScalingAlpha: Float
    /// `true` for `hybrid_sliding` layers, `false` for full (`hybrid`) attention layers.
    public let slidingLayers: [Bool]
    /// `true` for dense MLP layers, `false` for sparse MoE layers.
    public let denseLayers: [Bool]
    public let sconvKernelSize: Int
    public let rmsNormEps: Float
    /// Width of the dense MLP.
    public let intermediateSize: Int
    /// Width of each routed and shared expert.
    public let moeIntermediateSize: Int
    public let routedExperts: Int
    public let expertsPerToken: Int
    public let sharedExperts: Int
    public let routeScale: Float
    /// The final hidden states are divided by this before the LM head.
    public let logitsMupWidthMultiplier: Float

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case unpaddedVocabularySize = "unpadded_vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case swaAttentionHeads = "swa_num_attention_heads"
        case swaKVHeads = "swa_num_key_value_heads"
        case swaHeadDim = "swa_head_dim"
        case slidingWindow = "sliding_window_size"
        case dRel = "d_rel"
        case relExtent = "rel_extent"
        case logScalingNFloor = "log_scaling_n_floor"
        case logScalingAlpha = "log_scaling_alpha"
        case localLayerIDs = "local_layer_ids"
        case denseMLPIndex = "dense_mlp_idx"
        case mlpLayerTypes = "mlp_layer_types"
        case sconvKernelSize = "sconv_kernel_size"
        case convKernelSize = "conv_kernel_size"
        case rmsNormEps = "rms_norm_eps"
        case intermediateSize = "intermediate_size"
        case denseIntermediateSize = "dense_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case routedExperts = "n_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case sharedExperts = "n_shared_experts"
        case routeScale = "route_scale"
        case logitsMupWidthMultiplier = "logits_mup_width_multiplier"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 201024
        unpaddedVocabularySize = try c.decodeIfPresent(Int.self, forKey: .unpaddedVocabularySize)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 6144
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 66
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 64
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        swaAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .swaAttentionHeads) ?? 64
        swaKVHeads = try c.decodeIfPresent(Int.self, forKey: .swaKVHeads) ?? 16
        swaHeadDim = try c.decodeIfPresent(Int.self, forKey: .swaHeadDim) ?? 128
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        dRel = try c.decodeIfPresent(Int.self, forKey: .dRel) ?? 16
        relExtent = try c.decodeIfPresent(Int.self, forKey: .relExtent) ?? 1024
        logScalingNFloor = try c.decodeIfPresent(Int.self, forKey: .logScalingNFloor)
        logScalingAlpha = try c.decodeIfPresent(Float.self, forKey: .logScalingAlpha) ?? 0.1
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        sconvKernelSize =
            try c.decodeIfPresent(Int.self, forKey: .sconvKernelSize)
            ?? c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 4
        routedExperts = try c.decodeIfPresent(Int.self, forKey: .routedExperts) ?? 256
        expertsPerToken = try c.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? 6
        sharedExperts = try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? 2
        routeScale = try c.decodeIfPresent(Float.self, forKey: .routeScale) ?? 8.0
        logitsMupWidthMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .logitsMupWidthMultiplier) ?? 24.0

        // transformers: with `dense_intermediate_size`, `intermediate_size` is the expert width.
        let rawIntermediate = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 24576
        if let dense = try c.decodeIfPresent(Int.self, forKey: .denseIntermediateSize) {
            intermediateSize = dense
            moeIntermediateSize = rawIntermediate
        } else {
            intermediateSize = rawIntermediate
            moeIntermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 3072
        }

        // Layer patterns, as transformers derives them. Without `local_layer_ids`, every layer
        // whose 1-based index is a multiple of 6 is full attention.
        let layers = hiddenLayers
        if let local = try c.decodeIfPresent([Int].self, forKey: .localLayerIDs) {
            let localSet = Set(local)
            slidingLayers = (0 ..< layers).map { localSet.contains($0) }
        } else {
            slidingLayers = (0 ..< layers).map { ($0 + 1) % 6 != 0 }
        }

        if let types = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes) {
            guard types.count == layers else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: decoder.codingPath,
                        debugDescription:
                            "mlp_layer_types has \(types.count) entries, expected \(layers)"))
            }
            denseLayers = types.map { $0 == "dense" }
        } else {
            let denseIndex = try c.decodeIfPresent(Int.self, forKey: .denseMLPIndex) ?? 0
            denseLayers = (0 ..< layers).map { $0 < denseIndex }
        }

        guard kvHeads > 0, swaKVHeads > 0, sconvKernelSize >= 1 else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription:
                        "num_key_value_heads, swa_num_key_value_heads and sconv_kernel_size must be positive"
                ))
        }
        guard attentionHeads % kvHeads == 0, swaAttentionHeads % swaKVHeads == 0 else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "attention heads must be a multiple of kv heads"))
        }
    }
}
