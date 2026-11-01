// SharpAI fork overlay: openpangu_vl configuration. Not part of upstream mlx-swift-lm.
//
// Field names and values follow scripts/conformance/offline/fixtures/openpangu_vl/
// FreedomIntelligence__openPangu-VL-7B.json. Text fields sit at the top level. The vision block
// uses the Qwen2.5-VL field set (window_size, fullatt_block_indexes). The Qwen25VL vision tower is
// fileprivate, so this overlay has no ViT and no text decoder yet.

import Foundation
import MLXLMCommon

public enum SharpAIOpenpanguVlError: LocalizedError {
    case forwardNotImplemented(String)
    case duplicateWeightKey(String)
    case visionNotImplemented

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass is not implemented yet "
                + "(text decoder and ViT missing)."
        case .duplicateWeightKey(let key):
            return "Duplicate weight key after openpangu_vl sanitize: \(key)"
        case .visionNotImplemented:
            return
                "openpangu_vl image and video input needs the ViT, which is not implemented yet."
        }
    }
}

/// Vision block of the openpangu_vl config.
public struct SharpAIOpenpanguVlVisionConfiguration: Decodable, Sendable {
    public let depth: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHeads: Int
    public let inChannels: Int
    public let outHiddenSize: Int
    public let patchSize: Int
    public let spatialMergeSize: Int
    public let temporalPatchSize: Int
    public let windowSize: Int
    public let fullAttentionBlockIndexes: [Int]

    enum CodingKeys: String, CodingKey {
        case depth
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHeads = "num_heads"
        case inChannels = "in_chans"
        case outHiddenSize = "out_hidden_size"
        case patchSize = "patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case temporalPatchSize = "temporal_patch_size"
        case windowSize = "window_size"
        case fullAttentionBlockIndexes = "fullatt_block_indexes"
    }
}

public struct SharpAIOpenpanguVlConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    /// Read from the top-level "bias" key. Covers the attention projections.
    public let attentionBias: Bool
    public let rmsNormEps: Double
    public let ropeTheta: Double
    public let maxPositionEmbeddings: Int
    public let tieWordEmbeddings: Bool
    /// From rope_scaling.mrope_section. Empty when absent.
    public let mropeSection: [Int]
    public let mropeInterleaved: Bool
    public let vision: SharpAIOpenpanguVlVisionConfiguration?

    public var headDim: Int {
        attentionHeads > 0 ? hiddenSize / attentionHeads : 0
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case attentionBias = "bias"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeScaling = "rope_scaling"
        case visionConfig = "vision_config"
    }

    private struct RopeScaling: Decodable {
        let mropeSection: [Int]?
        let mropeInterleaved: Bool?

        enum CodingKeys: String, CodingKey {
            case mropeSection = "mrope_section"
            case mropeInterleaved = "mrope_interleaved"
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "openpangu_vl"
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        // UNVERIFIED: a missing KV head count is read as plain multi-head attention.
        kvHeads =
            try container.decodeIfPresent(Int.self, forKey: .kvHeads)
            ?? attentionHeads
        attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        rmsNormEps = try container.decode(Double.self, forKey: .rmsNormEps)
        ropeTheta = try container.decode(Double.self, forKey: .ropeTheta)
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        // UNVERIFIED: a missing tie_word_embeddings flag is read as false.
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        let rope = try container.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling)
        mropeSection = rope?.mropeSection ?? []
        mropeInterleaved = rope?.mropeInterleaved ?? false

        vision = try container.decodeIfPresent(
            SharpAIOpenpanguVlVisionConfiguration.self, forKey: .visionConfig)
    }

    public func validateModelConfiguration() throws {
        let dims = [
            vocabularySize, hiddenSize, hiddenLayers, intermediateSize,
            attentionHeads, kvHeads,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "openpangu_vl: vocab, hidden, layer, MLP and head sizes must be positive")
        }
        guard attentionHeads % kvHeads == 0, hiddenSize % attentionHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "openpangu_vl: hidden size must divide by attention heads, "
                    + "and attention heads by kv heads")
        }
        if !mropeSection.isEmpty {
            guard mropeSection.reduce(0, +) == headDim / 2 else {
                throw ModelFactoryError.invalidConfiguration(
                    "openpangu_vl: mrope_section must sum to half the head dimension")
            }
        }
        guard let vision = vision else {
            throw ModelFactoryError.invalidConfiguration(
                "openpangu_vl: vision_config is required")
        }
        guard vision.depth > 0, vision.numHeads > 0, vision.hiddenSize > 0,
            vision.hiddenSize % vision.numHeads == 0,
            vision.patchSize > 0, vision.spatialMergeSize > 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "openpangu_vl: vision depth, heads, hidden, patch and merge sizes must be positive, "
                    + "with hidden divisible by heads")
        }
    }
}
