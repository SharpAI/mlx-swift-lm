// SharpAI fork overlay: exaone4_5 configuration. Not part of upstream mlx-swift-lm.
//
// The text dimensions are decoded leniently, then converted to the strict Exaone4Configuration
// from MLXLLM, so the text backbone can reuse Exaone4Model.
// UNVERIFIED: defaults for omitted fields follow the HF Exaone4 convention. Check them against a
// real exaone4_5 config.json. No reference config is in the repo.
// UNVERIFIED: the nested `text_config` and `vision_config` keys follow the Gemma VLM layout.

import Foundation
import MLXLLM
import MLXLMCommon

public enum SharpAIExaone45Error: LocalizedError {
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its vision tower and forward pass "
                + "are not implemented yet."
        }
    }
}

/// Text dimensions for exaone4_5. Optional fields get HF Exaone4 defaults.
public struct SharpAIExaone45TextDimensions: Decodable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let tieWordEmbeddings: Bool
    public let ropeScaling: [String: StringOrNumber]?
    public let slidingWindow: Int?
    public let slidingWindowPattern: String?

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeScaling = "rope_scaling"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let hidden = try container.decode(Int.self, forKey: .hiddenSize)
        let heads = try container.decode(Int.self, forKey: .attentionHeads)

        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = hidden
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = heads
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? (hidden / max(heads, 1))
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        slidingWindowPattern = try container.decodeIfPresent(
            String.self, forKey: .slidingWindowPattern)
    }
}

/// Vision tower dimensions. The ViT uses grouped-query attention, so kv heads may differ from heads.
/// UNVERIFIED: field names beyond the core ones follow common Qwen-VL style. Confirm them.
public struct SharpAIExaone45VisionDimensions: Decodable, Sendable {
    public let hiddenSize: Int
    public let depth: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let patchSize: Int?
    public let spatialMergeSize: Int?
    public let outHiddenSize: Int?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case depth = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case patchSize = "patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case outHiddenSize = "out_hidden_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let heads = try container.decode(Int.self, forKey: .attentionHeads)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        depth = try container.decode(Int.self, forKey: .depth)
        attentionHeads = heads
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize)
        spatialMergeSize = try container.decodeIfPresent(Int.self, forKey: .spatialMergeSize)
        outHiddenSize = try container.decodeIfPresent(Int.self, forKey: .outHiddenSize)
    }
}

/// Encodes the text dimensions with the keys that Exaone4Configuration decodes.
private struct SharpAIExaone45TextPayload: Encodable {
    let text: SharpAIExaone45TextDimensions

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeScaling = "rope_scaling"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(text.vocabSize, forKey: .vocabSize)
        try container.encode(text.hiddenSize, forKey: .hiddenSize)
        try container.encode(text.hiddenLayers, forKey: .hiddenLayers)
        try container.encode(text.intermediateSize, forKey: .intermediateSize)
        try container.encode(text.attentionHeads, forKey: .attentionHeads)
        try container.encode(text.kvHeads, forKey: .kvHeads)
        try container.encode(text.headDim, forKey: .headDim)
        try container.encode(text.rmsNormEps, forKey: .rmsNormEps)
        try container.encode(text.maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(text.ropeTheta, forKey: .ropeTheta)
        try container.encode(text.tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encodeIfPresent(text.ropeScaling, forKey: .ropeScaling)
        try container.encodeIfPresent(text.slidingWindow, forKey: .slidingWindow)
        try container.encodeIfPresent(text.slidingWindowPattern, forKey: .slidingWindowPattern)
    }
}

public struct SharpAIExaone45Configuration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let textDimensions: SharpAIExaone45TextDimensions
    public let visionDimensions: SharpAIExaone45VisionDimensions?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "exaone4_5"

        if container.contains(.textConfig) {
            textDimensions = try container.decode(
                SharpAIExaone45TextDimensions.self, forKey: .textConfig)
        } else {
            textDimensions = try SharpAIExaone45TextDimensions(from: decoder)
        }

        visionDimensions = try container.decodeIfPresent(
            SharpAIExaone45VisionDimensions.self, forKey: .visionConfig)
    }

    /// Strict text config for the existing Exaone4Model code.
    public func exaone4TextConfiguration() throws -> Exaone4Configuration {
        let data = try JSONEncoder().encode(SharpAIExaone45TextPayload(text: textDimensions))
        return try JSONDecoder().decode(Exaone4Configuration.self, from: data)
    }

    public func validateModelConfiguration() throws {
        let text = textDimensions
        let dims = [
            text.vocabSize, text.hiddenSize, text.hiddenLayers, text.intermediateSize,
            text.attentionHeads, text.kvHeads, text.headDim, text.maxPositionEmbeddings,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "exaone4_5: text sizes must be positive")
        }
        guard text.attentionHeads % text.kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "exaone4_5: text num_attention_heads must be divisible by num_key_value_heads")
        }

        if let vision = visionDimensions {
            let visionDims = [
                vision.hiddenSize, vision.depth, vision.attentionHeads, vision.kvHeads,
            ]
            guard visionDims.allSatisfy({ $0 > 0 }) else {
                throw ModelFactoryError.invalidConfiguration(
                    "exaone4_5: vision sizes must be positive")
            }
            guard vision.attentionHeads % vision.kvHeads == 0 else {
                throw ModelFactoryError.invalidConfiguration(
                    "exaone4_5: vision num_attention_heads must be divisible by num_key_value_heads")
            }
        }
    }
}
