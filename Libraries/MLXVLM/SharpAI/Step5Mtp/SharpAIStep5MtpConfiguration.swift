// SharpAI fork overlay: step5_mtp configuration. Not part of upstream mlx-swift-lm.
//
// Decodes a step5_mtp draft config: model_type "step5_mtp" with num_nextn_predict_layers set.
// UNVERIFIED: the decode defaults, key names, and text_config nesting have no source here
// (vllm/transformers_utils/configs/step5.py is not on this machine). MoE, rope, and
// quantization fields are not decoded. The MTP block class is not written.

import Foundation
import MLXLMCommon

public enum SharpAIStep5MtpError: LocalizedError {
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass is blocked: the Step5 "
                + "decoder block (sparse attention, indexer, MoE) is not in the repo yet."
        }
    }
}

public struct SharpAIStep5MtpConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let attentionGroups: Int
    public let headDim: Int?
    public let hiddenLayers: Int
    public let nextnPredictLayers: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let zeroCentered: Bool
    public let fp32ResidualConnection: Bool
    public let tieWordEmbeddings: Bool
    public let maxPositionEmbeddings: Int?
    public let layerTypes: [String]?
    public let layerTypesWithMTP: [String]?

    /// Layer types for the dense stack plus the MTP layers, when the config provides them.
    public var layerTypesIncludingMTP: [String]? {
        layerTypesWithMTP ?? layerTypes
    }

    // UNVERIFIED: layer_types_with_mtp, fp32_residual_connection, and zero_centered
    // are not checked against a Step5 config.
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case attentionGroups = "num_attention_groups"
        case headDim = "head_dim"
        case hiddenLayers = "num_hidden_layers"
        case nextnPredictLayers = "num_nextn_predict_layers"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case zeroCentered = "zero_centered"
        case fp32ResidualConnection = "fp32_residual_connection"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layerTypes = "layer_types"
        case layerTypesWithMTP = "layer_types_with_mtp"
    }

    // UNVERIFIED: every default below is a guess, not a value from a Step5 checkpoint or vLLM.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "step5_mtp"
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 5120
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 13312
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 40
        attentionGroups = try container.decodeIfPresent(Int.self, forKey: .attentionGroups) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 48
        nextnPredictLayers =
            try container.decodeIfPresent(Int.self, forKey: .nextnPredictLayers) ?? 0
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 65536
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        zeroCentered = try container.decodeIfPresent(Bool.self, forKey: .zeroCentered) ?? true
        fp32ResidualConnection =
            try container.decodeIfPresent(Bool.self, forKey: .fp32ResidualConnection) ?? false
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        layerTypesWithMTP =
            try container.decodeIfPresent([String].self, forKey: .layerTypesWithMTP)
    }

    /// Decodes a draft config. A top-level `text_config` object is used when present.
    /// UNVERIFIED: this nesting is not checked against a Step5 config.
    public static func decode(from data: Data) throws -> SharpAIStep5MtpConfiguration {
        let wrapper = try JSONDecoder.json5().decode(TextConfigWrapper.self, from: data)
        if let nested = wrapper.textConfig {
            return nested
        }
        return try JSONDecoder.json5().decode(SharpAIStep5MtpConfiguration.self, from: data)
    }

    public func validateModelConfiguration() throws {
        let dims = [
            hiddenSize, intermediateSize, attentionHeads, attentionGroups, hiddenLayers,
            vocabularySize,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "step5_mtp: hidden, intermediate, head, group, layer and vocab sizes must be positive")
        }
        guard nextnPredictLayers > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "step5_mtp: num_nextn_predict_layers must be positive, got \(nextnPredictLayers)")
        }
        if let types = layerTypesIncludingMTP, !types.isEmpty,
            types.count < hiddenLayers + nextnPredictLayers
        {
            throw ModelFactoryError.invalidConfiguration(
                "step5_mtp: layer types must cover the dense stack and MTP layers, expected "
                    + "\(hiddenLayers + nextnPredictLayers) entries, got \(types.count)")
        }
        if let layerTypes, !layerTypes.isEmpty, let withMTP = layerTypesWithMTP,
            layerTypes != Array(withMTP.prefix(hiddenLayers))
        {
            throw ModelFactoryError.invalidConfiguration(
                "step5_mtp: layer_types must match the dense prefix of layer_types_with_mtp")
        }
    }
}

private struct TextConfigWrapper: Decodable {
    let textConfig: SharpAIStep5MtpConfiguration?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }
}
