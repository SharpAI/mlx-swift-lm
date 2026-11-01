// SharpAI fork overlay: cosmos3_edge configuration. Not part of upstream mlx-swift-lm.
//
// Field names and defaults follow the Hugging Face transformers config classes
// (`Cosmos3EdgeConfig`, `Cosmos3EdgeTextConfig`), fetched from
// https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/cosmos3_edge/configuration_cosmos3_edge.py
// The vision sub-config and projector are not mapped yet.

import Foundation
import MLXLMCommon

public enum SharpAICosmos3EdgeError: LocalizedError {
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its model class is not implemented yet "
                + "(vision tower, processor and weight loading are missing)."
        }
    }
}

public struct SharpAICosmos3EdgeConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "cosmos3_edge"

    public let modelType: String
    public let architectures: [String]
    public let textConfiguration: SharpAICosmos3EdgeTextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case textConfiguration = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType)
            ?? Self.expectedModelType
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        textConfiguration =
            try c.decodeIfPresent(
                SharpAICosmos3EdgeTextConfiguration.self, forKey: .textConfiguration)
            ?? SharpAICosmos3EdgeTextConfiguration()
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "cosmos3_edge: model_type must be \(Self.expectedModelType), got \(modelType)")
        }
    }
}

/// `text_config` of a cosmos3_edge checkpoint (HF `Cosmos3EdgeTextConfig`).
///
/// `mrope_section` and `rope_theta` come from `rope_parameters`. They have no defaults here, so a
/// text model cannot be built without them.
public struct SharpAICosmos3EdgeTextConfiguration: Decodable, Sendable {
    public var vocabularySize: Int = 131072
    public var hiddenSize: Int = 2048
    public var intermediateSize: Int = 9216
    public var hiddenLayers: Int = 28
    public var attentionHeads: Int = 16
    public var kvHeads: Int = 8
    public var headDim: Int = 128
    public var hiddenAct: String = "relu2"
    public var rmsNormEps: Float = 1e-5
    public var attentionBias: Bool = false
    public var mlpBias: Bool = false
    public var ropeTheta: Double?
    public var mropeSection: [Int]?

    public init() {}

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 131072
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 9216
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 28
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "relu2"
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        if let rope = try c.decodeIfPresent(RopeContainer.self, forKey: .ropeParameters) {
            ropeTheta = rope.ropeTheta
            mropeSection = rope.mropeSection
        }
    }

    private struct RopeContainer: Decodable {
        let ropeTheta: Double?
        let mropeSection: [Int]?

        enum Keys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case mropeSection = "mrope_section"
        }

        init(from decoder: Decoder) throws {
            let r = try decoder.container(keyedBy: Keys.self)
            ropeTheta = try r.decodeIfPresent(Double.self, forKey: .ropeTheta)
            mropeSection = try r.decodeIfPresent([Int].self, forKey: .mropeSection)
        }
    }
}
