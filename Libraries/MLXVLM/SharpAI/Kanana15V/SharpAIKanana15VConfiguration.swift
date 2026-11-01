// SharpAI fork overlay: kanana-1.5-v (KananaVForConditionalGeneration) configuration. Not part of upstream mlx-swift-lm.
//
// Key names match scripts/conformance/offline/fixtures/kanana-1.5-v/kakaocorp__kanana-1.5-v-3b-instruct.json.
// The text tower decodes as LlamaConfiguration. The projector (dynamic-c-abs, RegStage) is parsed but not built.

import Foundation
import MLXLLM
import MLXLMCommon

public enum SharpAIKanana15VError: LocalizedError {
    case projectorNotImplemented
    case unsupportedActivation(String)

    public var errorDescription: String? {
        switch self {
        case .projectorNotImplemented:
            return
                "kanana-1.5-v: the dynamic-c-abs (RegStage) projector is not implemented. UNVERIFIED: no reference source."
        case .unsupportedActivation(let name):
            return "kanana-1.5-v: unsupported vision hidden_act \(name), expected quick_gelu"
        }
    }
}

/// Qwen2-VL style vision encoder settings (`vision_config`, encoder type `qwen2-vl-ve`).
public struct SharpAIKanana15VVisionConfiguration: Decodable, Sendable {
    public let depth: Int
    public let embedDimensions: Int
    public let hiddenSize: Int
    public let attentionHeads: Int
    public let mlpRatio: Float
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let spatialMergeSize: Int
    public let inChannels: Int
    public let hiddenAct: String

    enum CodingKeys: String, CodingKey {
        case depth
        case embedDimensions = "embed_dim"
        case hiddenSize = "hidden_size"
        case attentionHeads = "num_heads"
        case mlpRatio = "mlp_ratio"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case inChannels = "in_channels"
        case hiddenAct = "hidden_act"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        depth = try c.decode(Int.self, forKey: .depth)
        embedDimensions = try c.decode(Int.self, forKey: .embedDimensions)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 4
        patchSize = try c.decode(Int.self, forKey: .patchSize)
        temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
        spatialMergeSize = try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
        inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "quick_gelu"
    }
}

/// C-Abstractor projector settings (`projector_config`). Parsed only; see the model file.
public struct SharpAIKanana15VProjectorConfiguration: Decodable, Sendable {
    public let projectorType: String
    public let depth: Int
    public let mlpDepth: Int
    public let encoderHiddenSize: Int
    public let hiddenSize: Int
    public let outputHiddenSize: Int
    public let mergeSize: Int
    public let posEmb: Bool
    public let posEmbSize: Int?

    enum CodingKeys: String, CodingKey {
        case projectorType = "projector_type"
        case depth
        case mlpDepth = "mlp_depth"
        case encoderHiddenSize = "encoder_hidden_size"
        case hiddenSize = "hidden_size"
        case outputHiddenSize = "output_hidden_size"
        case mergeSize = "merge_size"
        case posEmb = "pos_emb"
        case posEmbSize = "pos_emb_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        projectorType = try c.decode(String.self, forKey: .projectorType)
        depth = try c.decode(Int.self, forKey: .depth)
        mlpDepth = try c.decodeIfPresent(Int.self, forKey: .mlpDepth) ?? 2
        encoderHiddenSize = try c.decode(Int.self, forKey: .encoderHiddenSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        outputHiddenSize = try c.decode(Int.self, forKey: .outputHiddenSize)
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        posEmb = try c.decodeIfPresent(Bool.self, forKey: .posEmb) ?? false
        posEmbSize = try c.decodeIfPresent(Int.self, forKey: .posEmbSize)
    }
}

/// Top-level kanana-1.5-v configuration.
public struct SharpAIKanana15VConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    public static let expectedModelType = "kanana-1.5-v"

    public let modelType: String
    /// Llama text tower (`text_config`, architecture `LlamaForCausalLM`).
    public let textConfiguration: LlamaConfiguration
    /// Read from `text_config` because `LlamaConfiguration` keeps this field internal.
    public let tieWordEmbeddings: Bool
    public let visionConfiguration: SharpAIKanana15VVisionConfiguration
    public let projectorConfiguration: SharpAIKanana15VProjectorConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case projectorConfiguration = "projector_config"
    }

    enum TextKeys: String, CodingKey {
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        textConfiguration = try c.decode(LlamaConfiguration.self, forKey: .textConfiguration)

        let text = try c.nestedContainer(keyedBy: TextKeys.self, forKey: .textConfiguration)
        tieWordEmbeddings = try text.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        visionConfiguration = try c.decode(
            SharpAIKanana15VVisionConfiguration.self, forKey: .visionConfiguration)
        projectorConfiguration = try c.decode(
            SharpAIKanana15VProjectorConfiguration.self, forKey: .projectorConfiguration)
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "kanana-1.5-v: model_type must be \(Self.expectedModelType), got \(modelType)")
        }

        let vision = visionConfiguration
        guard vision.hiddenAct == "quick_gelu" else {
            throw SharpAIKanana15VError.unsupportedActivation(vision.hiddenAct)
        }
        guard vision.attentionHeads > 0, vision.embedDimensions % vision.attentionHeads == 0,
            (vision.embedDimensions / vision.attentionHeads) % 4 == 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "kanana-1.5-v: vision head dimension must be divisible by 4")
        }
    }
}
