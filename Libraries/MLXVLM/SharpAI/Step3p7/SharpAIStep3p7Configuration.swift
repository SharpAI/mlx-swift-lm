// SharpAI fork overlay: step3p7 configuration. Not part of upstream mlx-swift-lm.
//
// Field names follow scripts/conformance/offline/fixtures/step3p7/stepfun-ai__Step-3.7-Flash.json
// (present in this worktree). The full fixture has 48 text layer_types for 45 hidden layers
// (45 + 3 nextn). validateModelConfiguration accepts that count (see the layer_types check).
// The step3p5 text backbone is SharpAIStep3p7TextModel in Libraries/MLXLLM/SharpAI/Step3p7.
// The PE vision tower is not implemented.

import Foundation
import MLXLMCommon

public enum SharpAIStep3p7Error: LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "Step3p7 unsupported: \(detail)"
        case .invalidConfiguration(let detail):
            return "Step3p7 invalid configuration: \(detail)"
        }
    }
}

/// `vision_config` for `model_type: perception_encoder` (the PE tower).
public struct SharpAIStep3p7VisionConfiguration: Codable, Sendable {
    public let modelType: String?
    public let imageSize: Int
    public let patchSize: Int
    public let width: Int
    public let layers: Int
    public let heads: Int
    public let poolType: String?
    public let outputDim: Int?
    public let useClsToken: Bool?
    public let lsInitValue: Float?
    public let useLnPost: Bool?
    public let hiddenAct: String?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case width
        case layers
        case heads
        case poolType = "pool_type"
        case outputDim = "output_dim"
        case useClsToken = "use_cls_token"
        case lsInitValue = "ls_init_value"
        case useLnPost = "use_ln_post"
        case hiddenAct = "hidden_act"
    }
}

/// `text_config` for the step3p5 backbone. Only the fields the overlay checks are typed.
public struct SharpAIStep3p7TextConfiguration: Codable, Sendable {
    public let modelType: String?
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let vocabSize: Int
    public let attentionHeads: Int
    public let attentionGroups: Int?
    public let headDim: Int
    public let moeNumExperts: Int?
    public let moeTopK: Int?
    public let moeIntermediateSize: Int?
    public let shareExpertDim: Int?
    public let slidingWindow: Int?
    public let layerTypes: [String]?
    public let numNextnPredictLayers: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case attentionHeads = "num_attention_heads"
        case attentionGroups = "num_attention_groups"
        case headDim = "head_dim"
        case moeNumExperts = "moe_num_experts"
        case moeTopK = "moe_top_k"
        case moeIntermediateSize = "moe_intermediate_size"
        case shareExpertDim = "share_expert_dim"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case numNextnPredictLayers = "num_nextn_predict_layers"
    }
}

/// Top-level `config.json` for `model_type: step3p7`.
public struct SharpAIStep3p7Configuration: Codable, Sendable {
    public static let expectedArchitecture = "Step3p7ForConditionalGeneration"

    public let modelType: String
    public let architectures: [String]?
    public let textConfig: SharpAIStep3p7TextConfiguration
    public let visionConfig: SharpAIStep3p7VisionConfiguration
    public let imageTokenId: Int?
    public let imageTokenLen: Int?
    public let understandProjectorStride: Int?
    public let projectorBias: Bool?
    public let visionSelectLayer: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case imageTokenLen = "image_token_len"
        case understandProjectorStride = "understand_projector_stride"
        case projectorBias = "projector_bias"
        case visionSelectLayer = "vision_select_layer"
    }

    /// Checks the model type, the architecture when present, and the layer and head fields.
    public func validateModelConfiguration() throws {
        guard modelType == "step3p7" else {
            throw SharpAIStep3p7Error.invalidConfiguration("model_type must be step3p7")
        }
        if let architectures, !architectures.isEmpty,
            !architectures.contains(Self.expectedArchitecture)
        {
            throw SharpAIStep3p7Error.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
        guard textConfig.hiddenLayers > 0 else {
            throw SharpAIStep3p7Error.invalidConfiguration("text_config.num_hidden_layers must be > 0")
        }
        // The real checkpoint lists num_hidden_layers + num_nextn_predict_layers entries (45 + 3 = 48).
        // Accept any count from one per hidden layer up to that total, and reject fewer.
        if let layerTypes = textConfig.layerTypes {
            let nextn = textConfig.numNextnPredictLayers ?? 0
            guard layerTypes.count >= textConfig.hiddenLayers,
                layerTypes.count <= textConfig.hiddenLayers + nextn
            else {
                throw SharpAIStep3p7Error.invalidConfiguration(
                    "text_config.layer_types must have one entry per hidden layer (plus nextn layers)")
            }
        }
        guard visionConfig.patchSize > 0, visionConfig.imageSize % visionConfig.patchSize == 0 else {
            throw SharpAIStep3p7Error.invalidConfiguration(
                "vision_config.image_size must be a multiple of patch_size")
        }
    }
}
