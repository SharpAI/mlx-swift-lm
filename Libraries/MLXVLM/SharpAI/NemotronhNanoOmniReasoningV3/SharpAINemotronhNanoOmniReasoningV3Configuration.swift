// SharpAI fork overlay: NemotronH_Nano_Omni_Reasoning_V3 configuration. Not part of upstream mlx-swift-lm.
//
// The text backbone is nested under `llm_config`. The top-level keys used here are checked against
// Verified against: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/raw/main/config.json
// (vit_hidden_size 1280, projector_hidden_size 20480, downsample_ratio 0.5, ps_version "v2",
// vision_config.patch_size 16, vision_config.args.{model, cpe_max_size, cls_token_per_teacher,
// register_multiple, teachers}, sound_config.{hidden_size, projection_hidden_size, projection_bias}).

import Foundation
import MLXLLM
import MLXLMCommon

public struct SharpAINemotronhNanoOmniReasoningV3Configuration: Codable, Sendable {
    public var modelType: String
    public var architectures: [String]
    public var llmConfig: NemotronHConfiguration
    public var visionConfig: JSONValue?
    public var soundConfig: JSONValue?
    /// `vit_hidden_size`: RADIO embedding width. The projector reads (1 / downsample_ratio)^2 of it.
    public var vitHiddenSize: Int
    /// `projector_hidden_size`: hidden width of mlp1.
    public var projectorHiddenSize: Int
    /// `downsample_ratio`. Only 0.5 is ported, because the pixel shuffle is fixed to it.
    public var downsampleRatio: Float
    /// `ps_version`. Only "v2" is ported.
    public var psVersion: String
    /// RADIO tower shape decoded from `vision_config`. Nil when `vision_config` is absent.
    public var visionTower: SharpAINemotronhNanoVlV2VisionConfiguration?
    /// Projection shape decoded from `sound_config`. Nil when `sound_config` is null.
    public var soundProjection: SharpAINemotronhNanoOmniReasoningV3SoundProjectionShape?

    /// Arch strings seen for this model_type. Dispatch uses `model_type`, so both are accepted.
    public static let knownArchitectures = [
        "NemotronH_Nano_Omni_Reasoning_V3",
        "NemotronH_Super_Omni_Reasoning_V3",
    ]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case llmConfig = "llm_config"
        case visionConfig = "vision_config"
        case soundConfig = "sound_config"
        case vitHiddenSize = "vit_hidden_size"
        case projectorHiddenSize = "projector_hidden_size"
        case downsampleRatio = "downsample_ratio"
        case psVersion = "ps_version"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decodedModelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "NemotronH_Nano_Omni_Reasoning_V3"
        let decodedArchitectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        let decodedLLM = try container.decode(NemotronHConfiguration.self, forKey: .llmConfig)
        let decodedVisionJSON = try container.decodeIfPresent(JSONValue.self, forKey: .visionConfig)
        let decodedSoundJSON = try container.decodeIfPresent(JSONValue.self, forKey: .soundConfig)
        let vit = try container.decodeIfPresent(Int.self, forKey: .vitHiddenSize) ?? 1280
        let projector =
            try container.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 20480
        let ratio = try container.decodeIfPresent(Float.self, forKey: .downsampleRatio) ?? 0.5
        let psVersion = try container.decodeIfPresent(String.self, forKey: .psVersion) ?? "v2"

        let rawVision = try container.decodeIfPresent(
            SharpAINemotronhNanoOmniReasoningV3VisionRaw.self, forKey: .visionConfig)
        let rawSound = try container.decodeIfPresent(
            SharpAINemotronhNanoOmniReasoningV3SoundRaw.self, forKey: .soundConfig)

        modelType = decodedModelType
        architectures = decodedArchitectures
        llmConfig = decodedLLM
        visionConfig = decodedVisionJSON
        soundConfig = decodedSoundJSON
        vitHiddenSize = vit
        projectorHiddenSize = projector
        downsampleRatio = ratio
        self.psVersion = psVersion
        visionTower = try rawVision.map { try $0.towerConfiguration(embedDim: vit) }
        soundProjection = rawSound.map { $0.shape }
    }
}

extension SharpAINemotronhNanoOmniReasoningV3Configuration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard llmConfig.vocabSize > 0 else {
            throw SharpAINemotronhNanoOmniReasoningV3Error.invalidConfiguration(
                "llm_config.vocab_size must be positive")
        }
        guard !llmConfig.hybridOverridePattern.isEmpty else {
            throw SharpAINemotronhNanoOmniReasoningV3Error.invalidConfiguration(
                "llm_config needs a non-empty layer pattern")
        }
        guard vitHiddenSize > 0, projectorHiddenSize > 0 else {
            throw SharpAINemotronhNanoOmniReasoningV3Error.invalidConfiguration(
                "vit_hidden_size and projector_hidden_size must be positive")
        }
        guard downsampleRatio == 0.5 else {
            throw SharpAINemotronhNanoOmniReasoningV3Error.invalidConfiguration(
                "downsample_ratio must be 0.5; other ratios are not ported")
        }
        guard psVersion == "v2" else {
            throw SharpAINemotronhNanoOmniReasoningV3Error.invalidConfiguration(
                "ps_version must be v2; other pixel-shuffle versions are not ported")
        }
    }
}

/// Shape of the sound projection: RMSNorm, Linear, squared ReLU, Linear.
public struct SharpAINemotronhNanoOmniReasoningV3SoundProjectionShape: Sendable, Equatable {
    public let inputSize: Int
    public let hiddenSize: Int
    public let bias: Bool

    public init(inputSize: Int, hiddenSize: Int, bias: Bool) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.bias = bias
    }
}

/// `vision_config` as it appears in the checkpoint. Only the keys the tower needs are read.
struct SharpAINemotronhNanoOmniReasoningV3VisionRaw: Decodable {
    let patchSize: Int?
    let args: Args?

    struct Args: Decodable {
        let model: String?
        let cpeMaxSize: Int?
        let clsTokenPerTeacher: Bool?
        let registerMultiple: Int?
        let teachers: [JSONValue]?

        enum CodingKeys: String, CodingKey {
            case model
            case cpeMaxSize = "cpe_max_size"
            case clsTokenPerTeacher = "cls_token_per_teacher"
            case registerMultiple = "register_multiple"
            case teachers
        }
    }

    enum CodingKeys: String, CodingKey {
        case patchSize = "patch_size"
        case args
    }

    /// Only `vit_huge_patch16_224` is ported: depth 32, 16 heads, MLP ratio 4, LayerNorm eps 1e-6.
    func towerConfiguration(embedDim: Int) throws -> SharpAINemotronhNanoVlV2VisionConfiguration {
        let model = args?.model ?? "vit_huge_patch16_224"
        guard model == "vit_huge_patch16_224" else {
            throw SharpAINemotronhNanoOmniReasoningV3Error.unsupportedVisionModel(model)
        }
        // One cls token per distillation teacher when cls_token_per_teacher is set, else one.
        let teacherCount = args?.teachers?.count ?? 0
        let numClsTokens =
            (args?.clsTokenPerTeacher ?? false) ? max(teacherCount, 1) : 1
        return SharpAINemotronhNanoVlV2VisionConfiguration(
            patchSize: patchSize ?? 16,
            embedDim: embedDim,
            depth: 32,
            numHeads: 16,
            mlpHiddenSize: 4 * embedDim,
            cpeMaxSize: args?.cpeMaxSize ?? 2048,
            numClsTokens: numClsTokens,
            registerMultiple: args?.registerMultiple,
            layerNormEps: 1e-6)
    }
}

/// `sound_config` as it appears in the checkpoint. The encoder itself is not ported yet.
struct SharpAINemotronhNanoOmniReasoningV3SoundRaw: Decodable {
    let hiddenSize: Int?
    let projectionHiddenSize: Int?
    let projectionBias: Bool?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case projectionHiddenSize = "projection_hidden_size"
        case projectionBias = "projection_bias"
    }

    var shape: SharpAINemotronhNanoOmniReasoningV3SoundProjectionShape {
        SharpAINemotronhNanoOmniReasoningV3SoundProjectionShape(
            inputSize: hiddenSize ?? 1024,
            hiddenSize: projectionHiddenSize ?? 4096,
            bias: projectionBias ?? false)
    }
}

public enum SharpAINemotronhNanoOmniReasoningV3Error: LocalizedError, Equatable {
    case invalidConfiguration(String)
    case unsupportedVisionModel(String)
    case visionTowerUnavailable
    case oddPatchGrid(Int, Int)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "Invalid NemotronH Nano Omni configuration: \(detail)"
        case .unsupportedVisionModel(let model):
            return "Unsupported NemotronH Nano Omni vision model: \(model)"
        case .visionTowerUnavailable:
            return "This NemotronH Nano Omni model was built without a vision tower"
        case .oddPatchGrid(let height, let width):
            return "Pixel shuffle needs an even patch grid, got \(height)x\(width)"
        }
    }
}
