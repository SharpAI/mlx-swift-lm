// SharpAI fork overlay: hunyuan_vl configuration. Not part of upstream mlx-swift-lm.
//
// Decodes and validates the config. Field names, defaults and the M-RoPE section rule follow
// transformers `models/hunyuan_vl/configuration_hunyuan_vl.py` and `modular_hunyuan_vl.py`
// (fetched from main). The forward pass is in SharpAIHunyuanVlModel.swift.
// UNVERIFIED: real checkpoint values (tencent/HunyuanOCR config.json was not inspected).
// UNVERIFIED: the legacy `xdrope_section` alias is kept for old Tencent checkpoints, as upstream does.

import Foundation
import MLXLMCommon

public enum SharpAIHunyuanVlError: LocalizedError {
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but it has no LanguageModel conformance or "
                + "cache wiring yet. The text decoder and vision tower are implemented."
        }
    }
}

/// RoPE block. Carries the M-RoPE section list, stored under `mrope_section` (or the legacy
/// `xdrope_section` alias).
public struct SharpAIHunyuanVlRopeScaling: Decodable, Sendable {
    /// `type`, or `rope_type` when `type` is absent.
    public let type: String?
    public let alpha: Float?
    public let ropeTheta: Float?
    /// Half-head widths per M-RoPE axis (`mrope_section`, legacy `xdrope_section`).
    public let xdropeSection: [Int]?

    enum CodingKeys: String, CodingKey {
        case type
        case ropeType = "rope_type"
        case alpha
        case ropeTheta = "rope_theta"
        case xdropeSection = "xdrope_section"
        case mropeSection = "mrope_section"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let typeValue = try container.decodeIfPresent(String.self, forKey: .type)
        type = try typeValue ?? container.decodeIfPresent(String.self, forKey: .ropeType)
        alpha = try container.decodeIfPresent(Float.self, forKey: .alpha)
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
        let legacySection = try container.decodeIfPresent([Int].self, forKey: .xdropeSection)
        xdropeSection = try legacySection ?? container.decodeIfPresent([Int].self, forKey: .mropeSection)
    }
}

/// Text dimensions for hunyuan_vl. Defaults follow HunYuanVLTextConfig.
public struct SharpAIHunyuanVlTextDimensions: Decodable, Sendable {
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
    public let attentionBias: Bool
    /// Informational. The upstream forward always applies per-head Q/K RMSNorm after RoPE.
    public let useQkNorm: Bool
    public let tieWordEmbeddings: Bool
    public let ropeScaling: SharpAIHunyuanVlRopeScaling?

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
        case attentionBias = "attention_bias"
        case useQkNorm = "use_qk_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
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
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hidden / max(heads, 1))
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        // rope_parameters (current layout) wins over the legacy rope_scaling block.
        let scaling =
            try container.decodeIfPresent(
                SharpAIHunyuanVlRopeScaling.self, forKey: .ropeParameters)
            ?? container.decodeIfPresent(SharpAIHunyuanVlRopeScaling.self, forKey: .ropeScaling)
        ropeScaling = scaling
        let legacyTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = scaling?.ropeTheta ?? legacyTheta ?? 10_000
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        useQkNorm = try container.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }
}

/// Vision tower dimensions. Defaults follow HunYuanVLVisionConfig.
/// `text_hidden_size` is not decoded: the upstream config always sets it from the text hidden size.
public struct SharpAIHunyuanVlVisionDimensions: Decodable, Sendable {
    public let hiddenSize: Int
    public let depth: Int
    public let attentionHeads: Int
    public let intermediateSize: Int
    public let numChannels: Int
    public let patchSize: Int
    public let spatialMergeSize: Int
    public let maxImageSize: Int
    public let rmsNormEps: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case depth = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case numChannels = "num_channels"
        case patchSize = "patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case maxImageSize = "max_image_size"
        case rmsNormEps = "rms_norm_eps"
        case layerNormEps = "layer_norm_eps"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        depth = try container.decodeIfPresent(Int.self, forKey: .depth) ?? 27
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
        numChannels = try container.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        spatialMergeSize = try container.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
        maxImageSize = try container.decodeIfPresent(Int.self, forKey: .maxImageSize) ?? 2048
        // layer_norm_eps is an upstream attribute alias of rms_norm_eps.
        let rmsEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
        rmsNormEps = try rmsEps ?? container.decodeIfPresent(Float.self, forKey: .layerNormEps)
            ?? 1e-5
    }
}

public struct SharpAIHunyuanVlConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let textDimensions: SharpAIHunyuanVlTextDimensions
    public let visionDimensions: SharpAIHunyuanVlVisionDimensions?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "hunyuan_vl"

        if container.contains(.textConfig) {
            textDimensions = try container.decode(
                SharpAIHunyuanVlTextDimensions.self, forKey: .textConfig)
        } else {
            textDimensions = try SharpAIHunyuanVlTextDimensions(from: decoder)
        }

        visionDimensions = try container.decodeIfPresent(
            SharpAIHunyuanVlVisionDimensions.self, forKey: .visionConfig)
    }

    public func validateModelConfiguration() throws {
        let text = textDimensions
        let dims = [
            text.vocabSize, text.hiddenSize, text.hiddenLayers, text.intermediateSize,
            text.attentionHeads, text.kvHeads, text.headDim, text.maxPositionEmbeddings,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "hunyuan_vl: text sizes must be positive")
        }
        guard text.attentionHeads % text.kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "hunyuan_vl: text num_attention_heads must be divisible by num_key_value_heads")
        }
        guard text.headDim % 2 == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "hunyuan_vl: text head_dim must be even for RoPE")
        }
        // Upstream _validate_mrope_section: non-negative integer widths summing to head_dim / 2.
        if let section = text.ropeScaling?.xdropeSection {
            let expected = text.headDim / 2
            guard section.allSatisfy({ $0 >= 0 }), section.reduce(0, +) == expected else {
                throw ModelFactoryError.invalidConfiguration(
                    "hunyuan_vl: mrope_section must be non-negative and sum to head_dim / 2 "
                    + "(\(expected)), got \(section)")
            }
        }

        if let vision = visionDimensions {
            let visionDims = [
                vision.hiddenSize, vision.depth, vision.attentionHeads, vision.intermediateSize,
                vision.numChannels, vision.patchSize, vision.spatialMergeSize,
                vision.maxImageSize,
            ]
            guard visionDims.allSatisfy({ $0 > 0 }) else {
                throw ModelFactoryError.invalidConfiguration(
                    "hunyuan_vl: vision sizes must be positive")
            }
            guard vision.hiddenSize % vision.attentionHeads == 0 else {
                throw ModelFactoryError.invalidConfiguration(
                    "hunyuan_vl: vision hidden_size must be divisible by num_attention_heads")
            }
            guard vision.maxImageSize % vision.patchSize == 0 else {
                throw ModelFactoryError.invalidConfiguration(
                    "hunyuan_vl: vision max_image_size must be divisible by patch_size")
            }
        }
    }
}
