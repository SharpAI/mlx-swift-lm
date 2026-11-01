// SharpAI fork overlay: unlimited_ocr configuration. Not part of upstream mlx-swift-lm.
//
// The text stack is a DeepSeek-V2 block with standard multi-head attention (use_mla false,
// kv_lora_rank null). Upstream DeepseekV2Configuration maps a null kv_lora_rank to 512, so this
// file decodes its own fields.
// Verified against: https://huggingface.co/baidu/Unlimited-OCR/raw/main/config.json
// Defaults verified against: https://huggingface.co/baidu/Unlimited-OCR/raw/main/configuration_deepseek_v2.py

import Foundation
import MLXLMCommon

public enum SharpAIUnlimitedOcrError: LocalizedError {
    case forwardNotImplemented(String)
    case unsupportedInputSize(String)
    case visionNotConfigured
    case imageTokenCountMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return "Model type \(type) has no forward pass."
        case .unsupportedInputSize(let detail):
            return
                "unlimited_ocr: unsupported image grid (\(detail)). Only the 1024 px global view is "
                + "ported; the antialiased bicubic position interpolation used by the 640 px crop "
                + "path is not implemented."
        case .visionNotConfigured:
            return "unlimited_ocr: image input needs a checkpoint with vision_config."
        case .imageTokenCountMismatch(let expected, let actual):
            return
                "unlimited_ocr: prompt has \(actual) image placeholder tokens but the image "
                + "produces \(expected) features."
        }
    }
}

/// Text fields read from `language_config`. Missing sizes decode as 0 so validation rejects them.
/// Missing router and RoPE fields take the DeepSeek-V2 defaults from configuration_deepseek_v2.py.
public struct SharpAIUnlimitedOcrTextConfiguration: Decodable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let kvLoraRank: Int?
    public let useMLA: Bool
    public let nRoutedExperts: Int
    public let nSharedExperts: Int
    public let numExpertsPerTok: Int
    public let firstKDenseReplace: Int
    public let moeLayerFreq: Int
    public let topkMethod: String
    public let normTopKProb: Bool
    public let routedScalingFactor: Float
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float
    public let slidingWindow: Int?

    /// True when the text stack asks for DeepSeek-V2 MLA attention.
    public var usesMLA: Bool {
        useMLA || (kvLoraRank ?? 0) > 0
    }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case kvLoraRank = "kv_lora_rank"
        case useMLA = "use_mla"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case firstKDenseReplace = "first_k_dense_replace"
        case moeLayerFreq = "moe_layer_freq"
        case topkMethod = "topk_method"
        case normTopKProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 0
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 0
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 0
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 0
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 0
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 0
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank)
        useMLA = try c.decodeIfPresent(Bool.self, forKey: .useMLA) ?? false
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 0
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 0
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        topkMethod = try c.decodeIfPresent(String.self, forKey: .topkMethod) ?? "greedy"
        normTopKProb = try c.decodeIfPresent(Bool.self, forKey: .normTopKProb) ?? false
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 0
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
    }
}

public struct SharpAIUnlimitedOcrConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let text: SharpAIUnlimitedOcrTextConfiguration
    public let hasVisionConfig: Bool
    public let hasProjectorConfig: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case languageConfig = "language_config"
        case visionConfig = "vision_config"
        case projectorConfig = "projector_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "unlimited_ocr"

        if container.contains(.languageConfig) {
            text = try container.decode(
                SharpAIUnlimitedOcrTextConfiguration.self, forKey: .languageConfig)
        } else {
            text = try SharpAIUnlimitedOcrTextConfiguration(from: decoder)
        }

        hasVisionConfig = container.contains(.visionConfig)
        hasProjectorConfig = container.contains(.projectorConfig)
    }

    public func validateModelConfiguration() throws {
        let dims = [
            text.hiddenSize, text.numHiddenLayers, text.numAttentionHeads,
            text.numKeyValueHeads, text.vocabSize, text.intermediateSize,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "unlimited_ocr: text hidden, layer, head, vocab and intermediate sizes must be positive")
        }
        guard text.numAttentionHeads % text.numKeyValueHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "unlimited_ocr: num_attention_heads must be divisible by num_key_value_heads")
        }
        guard text.hiddenSize % text.numAttentionHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "unlimited_ocr: hidden_size must be divisible by num_attention_heads")
        }
        guard !text.usesMLA else {
            throw ModelFactoryError.invalidConfiguration(
                "unlimited_ocr: MLA attention (use_mla or kv_lora_rank > 0) is not implemented by this overlay")
        }
        guard text.topkMethod == "greedy" else {
            throw ModelFactoryError.invalidConfiguration(
                "unlimited_ocr: topk_method \(text.topkMethod) is not implemented; only greedy is")
        }
        guard text.moeLayerFreq > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "unlimited_ocr: moe_layer_freq must be positive")
        }
        if text.nRoutedExperts > 0 {
            guard text.moeIntermediateSize > 0, text.numExpertsPerTok > 0,
                text.numExpertsPerTok <= text.nRoutedExperts
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "unlimited_ocr: MoE needs moe_intermediate_size > 0 and 0 < num_experts_per_tok <= n_routed_experts"
                )
            }
        }
    }
}
