// SharpAI fork overlay: bagel configuration. Not part of upstream mlx-swift-lm.
//
// Understanding path only (image and text in, text out). The generation path (VAE, flow
// matching, `_moe_gen` experts, timestep and latent embedders) is not modeled.
// UNVERIFIED: key names and defaults follow the BAGEL-7B-MoT config as recalled. Check them
// against the checkpoint's config.json before relying on them.

import Foundation
import MLXLMCommon

public struct SharpAIBagelConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    /// Architecture name checked when `architectures` is present.
    public static let expectedArchitecture = "BagelForConditionalGeneration"

    /// Qwen2-style text decoder (`llm_config`).
    public struct TextConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let rmsNormEps: Float
        public let vocabularySize: Int
        public let ropeTheta: Float
        public let ropeTraditional: Bool
        public let tieWordEmbeddings: Bool
        public let qkNorm: Bool

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case ropeTheta = "rope_theta"
            case ropeTraditional = "rope_traditional"
            case tieWordEmbeddings = "tie_word_embeddings"
            case qkNorm = "qk_norm"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let heads = try c.decode(Int.self, forKey: .attentionHeads)
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
            intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
            attentionHeads = heads
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
            ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            // Defaults to false so a config without the key fails validation.
            qkNorm = try c.decodeIfPresent(Bool.self, forKey: .qkNorm) ?? false
        }
    }

    /// SigLIP vision tower (`vision_config`).
    public struct VisionConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let channels: Int
        public let imageSize: Int
        public let patchSize: Int
        public let layerNormEps: Float

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case channels = "num_channels"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case layerNormEps = "layer_norm_eps"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
            hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
            attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            imageSize = try c.decode(Int.self, forKey: .imageSize)
            patchSize = try c.decode(Int.self, forKey: .patchSize)
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        }
    }

    public let architectures: [String]?
    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let connectorAct: String
    /// Side length, in patches, of the position-embedding grid.
    /// UNVERIFIED: assumed to match the SigLIP position table, not the checkpoint.
    public let vitMaxNumPatchPerSide: Int
    /// Placeholder token id that marks where image features go in the prompt.
    /// UNVERIFIED: default is the Qwen2-VL image pad id, not a Bagel value.
    public let imageTokenIndex: Int

    enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case textConfiguration = "llm_config"
        case visionConfiguration = "vision_config"
        case connectorAct = "connector_act"
        case vitMaxNumPatchPerSide = "vit_max_num_patch_per_side"
        case imageTokenIndex = "image_token_index"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "bagel"
        textConfiguration = try c.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        connectorAct =
            try c.decodeIfPresent(String.self, forKey: .connectorAct) ?? "gelu_pytorch_tanh"
        vitMaxNumPatchPerSide =
            try c.decodeIfPresent(Int.self, forKey: .vitMaxNumPatchPerSide) ?? 70
        imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 151655
    }

    public func validateModelConfiguration() throws {
        if let architectures, !architectures.contains(Self.expectedArchitecture) {
            throw ModelFactoryError.invalidConfiguration(
                "bagel: architectures must include \(Self.expectedArchitecture)")
        }

        let text = textConfiguration
        guard text.qkNorm else {
            throw ModelFactoryError.invalidConfiguration(
                "bagel: llm_config.qk_norm must be true for this model")
        }
        guard text.attentionHeads > 0, text.hiddenSize % text.attentionHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "bagel: hidden_size must be divisible by num_attention_heads")
        }
        guard text.kvHeads > 0, text.attentionHeads % text.kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "bagel: num_attention_heads must be divisible by num_key_value_heads")
        }

        let vision = visionConfiguration
        guard vision.attentionHeads > 0, vision.hiddenSize % vision.attentionHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "bagel: vision hidden_size must be divisible by num_attention_heads")
        }
        guard vision.patchSize > 0, vision.imageSize % vision.patchSize == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "bagel: vision image_size must be a multiple of patch_size")
        }
        guard vitMaxNumPatchPerSide > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "bagel: vit_max_num_patch_per_side must be positive")
        }
        guard connectorAct == "gelu_pytorch_tanh" else {
            throw ModelFactoryError.invalidConfiguration(
                "bagel: connector_act \(connectorAct) is not supported")
        }
    }
}
