// SharpAI fork overlay: cosmos3_omni configuration. Not part of upstream mlx-swift-lm.
//
// The understanding path (text and vision) uses the existing Qwen3VLConfiguration.
// The generation tower (MoT second stream, diffusion, action, sound heads) is decoded by
// `SharpAICosmos3OmniGenerationConfiguration`; the VAE and sound tokenizer blocks are not decoded.
// Verified against: https://huggingface.co/hf-tiny-v2/tiny-random-Cosmos3OmniForConditionalGeneration/raw/main/config.json
//   (architectures `Cosmos3OmniForConditionalGeneration`, model_type `cosmos3_omni`).
// The name `Cosmos3ForConditionalGeneration` in the decoding fixtures is not a real architecture name.
// Routing uses model_type only, so the fixture name is never checked.

import Foundation
import MLXLMCommon

public enum SharpAICosmos3OmniError: LocalizedError {
    case forwardNotImplemented(String)
    case duplicateWeightKey(String)
    case unrecognizedWeightKey(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass is not implemented yet "
                + "(Cosmos3 generation experts and weight layout are unverified)."
        case .duplicateWeightKey(let key):
            return "Duplicate weight key after cosmos3_omni sanitize: \(key)"
        case .unrecognizedWeightKey(let key):
            return "cosmos3_omni checkpoint key is not part of the known layout: \(key)"
        }
    }
}

public struct SharpAICosmos3OmniConfiguration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public let modelType: String
    public let architectures: [String]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "cosmos3_omni"
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
    }

    /// Decodes the generation tower configuration.
    ///
    /// `rootConfig` is the root `config.json` and `transformerConfig` is `transformer/config.json`.
    /// Either may be nil, not both. See `SharpAICosmos3OmniGenerationConfiguration.decode`.
    public static func generationConfiguration(
        rootConfig: Data?, transformerConfig: Data? = nil
    ) throws -> SharpAICosmos3OmniGenerationConfiguration {
        try SharpAICosmos3OmniGenerationConfiguration.decode(
            rootConfig: rootConfig, transformerConfig: transformerConfig)
    }

    public func validateModelConfiguration() throws {
        guard modelType == "cosmos3_omni" else {
            throw ModelFactoryError.invalidConfiguration(
                "cosmos3_omni: unexpected model_type \(modelType)")
        }
    }

    /// Decodes the understanding path as a Qwen3-VL configuration.
    ///
    /// The top-level `text_config` and `vision_config` blocks are decoded with the Qwen3-VL layout.
    /// `text_config.model_type` is not checked. Verified against
    /// https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/cosmos3_omni/configuration_cosmos3_omni.py
    /// (default `text_config` model_type `qwen3_vl_text`, `tie_word_embeddings` default False).
    public static func qwen3VLConfiguration(from data: Data) throws -> Qwen3VLConfiguration {
        try JSONDecoder.json5().decode(
            Qwen3VLConfiguration.self, from: try normalizedUnderstandingJSON(data))
    }

    /// Copies the Cosmos3 RoPE and tie settings into the keys that `Qwen3VLConfiguration` reads.
    ///
    /// The tiny reference config (URL in the file header) writes RoPE as
    /// `text_config.rope_parameters` (`rope_theta`, `mrope_interleaved`, `mrope_section`) and sets
    /// `tie_word_embeddings` at the top level. Existing `rope_theta`, `rope_scaling` and
    /// `tie_word_embeddings` keys in `text_config` are kept. JSON5 input (comments, trailing commas)
    /// is parsed with the same rules as the JSON5 decoder, so it is normalised too. Input that
    /// cannot be parsed at all is returned unchanged.
    static func normalizedUnderstandingJSON(_ data: Data) throws -> Data {
        guard var root = try? JSONSerialization.jsonObject(with: data, options: [.json5Allowed])
                as? [String: Any],
            var text = root["text_config"] as? [String: Any]
        else {
            return data
        }

        if let parameters = text["rope_parameters"] as? [String: Any] {
            if text["rope_scaling"] == nil {
                text["rope_scaling"] = parameters
            }
            if text["rope_theta"] == nil, let theta = parameters["rope_theta"] {
                text["rope_theta"] = theta
            }
        }
        if text["tie_word_embeddings"] == nil {
            if let tie = root["tie_word_embeddings"] {
                text["tie_word_embeddings"] = tie
            } else {
                text["tie_word_embeddings"] = false
            }
        }

        root["text_config"] = text
        return try JSONSerialization.data(withJSONObject: root)
    }
}

/// Generation tower settings of `cosmos3_omni` (second MoT stream plus modality heads).
///
/// Nesting in nvidia/Cosmos3-Nano (Verified against
/// https://huggingface.co/nvidia/Cosmos3-Nano/raw/main/transformer/config.json and
/// https://huggingface.co/nvidia/Cosmos3-Nano/raw/main/config.json): `transformer/config.json` is
/// flat and holds every field below. The root `config.json` carries the transformer dimensions in
/// `text_config` and a subset of the generation fields in `model.config` (`max_action_dim`,
/// `num_embodiment_domains`, `sound_dim`, `joint_attn_implementation`, ...) and in
/// `model.config.diffusion_expert_config` (`base_fps`, `enable_fps_modulation`,
/// `unified_3d_mrope_*`). `decode` merges these, lowest to highest precedence: `model.config`,
/// `model.config.diffusion_expert_config`, root `text_config`, `transformer/config.json`.
///
/// Defaults are the keyword defaults of `Cosmos3OmniTransformer.__init__`. Verified against
/// https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_cosmos3.py
/// (lines 395-430). Fields that signature does not have (`use_moe`, `joint_attn_implementation`,
/// `qk_norm_for_diffusion`) default to the Nano values and are marked UNVERIFIED below.
public struct SharpAICosmos3OmniGenerationConfiguration: Decodable, Sendable, Equatable {
    // Transformer dimensions (required, no default).
    public let hiddenSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let vocabSize: Int
    public let rmsNormEps: Double

    // Mixture-of-Transformers switches.
    // UNVERIFIED: the Diffusers class ignores `use_moe`; the defaults below are the Nano values.
    public let useMoe: Bool
    public let jointAttnImplementation: String
    public let qkNormForDiffusion: Bool
    public let qkNormForText: Bool

    // Vision latents.
    public let latentChannel: Int
    public let latentPatchSize: Int
    public let patchLatentDim: Int
    public let timestepScale: Double

    // Action and sound heads.
    public let actionGen: Bool
    public let soundGen: Bool
    public let maxActionDim: Int?
    public let numEmbodimentDomains: Int
    public let soundDim: Int?
    public let soundLatentFps: Double

    // Position embedding (unified 3D mRoPE).
    public let ropeTheta: Double
    public let mropeSection: [Int]
    public let mropeInterleaved: Bool
    public let unified3DMropeTemporalModalityMargin: Int
    public let unified3DMropeResetSpatialIds: Bool
    public let enableFpsModulation: Bool
    public let baseFps: Double

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case useMoe = "use_moe"
        case jointAttnImplementation = "joint_attn_implementation"
        case qkNormForDiffusion = "qk_norm_for_diffusion"
        case qkNormForText = "qk_norm_for_text"
        case latentChannel = "latent_channel"
        case latentPatchSize = "latent_patch_size"
        case patchLatentDim = "patch_latent_dim"
        case timestepScale = "timestep_scale"
        case actionGen = "action_gen"
        case soundGen = "sound_gen"
        case maxActionDim = "max_action_dim"
        case actionDim = "action_dim"
        case numEmbodimentDomains = "num_embodiment_domains"
        case soundDim = "sound_dim"
        case soundLatentFps = "sound_latent_fps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case unified3DMropeTemporalModalityMargin = "unified_3d_mrope_temporal_modality_margin"
        case unified3DMropeResetSpatialIds = "unified_3d_mrope_reset_spatial_ids"
        case enableFpsModulation = "enable_fps_modulation"
        case baseFps = "base_fps"
    }

    struct RopeScaling: Decodable {
        let mropeSection: [Int]?
        let mropeInterleaved: Bool?

        enum CodingKeys: String, CodingKey {
            case mropeSection = "mrope_section"
            case mropeInterleaved = "mrope_interleaved"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        rmsNormEps = try c.decodeIfPresent(Double.self, forKey: .rmsNormEps) ?? 1e-6

        useMoe = try c.decodeIfPresent(Bool.self, forKey: .useMoe) ?? true
        jointAttnImplementation =
            try c.decodeIfPresent(String.self, forKey: .jointAttnImplementation) ?? "two_way"
        qkNormForDiffusion = try c.decodeIfPresent(Bool.self, forKey: .qkNormForDiffusion) ?? true
        qkNormForText = try c.decodeIfPresent(Bool.self, forKey: .qkNormForText) ?? true

        let channel = try c.decodeIfPresent(Int.self, forKey: .latentChannel) ?? 48
        let patch = try c.decodeIfPresent(Int.self, forKey: .latentPatchSize) ?? 2
        latentChannel = channel
        latentPatchSize = patch
        // 48 x 2 x 2 = 192 for Nano (research: patch_latent_dim = latent_channel x patch x patch).
        patchLatentDim =
            try c.decodeIfPresent(Int.self, forKey: .patchLatentDim) ?? channel * patch * patch
        timestepScale = try c.decodeIfPresent(Double.self, forKey: .timestepScale) ?? 0.001

        actionGen = try c.decodeIfPresent(Bool.self, forKey: .actionGen) ?? false
        soundGen = try c.decodeIfPresent(Bool.self, forKey: .soundGen) ?? false
        // `transformer/config.json` writes `action_dim` and `max_action_dim` with the same value (64).
        // UNVERIFIED: `max_action_dim` is treated as the padded width of `action_dim`.
        maxActionDim =
            try c.decodeIfPresent(Int.self, forKey: .maxActionDim)
            ?? c.decodeIfPresent(Int.self, forKey: .actionDim)
        numEmbodimentDomains = try c.decodeIfPresent(Int.self, forKey: .numEmbodimentDomains) ?? 32
        soundDim = try c.decodeIfPresent(Int.self, forKey: .soundDim)
        soundLatentFps = try c.decodeIfPresent(Double.self, forKey: .soundLatentFps) ?? 25

        ropeTheta = try c.decodeIfPresent(Double.self, forKey: .ropeTheta) ?? 5_000_000
        let scaling = try c.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling)
        mropeSection = scaling?.mropeSection ?? [24, 20, 20]
        mropeInterleaved = scaling?.mropeInterleaved ?? true
        unified3DMropeTemporalModalityMargin =
            try c.decodeIfPresent(Int.self, forKey: .unified3DMropeTemporalModalityMargin) ?? 15000
        unified3DMropeResetSpatialIds =
            try c.decodeIfPresent(Bool.self, forKey: .unified3DMropeResetSpatialIds) ?? true
        enableFpsModulation = try c.decodeIfPresent(Bool.self, forKey: .enableFpsModulation) ?? true
        baseFps = try c.decodeIfPresent(Double.self, forKey: .baseFps) ?? 24

        try validate()
    }

    /// Checks cross-field consistency. Throws `ModelFactoryError.invalidConfiguration`.
    public func validate() throws {
        func fail(_ message: String) -> ModelFactoryError {
            .invalidConfiguration("cosmos3_omni generation: \(message)")
        }
        for (name, value) in [
            ("hidden_size", hiddenSize), ("num_attention_heads", numAttentionHeads),
            ("num_key_value_heads", numKeyValueHeads), ("head_dim", headDim),
            ("intermediate_size", intermediateSize), ("num_hidden_layers", numHiddenLayers),
            ("vocab_size", vocabSize), ("latent_channel", latentChannel),
            ("latent_patch_size", latentPatchSize), ("num_embodiment_domains", numEmbodimentDomains),
        ] where value <= 0 {
            throw fail("\(name) must be positive, got \(value)")
        }
        guard numAttentionHeads % numKeyValueHeads == 0 else {
            throw fail(
                "num_attention_heads \(numAttentionHeads) is not a multiple of "
                    + "num_key_value_heads \(numKeyValueHeads)")
        }
        guard patchLatentDim == latentChannel * latentPatchSize * latentPatchSize else {
            throw fail(
                "patch_latent_dim \(patchLatentDim) != latent_channel x latent_patch_size^2 "
                    + "(\(latentChannel * latentPatchSize * latentPatchSize))")
        }
        guard mropeSection.count == 3, mropeSection.allSatisfy({ $0 > 0 }),
            mropeSection.reduce(0, +) * 2 == headDim
        else {
            throw fail("mrope_section \(mropeSection) must be 3 positive values summing to head_dim / 2")
        }
        guard jointAttnImplementation == "two_way" else {
            // UNVERIFIED: the research found a "three_way" packing layout in the original code
            // (attention.py:868-884) but no checkpoint that uses it; only two_way is implemented.
            throw fail("unsupported joint_attn_implementation \(jointAttnImplementation)")
        }
        if actionGen, (maxActionDim ?? 0) <= 0 {
            throw fail("action_gen requires max_action_dim (or action_dim)")
        }
        if soundGen, (soundDim ?? 0) <= 0 {
            throw fail("sound_gen requires sound_dim")
        }
    }

    /// Merges the root and transformer configs (see the type comment) and decodes the result.
    ///
    /// Input is parsed as JSON5. Throws `ModelFactoryError.invalidConfiguration` when both inputs
    /// are nil or not a JSON object, and the decoding error when a required dimension is missing.
    public static func decode(rootConfig: Data?, transformerConfig: Data?) throws -> Self {
        let merged = try mergedObject(rootConfig: rootConfig, transformerConfig: transformerConfig)
        return try JSONDecoder().decode(
            Self.self, from: try JSONSerialization.data(withJSONObject: merged))
    }

    /// The merged JSON object described in the type comment, before any field is decoded.
    static func mergedObject(rootConfig: Data?, transformerConfig: Data?) throws -> [String: Any] {
        func object(_ data: Data?) throws -> [String: Any]? {
            guard let data else { return nil }
            guard
                let parsed = try? JSONSerialization.jsonObject(
                    with: data, options: [.json5Allowed]) as? [String: Any]
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "cosmos3_omni generation: config is not a JSON object")
            }
            return parsed
        }
        let root = try object(rootConfig)
        let transformer = try object(transformerConfig)
        guard root != nil || transformer != nil else {
            throw ModelFactoryError.invalidConfiguration(
                "cosmos3_omni generation: no configuration given")
        }

        var merged = [String: Any]()
        if let model = (root?["model"] as? [String: Any])?["config"] as? [String: Any] {
            merged.merge(model) { _, new in new }
            if let diffusion = model["diffusion_expert_config"] as? [String: Any] {
                merged.merge(diffusion) { _, new in new }
            }
        }
        if var text = root?["text_config"] as? [String: Any] {
            // The tiny reference config writes RoPE as `rope_parameters` (see
            // `normalizedUnderstandingJSON`).
            if let parameters = text["rope_parameters"] as? [String: Any] {
                if text["rope_scaling"] == nil { text["rope_scaling"] = parameters }
                if text["rope_theta"] == nil, let theta = parameters["rope_theta"] {
                    text["rope_theta"] = theta
                }
            }
            merged.merge(text) { _, new in new }
        }
        if let transformer {
            merged.merge(transformer) { _, new in new }
        }

        return merged
    }
}
