// SharpAI fork overlay: muse_glimmer_assistant configuration. Not part of upstream mlx-swift-lm.
//
// Keys follow scripts/conformance/offline/fixtures/muse_glimmer_assistant/
// meta-models__Muse-Glimmer-30B-assistant.json. The top-level token ids and target_layer_ids are
// decoded here. UNVERIFIED: defaults for keys the fixture omits (rms_norm_eps,
// max_position_embeddings, attention_bias, rope theta fallback). The fixture has no vocab_size,
// so mask_token_id cannot be bounds-checked here.

import Foundation
import MLXLMCommon

public enum SharpAIMuseGlimmerAssistantError: LocalizedError {
    case drafterNotImplemented(String)
    case incompatibleTarget(String)

    public var errorDescription: String? {
        switch self {
        case .drafterNotImplemented(let type):
            return
                "Model type \(type) config is valid, but the drafter forward pass is not "
                + "implemented yet. Its weight names are unverified."
        case .incompatibleTarget(let reason):
            return "muse_glimmer_assistant does not fit this target: \(reason)"
        }
    }
}

public struct SharpAIMuseGlimmerAssistantConfiguration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public static let expectedModelType = "muse_glimmer_assistant"

    public let modelType: String
    public let architectures: [String]
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int
    public let attentionBias: Bool
    /// nil when the config has no sliding window. Required by validation when any layer slides.
    public let slidingWindow: Int?
    /// Per-layer attention kind: `full_attention` or `sliding_attention`.
    public let layerTypes: [String]
    /// From `rope_parameters.rope_theta`, else top-level `rope_theta`.
    public let ropeTheta: Float
    public let ropeParameters: [String: StringOrNumber]?
    public let blockSize: Int?
    public let bosTokenId: Int?
    public let eosTokenId: Int?
    public let padTokenId: Int?
    public let maskTokenId: Int?
    /// Target hidden-layer indices the drafter reads. Top-level `target_layer_ids`.
    public let targetLayerIds: [Int]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case blockSize = "block_size"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case maskTokenId = "mask_token_id"
        case targetLayerIds = "target_layer_ids"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType)
            ?? Self.expectedModelType
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (attentionHeads > 0 ? hiddenSize / attentionHeads : 0)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        let topLevelTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta =
            ropeParameters?["rope_theta"]?.asFloat() ?? topLevelTheta ?? 500_000.0
        blockSize = try c.decodeIfPresent(Int.self, forKey: .blockSize)
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId)
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId)
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId)
        maskTokenId = try c.decodeIfPresent(Int.self, forKey: .maskTokenId)
        targetLayerIds = try c.decodeIfPresent([Int].self, forKey: .targetLayerIds) ?? []
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "muse_glimmer_assistant: model_type must be \(Self.expectedModelType), got \(modelType)")
        }

        guard hiddenSize > 0, hiddenLayers > 0, attentionHeads > 0, kvHeads > 0, headDim > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "muse_glimmer_assistant: hidden_size, num_hidden_layers, attention heads and "
                + "head_dim must be positive")
        }

        guard attentionHeads % kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "muse_glimmer_assistant: num_attention_heads must be a multiple of "
                + "num_key_value_heads")
        }

        guard !targetLayerIds.isEmpty, targetLayerIds.allSatisfy({ $0 >= 0 }),
            Set(targetLayerIds).count == targetLayerIds.count
        else {
            throw ModelFactoryError.invalidConfiguration(
                "muse_glimmer_assistant: target_layer_ids must be non-empty, non-negative and unique")
        }

        guard layerTypes.count == hiddenLayers else {
            throw ModelFactoryError.invalidConfiguration(
                "muse_glimmer_assistant: layer_types needs one entry per layer "
                + "(\(hiddenLayers)), got \(layerTypes.count)")
        }

        let knownTypes: Set<String> = ["full_attention", "sliding_attention"]
        guard layerTypes.allSatisfy({ knownTypes.contains($0) }) else {
            throw ModelFactoryError.invalidConfiguration(
                "muse_glimmer_assistant: layer_types must be full_attention or sliding_attention")
        }

        if layerTypes.contains("sliding_attention") {
            guard let window = slidingWindow, window > 0 else {
                throw ModelFactoryError.invalidConfiguration(
                    "muse_glimmer_assistant: sliding_attention layers need a positive sliding_window")
            }
        }

        if let blockSize, blockSize < 1 {
            throw ModelFactoryError.invalidConfiguration(
                "muse_glimmer_assistant: block_size must be positive, got \(blockSize)")
        }
    }
}

// MARK: - Target conformance

extension SharpAIMuseGlimmerAssistantConfiguration {

    /// Checks the drafter against a target described by its model type and text dimensions.
    /// UNVERIFIED: requiring equal hidden sizes is an inference from the shared embedding and
    /// the fixture's hidden_size (6656) matching the target's.
    public func validateTarget(
        modelType targetModelType: String, hiddenSize targetHiddenSize: Int,
        hiddenLayers targetHiddenLayers: Int
    ) throws {
        guard targetModelType == "muse_glimmer" else {
            throw SharpAIMuseGlimmerAssistantError.incompatibleTarget(
                "target model_type must be muse_glimmer, got \(targetModelType)")
        }
        guard hiddenSize == targetHiddenSize else {
            throw SharpAIMuseGlimmerAssistantError.incompatibleTarget(
                "drafter hidden_size \(hiddenSize) must equal target hidden_size \(targetHiddenSize)")
        }
        guard let maxTargetLayer = targetLayerIds.max(), maxTargetLayer < targetHiddenLayers else {
            throw SharpAIMuseGlimmerAssistantError.incompatibleTarget(
                "target_layer_ids must be below the target's \(targetHiddenLayers) layers")
        }
    }

    /// Checks the drafter against a loaded Muse-Glimmer VLM config.
    public func validateTarget(_ target: MuseGlimmerConfiguration) throws {
        try validateTarget(
            modelType: target.modelType,
            hiddenSize: target.textConfiguration.hiddenSize,
            hiddenLayers: target.textConfiguration.hiddenLayers)
    }

    /// Non-throwing form of ``validateTarget(_:)-(MuseGlimmerConfiguration)``.
    public func isCompatible(with target: MuseGlimmerConfiguration) -> Bool {
        (try? validateTarget(target)) != nil
    }
}
