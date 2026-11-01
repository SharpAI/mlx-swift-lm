// SharpAI fork overlay: step5_vl configuration. Not part of upstream mlx-swift-lm.
//
// Field names follow scripts/conformance/offline/fixtures/step5_vl/stepfun-ai__Step-5-Preview.json.
// UNVERIFIED: the fixture is a config.json excerpt. No checkpoint, model source, or CSA
// sparse-attention reference exists in this repo, vLLM, or mlx-lm on this machine.
// Sparse-attention, MoE, and MTP fields are decoded but not used.

import Foundation
import MLXLMCommon

public enum SharpAIStep5VlError: LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "step5_vl unsupported: \(detail)"
        case .invalidConfiguration(let detail):
            return "step5_vl invalid configuration: \(detail)"
        }
    }
}

/// `vision_config` for `model_type: perception_encoder` (the PE tower).
public struct SharpAIStep5VlVisionConfiguration: Codable, Sendable {
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

/// `text_config.sparse_config`. The indexer and compression are not implemented.
public struct SharpAIStep5VlSparseConfiguration: Codable, Sendable {
    public let enabled: Bool?
    public let proxyDim: Int?
    public let topk: Int?
    public let regionBlockSize: Int?
    public let numProviderGroups: Int?
    public let attentionImpl: String?
    public let applyToLayerTypes: [String]?

    enum CodingKeys: String, CodingKey {
        case enabled
        case proxyDim = "proxy_dim"
        case topk
        case regionBlockSize = "region_block_size"
        case numProviderGroups = "num_provider_groups"
        case attentionImpl = "attention_impl"
        case applyToLayerTypes = "apply_to_layer_types"
    }
}

/// `text_config` for the step5 text backbone embedded in step5_vl.
public struct SharpAIStep5VlTextConfiguration: Codable, Sendable {
    public let modelType: String?
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let attentionGroups: Int
    /// Not `hiddenSize / attentionHeads` in this model (4096 / 64 is not 192).
    public let headDim: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int?
    public let slidingWindow: Int?
    public let layerTypes: [String]?
    /// One value per layer, or a single value when the config gives a scalar.
    public let ropeTheta: [Float]
    public let partialRotaryFactors: [Float]?
    public let useHeadWiseAttnGate: Bool?
    public let numNextnPredictLayers: Int?
    public let useMoe: Bool?
    public let moeNumExperts: Int?
    public let moeTopK: Int?
    public let moeIntermediateSize: Int?
    public let shareExpertDim: Int?
    public let sparseConfig: SharpAIStep5VlSparseConfiguration?
    /// Per-layer SwiGLU clamp for the dense MLP (`swiglu_limits`). Nil when absent.
    public let swigluLimits: [Float]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case attentionGroups = "num_attention_groups"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case ropeTheta = "rope_theta"
        case partialRotaryFactors = "partial_rotary_factors"
        case useHeadWiseAttnGate = "use_head_wise_attn_gate"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case useMoe = "use_moe"
        case moeNumExperts = "moe_num_experts"
        case moeTopK = "moe_top_k"
        case moeIntermediateSize = "moe_intermediate_size"
        case shareExpertDim = "share_expert_dim"
        case sparseConfig = "sparse_config"
        case swigluLimits = "swiglu_limits"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        attentionGroups =
            try container.decodeIfPresent(Int.self, forKey: .attentionGroups) ?? attentionHeads
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (attentionHeads > 0 ? hiddenSize / attentionHeads : 0)
        // UNVERIFIED: default 1e-5 when absent.
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        // UNVERIFIED: default 10000 when absent.
        ropeTheta = try Self.decodeFloats(container, .ropeTheta) ?? [10_000]
        partialRotaryFactors = try Self.decodeFloats(container, .partialRotaryFactors)
        useHeadWiseAttnGate = try container.decodeIfPresent(
            Bool.self, forKey: .useHeadWiseAttnGate)
        numNextnPredictLayers = try container.decodeIfPresent(
            Int.self, forKey: .numNextnPredictLayers)
        useMoe = try container.decodeIfPresent(Bool.self, forKey: .useMoe)
        moeNumExperts = try container.decodeIfPresent(Int.self, forKey: .moeNumExperts)
        moeTopK = try container.decodeIfPresent(Int.self, forKey: .moeTopK)
        moeIntermediateSize = try container.decodeIfPresent(
            Int.self, forKey: .moeIntermediateSize)
        shareExpertDim = try container.decodeIfPresent(Int.self, forKey: .shareExpertDim)
        sparseConfig = try container.decodeIfPresent(
            SharpAIStep5VlSparseConfiguration.self, forKey: .sparseConfig)
        swigluLimits = try container.decodeIfPresent([Float].self, forKey: .swigluLimits)
    }

    /// Reads a float list, or a single float as a one-item list. Returns nil when absent or null.
    private static func decodeFloats(
        _ container: KeyedDecodingContainer<CodingKeys>, _ key: CodingKeys
    ) throws -> [Float]? {
        guard container.contains(key) else { return nil }
        if try container.decodeNil(forKey: key) { return nil }
        if let values = try? container.decode([Float].self, forKey: key) { return values }
        return [try container.decode(Float.self, forKey: key)]
    }
}

/// Top-level `config.json` for `model_type: step5_vl`.
public struct SharpAIStep5VlConfiguration: Codable, Sendable {
    public static let expectedArchitecture = "Step5ForConditionalGeneration"

    public let modelType: String
    public let architectures: [String]?
    public let textConfig: SharpAIStep5VlTextConfiguration
    public let visionConfig: SharpAIStep5VlVisionConfiguration
    public let imageTokenId: Int?
    public let imageTokenLen: Int?
    public let patchTokenLen: Int?
    public let understandProjectorStride: Int
    public let projectorBias: Bool
    public let visionSelectLayer: Int
    public let useImStartEnd: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case imageTokenLen = "image_token_len"
        case patchTokenLen = "patch_token_len"
        case understandProjectorStride = "understand_projector_stride"
        case projectorBias = "projector_bias"
        case visionSelectLayer = "vision_select_layer"
        case useImStartEnd = "use_im_start_end"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "step5_vl"
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures)
        textConfig = try container.decode(
            SharpAIStep5VlTextConfiguration.self, forKey: .textConfig)
        visionConfig = try container.decode(
            SharpAIStep5VlVisionConfiguration.self, forKey: .visionConfig)
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId)
        imageTokenLen = try container.decodeIfPresent(Int.self, forKey: .imageTokenLen)
        patchTokenLen = try container.decodeIfPresent(Int.self, forKey: .patchTokenLen)
        understandProjectorStride =
            try container.decodeIfPresent(Int.self, forKey: .understandProjectorStride) ?? 2
        projectorBias = try container.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? false
        visionSelectLayer =
            try container.decodeIfPresent(Int.self, forKey: .visionSelectLayer) ?? -1
        useImStartEnd = try Self.decodeFlag(container, .useImStartEnd) ?? false
    }

    /// The fixture writes `use_im_start_end` as the string "true", so accept both forms.
    private static func decodeFlag(
        _ container: KeyedDecodingContainer<CodingKeys>, _ key: CodingKeys
    ) throws -> Bool? {
        if let flag = try? container.decodeIfPresent(Bool.self, forKey: key) { return flag }
        guard let text = try container.decodeIfPresent(String.self, forKey: key) else {
            return nil
        }
        return text.lowercased() == "true"
    }

    /// Checks the model type, architecture, text layer and head fields, and vision patch size.
    public func validateModelConfiguration() throws {
        guard modelType == "step5_vl" else {
            throw SharpAIStep5VlError.invalidConfiguration("model_type must be step5_vl")
        }
        if let architectures, !architectures.isEmpty,
            !architectures.contains(Self.expectedArchitecture)
        {
            throw SharpAIStep5VlError.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }

        let text = textConfig
        guard text.hiddenLayers > 0 else {
            throw SharpAIStep5VlError.invalidConfiguration(
                "text_config.num_hidden_layers must be > 0")
        }
        guard text.attentionGroups > 0, text.attentionHeads % text.attentionGroups == 0 else {
            throw SharpAIStep5VlError.invalidConfiguration(
                "text_config.num_attention_heads must be a multiple of num_attention_groups")
        }
        if let layerTypes = text.layerTypes {
            // UNVERIFIED: the fixture lists hidden_layers + num_nextn_predict_layers entries.
            guard layerTypes.count >= text.hiddenLayers else {
                throw SharpAIStep5VlError.invalidConfiguration(
                    "text_config.layer_types must cover every hidden layer")
            }
            for layerType in layerTypes
            where layerType != "full_attention" && layerType != "sliding_attention" {
                throw SharpAIStep5VlError.invalidConfiguration(
                    "unknown text_config.layer_types entry \(layerType)")
            }
        }
        guard text.ropeTheta.count == 1 || text.ropeTheta.count >= text.hiddenLayers else {
            throw SharpAIStep5VlError.invalidConfiguration(
                "text_config.rope_theta must be scalar or cover every hidden layer")
        }

        let vision = visionConfig
        guard vision.patchSize > 0, vision.imageSize % vision.patchSize == 0 else {
            throw SharpAIStep5VlError.invalidConfiguration(
                "vision_config.image_size must be a multiple of patch_size")
        }
        guard understandProjectorStride > 0 else {
            throw SharpAIStep5VlError.invalidConfiguration(
                "understand_projector_stride must be > 0")
        }
    }
}
