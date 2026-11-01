// SharpAI fork overlay: glm5_next configuration. Not part of upstream mlx-swift-lm.
//
// Field names follow the zai-org/GLM-5.3-Flash config.json shape (offline fixture
// scripts/conformance/offline/fixtures/glm5_next). Every field is optional so the decoder
// accepts partial configs. UNVERIFIED: the KDA, kpool, mHC and vision semantics are not
// implemented; these fields only carry the values.

import Foundation
import MLXLMCommon

public enum SharpAIGlm5NextError: LocalizedError {
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its model class is not implemented yet "
                + "(architecture UNVERIFIED)."
        }
    }
}

public struct SharpAIGlm5NextLinearAttentionConfiguration: Decodable, Sendable {
    public let numHeads: Int?
    public let headDim: Int?
    public let shortConvKernelSize: Int?
    public let gateLowerBound: Float?
    public let kdaLayers: [Int]?
    public let fullAttnLayers: [Int]?

    enum CodingKeys: String, CodingKey {
        case numHeads = "num_heads"
        case headDim = "head_dim"
        case shortConvKernelSize = "short_conv_kernel_size"
        case gateLowerBound = "gate_lower_bound"
        case kdaLayers = "kda_layers"
        case fullAttnLayers = "full_attn_layers"
    }
}

public struct SharpAIGlm5NextTextConfiguration: Decodable, Sendable {
    public let modelType: String?
    public let hiddenSize: Int?
    public let intermediateSize: Int?
    public let moeIntermediateSize: Int?
    public let numHiddenLayers: Int?
    public let numAttentionHeads: Int?
    public let numKeyValueHeads: Int?
    public let vocabSize: Int?
    public let rmsNormEps: Float?
    public let maxPositionEmbeddings: Int?
    public let attentionBias: Bool?
    public let tieWordEmbeddings: Bool?

    // MLA (NoPE: qk_rope_head_dim is 0 and mla_use_nope is true in the shipped config).
    public let qLoraRank: Int?
    public let kvLoraRank: Int?
    public let qkNopeHeadDim: Int?
    public let qkRopeHeadDim: Int?
    public let qkHeadDim: Int?
    public let vHeadDim: Int?
    public let mlaUseNope: Bool?

    // DeepSeek sparse attention indexer with kpool compression.
    public let indexTopk: Int?
    public let indexHeadDim: Int?
    public let indexNHeads: Int?
    public let indexKpool: Int?
    public let indexKpoolAlwaysSelectTail: Bool?
    public let indexKpoolCompress: Bool?
    public let indexShareForMtpIteration: Bool?
    public let indexerRopeInterleave: Bool?
    public let indexerTypes: [String]?

    // Hyper-connections (mHC).
    public let mhc: Bool?
    public let hcMult: Int?
    public let hcSinkhornIters: Int?
    public let hcEps: Float?

    // Layer layout.
    public let layerTypes: [String]?
    public let mlpLayerTypes: [String]?
    public let linearAttnConfig: SharpAIGlm5NextLinearAttentionConfiguration?
    public let firstKDenseReplace: Int?
    public let numNextnPredictLayers: Int?

    // MoE.
    public let nRoutedExperts: Int?
    public let nSharedExperts: Int?
    public let numExpertsPerTok: Int?
    public let nGroup: Int?
    public let topkGroup: Int?
    public let routedScalingFactor: Float?
    public let normTopkProb: Bool?
    public let scoringFunc: String?
    public let topkMethod: String?
    public let moeRouterDtype: String?
    public let swigluLimit: Float?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkHeadDim = "qk_head_dim"
        case vHeadDim = "v_head_dim"
        case mlaUseNope = "mla_use_nope"
        case indexTopk = "index_topk"
        case indexHeadDim = "index_head_dim"
        case indexNHeads = "index_n_heads"
        case indexKpool = "index_kpool"
        case indexKpoolAlwaysSelectTail = "index_kpool_always_select_tail"
        case indexKpoolCompress = "index_kpool_compress"
        case indexShareForMtpIteration = "index_share_for_mtp_iteration"
        case indexerRopeInterleave = "indexer_rope_interleave"
        case indexerTypes = "indexer_types"
        case mhc
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case layerTypes = "layer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case linearAttnConfig = "linear_attn_config"
        case firstKDenseReplace = "first_k_dense_replace"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case scoringFunc = "scoring_func"
        case topkMethod = "topk_method"
        case moeRouterDtype = "moe_router_dtype"
        case swigluLimit = "swiglu_limit"
    }
}

public struct SharpAIGlm5NextVisionConfiguration: Decodable, Sendable {
    public let modelType: String?
    public let depth: Int?
    public let hiddenSize: Int?
    public let intermediateSize: Int?
    public let numHeads: Int?
    public let imageSize: Int?
    public let inChannels: Int?
    public let patchSize: Int?
    public let temporalPatchSize: Int?
    public let spatialMergeSize: Int?
    public let outHiddenSize: Int?
    public let projectionIntermediateSize: Int?
    public let rmsNormEps: Float?
    public let attentionBias: Bool?
    public let hiddenAct: String?
    public let swigluLimit: Float?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case depth
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHeads = "num_heads"
        case imageSize = "image_size"
        case inChannels = "in_channels"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case outHiddenSize = "out_hidden_size"
        case projectionIntermediateSize = "projection_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case hiddenAct = "hidden_act"
        case swigluLimit = "swiglu_limit"
    }
}

public struct SharpAIGlm5NextConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "glm5_next"

    public let modelType: String?
    public let architectures: [String]?
    public let textConfig: SharpAIGlm5NextTextConfiguration?
    public let visionConfig: SharpAIGlm5NextVisionConfiguration?
    public let imageTokenId: Int?
    public let videoTokenId: Int?
    public let imageStartTokenId: Int?
    public let imageEndTokenId: Int?
    public let videoStartTokenId: Int?
    public let videoEndTokenId: Int?
    public let tieWordEmbeddings: Bool?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case imageStartTokenId = "image_start_token_id"
        case imageEndTokenId = "image_end_token_id"
        case videoStartTokenId = "video_start_token_id"
        case videoEndTokenId = "video_end_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "glm5_next: model_type must be \(Self.expectedModelType), got \(modelType ?? "nil")")
        }
        guard textConfig != nil else {
            throw ModelFactoryError.invalidConfiguration("glm5_next: text_config is missing")
        }
    }
}
