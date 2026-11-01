// SharpAI fork overlay: kimi_k3 text_config (KimiLinearConfig). Not part of upstream mlx-swift-lm.
//
// Keys and defaults follow moonshotai/Kimi-K3 configuration_kimi_k3.py (KimiLinearConfig, fetched 2026-10-30)
// and the checkpoint config.json (text_config). Keys the forward pass does not read (rope_theta, quantization_config,
// topk_method, ...) are not decoded. Verified against: https://huggingface.co/moonshotai/Kimi-K3/resolve/main/config.json

import Foundation
import MLXLMCommon

/// `linear_attn_config`: which layers are Kimi Delta Attention (KDA) and the KDA head layout.
/// Layer lists are 1-based in the config (`(layer_idx + 1) in kda_layers`).
public struct SharpAIKimiK3LinearAttentionConfiguration: Decodable, Sendable {
    public let kdaLayers: [Int]
    public let fullAttnLayers: [Int]
    public let numHeads: Int
    public let headDim: Int
    public let shortConvKernelSize: Int
    public let useFullRankGate: Bool
    /// `gate_lower_bound`. When set, the decay is `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`.
    public let gateLowerBound: Float?

    enum CodingKeys: String, CodingKey {
        case kdaLayers = "kda_layers"
        case fullAttnLayers = "full_attn_layers"
        case numHeads = "num_heads"
        case headDim = "head_dim"
        case shortConvKernelSize = "short_conv_kernel_size"
        case useFullRankGate = "use_full_rank_gate"
        case gateLowerBound = "gate_lower_bound"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        kdaLayers = try container.decode([Int].self, forKey: .kdaLayers)
        fullAttnLayers = try container.decode([Int].self, forKey: .fullAttnLayers)
        numHeads = try container.decode(Int.self, forKey: .numHeads)
        headDim = try container.decode(Int.self, forKey: .headDim)
        shortConvKernelSize =
            try container.decodeIfPresent(Int.self, forKey: .shortConvKernelSize) ?? 4
        useFullRankGate = try container.decodeIfPresent(Bool.self, forKey: .useFullRankGate) ?? false
        gateLowerBound = try container.decodeIfPresent(Float.self, forKey: .gateLowerBound)
    }
}

/// Typed `text_config` for `model_type: kimi_k3` (the `kimi_linear` text backbone).
public struct SharpAIKimiK3TextConfiguration: Decodable, Sendable {
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let hiddenAct: String
    public let rmsNormEps: Float

    // MoE
    public let numExperts: Int?
    public let numExpertsPerToken: Int?
    public let moeIntermediateSize: Int?
    public let numSharedExperts: Int
    public let moeRenormalize: Bool
    public let moeRouterActivationFunc: String
    public let routedScalingFactor: Float
    public let firstKDenseReplace: Int
    public let moeLayerFreq: Int
    public let numExpertGroup: Int
    public let topkGroup: Int
    /// Latent MoE: routed experts run at this width (down-projected from hidden_size and up-projected back).
    public let routedExpertHiddenSize: Int?
    public let latentMoeUseNorm: Bool

    // MLA (multi-head latent attention, full-attention layers)
    public let qLoraRank: Int?
    public let kvLoraRank: Int?
    public let qkNopeHeadDim: Int?
    public let qkRopeHeadDim: Int?
    public let vHeadDim: Int?
    public let mlaUseNope: Bool
    public let mlaUseOutputGate: Bool

    // Situ activation parameters (`situ` hidden_act)
    public let activationSituBeta: Float?
    public let activationSituLinearBeta: Float?

    /// Attention residual block size. nil disables the block residual bank (plain residuals).
    public let attnResBlockSize: Int?

    public let linearAttnConfig: SharpAIKimiK3LinearAttentionConfiguration?

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case hiddenAct = "hidden_act"
        case rmsNormEps = "rms_norm_eps"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_token"
        case moeIntermediateSize = "moe_intermediate_size"
        case numSharedExperts = "num_shared_experts"
        case moeRenormalize = "moe_renormalize"
        case moeRouterActivationFunc = "moe_router_activation_func"
        case routedScalingFactor = "routed_scaling_factor"
        case firstKDenseReplace = "first_k_dense_replace"
        case moeLayerFreq = "moe_layer_freq"
        case numExpertGroup = "num_expert_group"
        case topkGroup = "topk_group"
        case routedExpertHiddenSize = "routed_expert_hidden_size"
        case latentMoeUseNorm = "latent_moe_use_norm"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case mlaUseNope = "mla_use_nope"
        case mlaUseOutputGate = "mla_use_output_gate"
        case activationSituBeta = "activation_situ_beta"
        case activationSituLinearBeta = "activation_situ_linear_beta"
        case attnResBlockSize = "attn_res_block_size"
        case linearAttnConfig = "linear_attn_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 163840
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11008
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6

        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts)
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken)
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0
        moeRenormalize = try c.decodeIfPresent(Bool.self, forKey: .moeRenormalize) ?? true
        moeRouterActivationFunc =
            try c.decodeIfPresent(String.self, forKey: .moeRouterActivationFunc) ?? "sigmoid"
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        numExpertGroup = try c.decodeIfPresent(Int.self, forKey: .numExpertGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        routedExpertHiddenSize = try c.decodeIfPresent(Int.self, forKey: .routedExpertHiddenSize)
        latentMoeUseNorm = try c.decodeIfPresent(Bool.self, forKey: .latentMoeUseNorm) ?? false

        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank)
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim)
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim)
        mlaUseNope = try c.decodeIfPresent(Bool.self, forKey: .mlaUseNope) ?? false
        mlaUseOutputGate = try c.decodeIfPresent(Bool.self, forKey: .mlaUseOutputGate) ?? false

        activationSituBeta = try c.decodeIfPresent(Float.self, forKey: .activationSituBeta)
        activationSituLinearBeta =
            try c.decodeIfPresent(Float.self, forKey: .activationSituLinearBeta)
        attnResBlockSize = try c.decodeIfPresent(Int.self, forKey: .attnResBlockSize)
        linearAttnConfig =
            try c.decodeIfPresent(
                SharpAIKimiK3LinearAttentionConfiguration.self, forKey: .linearAttnConfig)
    }

    /// True when layer `index` (0-based) is a KDA layer: `(index + 1) in kda_layers`.
    public func isKDALayer(_ index: Int) -> Bool {
        guard let linearAttnConfig else { return false }
        return linearAttnConfig.kdaLayers.contains(index + 1)
    }

    /// Rejects configurations the Swift port does not implement. Throws `SharpAIKimiK3Error.unsupported`.
    public func validateForConstruction() throws {
        func unsupported(_ detail: String) -> SharpAIKimiK3Error {
            .unsupported(detail)
        }
        guard hiddenAct == "situ" || hiddenAct == "silu" else {
            throw unsupported("hidden_act \(hiddenAct) is not ported (situ and silu are)")
        }
        guard qLoraRank != nil, kvLoraRank != nil, qkNopeHeadDim != nil,
            qkRopeHeadDim != nil, vHeadDim != nil, mlaUseNope
        else {
            throw unsupported(
                "MLA needs q_lora_rank, kv_lora_rank, qk_nope/rope_head_dim, v_head_dim and mla_use_nope")
        }
        guard kvHeads == attentionHeads else {
            throw unsupported("MLA layers need num_key_value_heads == num_attention_heads")
        }
        guard numExpertGroup <= 1 else {
            throw unsupported("grouped top-k routing (num_expert_group > 1) is not ported")
        }
        guard moeRouterActivationFunc == "sigmoid" || moeRouterActivationFunc == "softmax" else {
            throw unsupported("router activation \(moeRouterActivationFunc) is not ported")
        }
        if numExperts != nil {
            guard numExpertsPerToken != nil, moeIntermediateSize != nil else {
                throw unsupported("MoE needs num_experts_per_token and moe_intermediate_size")
            }
        }
        if let linearAttnConfig {
            guard linearAttnConfig.numHeads > 0, linearAttnConfig.headDim > 0 else {
                throw unsupported("linear_attn_config needs positive num_heads and head_dim")
            }
        }
    }
}

extension SharpAIKimiK3Configuration {
    /// Decodes `text_config` into the typed configuration.
    public func decodedTextConfiguration() throws -> SharpAIKimiK3TextConfiguration {
        let data = try JSONEncoder().encode(textConfig)
        return try JSONDecoder.json5().decode(SharpAIKimiK3TextConfiguration.self, from: data)
    }
}
