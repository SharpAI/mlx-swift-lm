// Copyright © 2025 Apple Inc.

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek_v32.py
// and https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/glm_moe_dsa.py
//
// DeepSeek V3.2 is V3 plus DeepSeek Sparse Attention (DSA): a "lightning indexer"
// scores every cached key against the query and attention is restricted to the
// top `index_topk` of them. GLM-5.2 (`glm_moe_dsa`) is the same architecture — in
// mlx-lm its model class is a bare `class Model(DSV32Model)` over a differently
// shaped config — so both land here.
//
// STAGE 1: the indexer is not implemented; its weights are dropped and attention
// runs dense. This is not an approximation below `index_topk` — the reference
// indexer returns no selection at all until the cache is longer than that:
//
//     if k.shape[2] <= self.index_topk:
//         return None
//
// GLM-5.2 ships `index_topk: 2048`, so output is exact for the first 2048
// positions of context and diverges beyond them (dense rather than top-2048
// sparse — a quality question on long context, not a failure). Implementing the
// indexer is stage 2 and also closes the DeepseekV4Compressor/Indexer TODO.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct DeepseekV32Configuration: Codable, Sendable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var normTopkProb: Bool
    var nGroup: Int?
    var topkGroup: Int?
    var numExpertsPerTok: Int?
    var moeLayerFreq: Int
    var firstKDenseReplace: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    /// Kept as a stored property so the synthesised `Encodable` conformance has
    /// somewhere to put the key; `ropeTheta`/`ropeScaling` above are what the model
    /// actually reads.
    var ropeParameters: [String: StringOrNumber]?
    var attentionBias: Bool

    // DSA-specific. Retained for stage 2 and to document what is being dropped.
    var indexTopk: Int
    var indexNHeads: Int
    var indexHeadDim: Int
    var numNextnPredictLayers: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case indexTopk = "index_topk"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case numNextnPredictLayers = "num_nextn_predict_layers"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        routedScalingFactor = try c.decode(Float.self, forKey: .routedScalingFactor)
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qLoraRank = try c.decode(Int.self, forKey: .qLoraRank)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup)
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup)
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
        moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false

        indexTopk = try c.decodeIfPresent(Int.self, forKey: .indexTopk) ?? 2048
        indexNHeads = try c.decodeIfPresent(Int.self, forKey: .indexNHeads) ?? 0
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 0
        numNextnPredictLayers =
            try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0

        // GLM-5.2 carries no top-level `rope_theta`/`rope_scaling`; both live inside a
        // `rope_parameters` dict, which mlx-lm unpacks in ModelArgs.__post_init__.
        // DeepSeek's own V3.2 configs keep the flat layout, so accept either.
        let ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        self.ropeParameters = ropeParameters

        if let flat = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            ropeTheta = flat
        } else if case .float(let theta) = ropeParameters?["rope_theta"] {
            ropeTheta = theta
        } else if case .int(let theta) = ropeParameters?["rope_theta"] {
            ropeTheta = Float(theta)
        } else {
            throw DecodingError.keyNotFound(
                CodingKeys.ropeTheta,
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription:
                        "no rope_theta at the top level or inside rope_parameters"))
        }

        if let flat = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling) {
            ropeScaling = flat
        } else if let ropeParameters,
            case .string(let type)? = ropeParameters["rope_type"], type != "default"
        {
            // Only forward rope_parameters as scaling when it actually describes a
            // scheme; "default" means plain RoPE and a non-nil dict here would make
            // the V3 stack look for yarn fields that are not there.
            ropeScaling = ropeParameters
        } else {
            ropeScaling = nil
        }
    }

    /// The V3.2 base stack is V3: same MLA projections, same grouped `noaux_tc` gate,
    /// same MoE block. Only the indexer is new, and stage 1 does not build it — so the
    /// model is constructed straight from the equivalent V3 configuration.
    var v3Configuration: DeepseekV3Configuration {
        DeepseekV3Configuration(
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            moeIntermediateSize: moeIntermediateSize,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            nSharedExperts: nSharedExperts,
            nRoutedExperts: nRoutedExperts,
            routedScalingFactor: routedScalingFactor,
            kvLoraRank: kvLoraRank,
            qLoraRank: qLoraRank,
            qkRopeHeadDim: qkRopeHeadDim,
            vHeadDim: vHeadDim,
            qkNopeHeadDim: qkNopeHeadDim,
            normTopkProb: normTopkProb,
            nGroup: nGroup,
            topkGroup: topkGroup,
            numExpertsPerTok: numExpertsPerTok,
            moeLayerFreq: moeLayerFreq,
            firstKDenseReplace: firstKDenseReplace,
            maxPositionEmbeddings: maxPositionEmbeddings,
            rmsNormEps: rmsNormEps,
            ropeTheta: ropeTheta,
            ropeScaling: ropeScaling,
            attentionBias: attentionBias
        )
    }
}

public class DeepseekV32Model: DeepseekV3Model {
    let dsaArgs: DeepseekV32Configuration

    init(_ args: DeepseekV32Configuration) {
        self.dsaArgs = args
        super.init(args.v3Configuration)
    }

    override public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Drop the lightning-indexer sub-module. Stage 1 has no module to load these
        // into, and leaving them in surfaces as an unexpected-key failure rather than
        // anything diagnostic. Mirrors what DeepseekV4 does for the same weights.
        // TODO: implement the Indexer and stop dropping these (stage 2).
        let withoutIndexer = weights.filter { key, _ in
            !key.contains(".indexer.") && !key.contains(".attn.compressor.")
        }
        return super.sanitize(weights: withoutIndexer)
    }
}
