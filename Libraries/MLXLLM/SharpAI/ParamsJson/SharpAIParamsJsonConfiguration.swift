// SharpAI fork overlay: native params.json configuration for EagleMistralLarge3ForCausalLM.
// Not part of upstream mlx-swift-lm.
//
// Verified against: https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle/resolve/main/params.json
// (the EAGLE drafter's own params.json). Key names, MLA dims (q_lora 1536, kv_lora 512, nope 128,
// rope 64, v 128), moe block and llama_4_scaling match it. The drafter's yarn block sets
// apply_scale=false. vLLM mistral.py turns that into attention_factor=1.0 (no magnitude
// correction), so decode maps it to mscale = mscale_all_dim = 1, which the shared RoPE reads.

import Foundation
import MLXLMCommon

public enum SharpAIParamsJsonError: LocalizedError, Equatable {
    case invalidConfiguration(String)
    case eagleRequiresTargetModel(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "ParamsJson invalid configuration: \(detail)"
        case .eagleRequiresTargetModel(let type):
            return
                "Model type \(type) is an EAGLE drafter. It needs the target model's hidden "
                + "states, which LanguageModel cannot carry, so it cannot load standalone."
        }
    }
}

/// Decoded from a Mistral native `params.json`.
public struct SharpAIParamsJsonConfiguration: Decodable, Sendable {

    public struct MoE: Equatable, Sendable {
        public var numExperts: Int
        public var numExpertsPerTok: Int
        public var expertHiddenDim: Int
        public var numSharedExperts: Int
        public var routedScale: Float
        public var normTopkProb: Bool
    }

    public struct Llama4Scaling: Equatable, Sendable {
        public var beta: Float
        public var originalMaxPositionEmbeddings: Int
    }

    public var dim: Int
    public var nLayers: Int
    public var nHeads: Int
    public var nKVHeads: Int
    public var vocabSize: Int
    public var normEps: Float
    public var ropeTheta: Float
    /// Built from the optional `yarn` block, with `type` set to "yarn". Nil when absent.
    public var ropeScaling: [String: StringOrNumber]?
    public var maxPositionEmbeddings: Int
    public var qLoraRank: Int
    public var kvLoraRank: Int
    public var qkNopeHeadDim: Int
    public var qkRopeHeadDim: Int
    public var vHeadDim: Int
    public var moe: MoE
    public var llama4Scaling: Llama4Scaling?
    public var tiedEmbeddings: Bool

    public var qHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }

    enum CodingKeys: String, CodingKey {
        case dim
        case nLayers = "n_layers"
        case nHeads = "n_heads"
        case nKVHeads = "n_kv_heads"
        case vocabSize = "vocab_size"
        case normEps = "norm_eps"
        case ropeTheta = "rope_theta"
        case yarn
        case maxPositionEmbeddings = "max_position_embeddings"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case moe
        case llama4Scaling = "llama_4_scaling"
        case tiedEmbeddings = "tied_embeddings"
    }

    enum MoECodingKeys: String, CodingKey {
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case expertHiddenDim = "expert_hidden_dim"
        case numSharedExperts = "num_shared_experts"
        case routedScale = "routed_scale"
        case normTopkProb = "norm_topk_prob"
    }

    enum Llama4CodingKeys: String, CodingKey {
        case beta
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dim = try c.decode(Int.self, forKey: .dim)
        nLayers = try c.decode(Int.self, forKey: .nLayers)
        nHeads = try c.decode(Int.self, forKey: .nHeads)
        nKVHeads = try c.decodeIfPresent(Int.self, forKey: .nKVHeads) ?? nHeads
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        qLoraRank = try c.decode(Int.self, forKey: .qLoraRank)
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        tiedEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tiedEmbeddings) ?? false

        if var yarn = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .yarn) {
            yarn["type"] = .string("yarn")
            // vLLM mistral.py: apply_scale false becomes attention_factor 1.0, so no YaRN magnitude
            // correction. The shared RoPE reads mscale and mscale_all_dim, not attention_factor.
            // Equal values give a ratio of 1.0, which is the same result.
            if case .bool(false)? = yarn["apply_scale"] {
                yarn["mscale"] = .int(1)
                yarn["mscale_all_dim"] = .int(1)
            }
            ropeScaling = yarn
        } else {
            ropeScaling = nil
        }

        let m = try c.nestedContainer(keyedBy: MoECodingKeys.self, forKey: .moe)
        moe = MoE(
            numExperts: try m.decode(Int.self, forKey: .numExperts),
            numExpertsPerTok: try m.decode(Int.self, forKey: .numExpertsPerTok),
            expertHiddenDim: try m.decode(Int.self, forKey: .expertHiddenDim),
            numSharedExperts: try m.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0,
            routedScale: try m.decodeIfPresent(Float.self, forKey: .routedScale) ?? 1.0,
            normTopkProb: try m.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        )

        if c.contains(.llama4Scaling) {
            let l = try c.nestedContainer(keyedBy: Llama4CodingKeys.self, forKey: .llama4Scaling)
            llama4Scaling = Llama4Scaling(
                beta: try l.decode(Float.self, forKey: .beta),
                originalMaxPositionEmbeddings: try l.decode(
                    Int.self, forKey: .originalMaxPositionEmbeddings)
            )
        } else {
            llama4Scaling = nil
        }
    }

    /// Decodes params.json bytes with the json5 decoder used by the other loaders.
    public static func decode(_ data: Data) throws -> SharpAIParamsJsonConfiguration {
        try JSONDecoder.json5().decode(SharpAIParamsJsonConfiguration.self, from: data)
    }

    /// Checks sizes that the model needs before it builds any module.
    public func validate() throws {
        let positive: [(String, Int)] = [
            ("dim", dim),
            ("n_layers", nLayers),
            ("n_heads", nHeads),
            ("vocab_size", vocabSize),
            ("q_lora_rank", qLoraRank),
            ("kv_lora_rank", kvLoraRank),
            ("qk_nope_head_dim", qkNopeHeadDim),
            ("qk_rope_head_dim", qkRopeHeadDim),
            ("v_head_dim", vHeadDim),
            ("moe.num_experts", moe.numExperts),
            ("moe.expert_hidden_dim", moe.expertHiddenDim),
        ]
        for (name, value) in positive where value <= 0 {
            throw SharpAIParamsJsonError.invalidConfiguration("\(name) must be positive")
        }
        guard qkRopeHeadDim % 2 == 0 else {
            throw SharpAIParamsJsonError.invalidConfiguration("qk_rope_head_dim must be even")
        }
        guard moe.numExpertsPerTok >= 1, moe.numExpertsPerTok <= moe.numExperts else {
            throw SharpAIParamsJsonError.invalidConfiguration(
                "moe.num_experts_per_tok must be between 1 and moe.num_experts")
        }
        guard moe.numSharedExperts >= 0 else {
            throw SharpAIParamsJsonError.invalidConfiguration(
                "moe.num_shared_experts must not be negative")
        }
        if let scaling = llama4Scaling, scaling.originalMaxPositionEmbeddings <= 0 {
            throw SharpAIParamsJsonError.invalidConfiguration(
                "llama_4_scaling.original_max_position_embeddings must be positive")
        }
    }
}
