// SharpAI fork overlay test fixtures for longcat_flash. Not part of upstream.

import Foundation
import MLXLMCommon

@testable import MLXLLM

enum SharpAILongcatFlashTestFixtures {
    /// Tiny shapes: hidden 16, 4 routed and 2 zero experts, top-2 routing, n-gram order 3.
    /// The n-gram table has 4 * 32 = 128 rows, split across 4 embedders (k * (n - 1)).
    static func json(numLayers: Int = 2) -> String {
        """
            {
                "model_type": "longcat_flash",
                "architectures": ["LongcatFlashNgramForCausalLM"],
                "vocab_size": 32,
                "hidden_size": 16,
                "ffn_hidden_size": 24,
                "expert_ffn_hidden_size": 8,
                "num_layers": \(numLayers),
                "num_attention_heads": 2,
                "kv_lora_rank": 8,
                "q_lora_rank": 8,
                "qk_rope_head_dim": 2,
                "qk_nope_head_dim": 4,
                "v_head_dim": 4,
                "mla_scale_q_lora": true,
                "mla_scale_kv_lora": true,
                "routed_scaling_factor": 6.0,
                "n_routed_experts": 4,
                "zero_expert_num": 2,
                "zero_expert_type": "identity",
                "moe_topk": 2,
                "rms_norm_eps": 1e-5,
                "rope_theta": 5000000.0,
                "max_position_embeddings": 128,
                "ngram_vocab_size_ratio": 4,
                "emb_neighbor_num": 3,
                "emb_split_num": 2
            }
            """
    }

    static func configuration(numLayers: Int = 2) throws -> SharpAILongcatFlashConfiguration {
        try JSONDecoder.json5().decode(
            SharpAILongcatFlashConfiguration.self, from: Data(json(numLayers: numLayers).utf8))
    }

    static func model(numLayers: Int = 2) throws -> SharpAILongcatFlashModel {
        SharpAILongcatFlashModel(try configuration(numLayers: numLayers))
    }
}
