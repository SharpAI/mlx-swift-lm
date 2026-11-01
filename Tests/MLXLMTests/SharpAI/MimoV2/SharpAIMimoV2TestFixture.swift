// SharpAI fork overlay test fixture for the mimo_v2 model. Not part of upstream.
// Tiny synthetic config only. No checkpoint values.

import Foundation

enum SharpAIMimoV2TestFixture {

    /// 4 layers: SWA dense, global MoE, SWA MoE, global MoE. Fused qkv with 2 KV chunks.
    static func tinyJSON(
        tied: Bool = false, sharedExperts: Int = 0, omniTowers: Bool = false
    ) -> String {
        let omni = omniTowers ? ", \"vision_config\": {}" : ""
        return """
            {
              "model_type": "mimo_v2",
              "vocab_size": 100,
              "hidden_size": 32,
              "intermediate_size": 48,
              "num_hidden_layers": 4,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 8,
              "swa_num_attention_heads": 4,
              "swa_num_key_value_heads": 2,
              "swa_head_dim": 8,
              "hybrid_layer_pattern": [1, 0, 1, 0],
              "sliding_window_size": 4,
              "rope_theta": 10000.0,
              "swa_rope_theta": 10000.0,
              "max_position_embeddings": 128,
              "layernorm_epsilon": 1e-6,
              "hidden_act": "silu",
              "attention_bias": false,
              "add_swa_attention_sink_bias": true,
              "attention_value_scale": 0.5,
              "attention_projection_layout": "fused_qkv",
              "tie_word_embeddings": \(tied),
              "num_nextn_predict_layers": 1,
              "moe_layer_freq": [0, 1, 1, 1],
              "n_routed_experts": 4,
              "num_experts_per_tok": 2,
              "moe_intermediate_size": 16,
              "n_shared_experts": \(sharedExperts),
              "norm_topk_prob": true,
              "n_group": 1,
              "topk_group": 1,
              "routed_scaling_factor": 1.0\(omni)
            }
            """
    }
}
