// SharpAI fork overlay test fixtures for hy_v3. Not part of upstream.
//
// Inline JSON only. No network, no real weights.

import Foundation

enum SharpAIHyV3TestFixture {

    /// Tiny config: 3 layers, layer 0 dense, layers 1-2 sparse MoE with 2 experts.
    static func tinyJSON(tied: Bool = false, sigmoid: Bool = true) -> String {
        """
        {
            "model_type": "hy_v3",
            "architectures": ["HYV3ForCausalLM"],
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-05,
            "rope_parameters": { "rope_theta": 10000.0, "rope_type": "default" },
            "max_position_embeddings": 128,
            "qk_norm": true,
            "first_k_dense_replace": 1,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "num_shared_experts": 1,
            "moe_router_use_sigmoid": \(sigmoid ? "true" : "false"),
            "moe_router_enable_expert_bias": true,
            "route_norm": true,
            "router_scaling_factor": 2.0,
            "tie_word_embeddings": \(tied ? "true" : "false"),
            "num_nextn_predict_layers": 1
        }
        """
    }

    /// Fields from tencent/Hy3 config.json, copied from scripts/conformance/offline/fixtures/hy_v3.
    static let hy3JSON = """
        {
          "architectures": ["HYV3ForCausalLM"],
          "bos_token_id": 120000,
          "dtype": "bfloat16",
          "eos_token_id": 120025,
          "expert_hidden_dim": 1536,
          "moe_intermediate_size": 1536,
          "first_k_dense_replace": 1,
          "head_dim": 128,
          "hidden_act": "silu",
          "hidden_size": 4096,
          "intermediate_size": 13312,
          "max_position_embeddings": 262144,
          "model_type": "hy_v3",
          "moe_router_enable_expert_bias": true,
          "moe_router_use_sigmoid": true,
          "num_attention_heads": 64,
          "num_experts": 192,
          "num_experts_per_tok": 8,
          "num_hidden_layers": 80,
          "num_key_value_heads": 8,
          "num_shared_experts": 1,
          "qk_norm": true,
          "rms_norm_eps": 1e-05,
          "rope_parameters": { "rope_theta": 11158840.0, "rope_type": "default" },
          "route_norm": true,
          "router_scaling_factor": 2.826,
          "tie_word_embeddings": false,
          "vocab_size": 120832,
          "num_nextn_predict_layers": 1
        }
        """
}
