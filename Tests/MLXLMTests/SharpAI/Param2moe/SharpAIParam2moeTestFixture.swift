// SharpAI fork overlay test fixtures for param2moe. Not part of upstream.

import Foundation

enum SharpAIParam2moeTestFixture {

    /// Copy of scripts/conformance/offline/fixtures/param2moe/bharatgenai__Param2-17B-A2.4B-Thinking.json.
    static let fullJSON = """
        {
          "architectures": ["Param2MoEForCausalLM"],
          "model_type": "param2moe",
          "attention_dropout": 0.0,
          "dtype": "bfloat16",
          "eos_token_id": 3,
          "first_k_dense_replace": 1,
          "head_dim": 64,
          "hidden_act": "silu",
          "hidden_size": 2048,
          "intermediate_size": 9216,
          "max_position_embeddings": 32768,
          "moe_intermediate_size": 2048,
          "moe_router_enable_expert_bias": true,
          "moe_shared_expert_intermediate_size": 4096,
          "mtp_loss_scaling_factor": 0,
          "n_group": 1,
          "norm_topk_prob": true,
          "num_attention_heads": 32,
          "num_experts": 64,
          "num_experts_per_tok": 6,
          "num_hidden_layers": 21,
          "num_key_value_heads": 8,
          "num_nextn_predict_layers": 0,
          "num_shared_experts": 2,
          "partial_rotary_factor": 1.0,
          "rms_norm_eps": 1e-06,
          "rope_parameters": {
            "partial_rotary_factor": 1.0,
            "rope_theta": 500000,
            "rope_type": "default"
          },
          "rope_theta": 500000,
          "routed_scaling_factor": 2.5,
          "router_dtype": "fp32",
          "score_function": "sigmoid",
          "tie_word_embeddings": true,
          "topk_group": 1,
          "use_bias": false,
          "use_qk_norm": true,
          "use_qkv_bias": false,
          "use_rmsnorm": true,
          "vocab_size": 128000
        }
        """

    /// A small model with the same layout: dense layer 0, sparse layers 1 and 2, two groups.
    static func tinyJSON(_ overrides: [String: Any] = [:]) throws -> String {
        var object: [String: Any] = [
            "model_type": "param2moe",
            "architectures": ["Param2MoEForCausalLM"],
            "vocab_size": 100,
            "hidden_size": 32,
            "num_hidden_layers": 3,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 0.000001,
            "max_position_embeddings": 128,
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
            "use_bias": false,
            "use_qkv_bias": false,
            "use_qk_norm": true,
            "tie_word_embeddings": true,
            "first_k_dense_replace": 1,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "num_shared_experts": 1,
            "moe_shared_expert_intermediate_size": 16,
            "moe_router_enable_expert_bias": true,
            "routed_scaling_factor": 2.5,
            "score_function": "sigmoid",
            "norm_topk_prob": true,
            "n_group": 2,
            "topk_group": 1,
        ]
        for (key, value) in overrides {
            object[key] = value
        }
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }
}
