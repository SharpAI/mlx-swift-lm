// SharpAI fork test fixture for sarvam_mla. Not part of upstream.

import Foundation

enum SharpAISarvamMlaTestFixture {
    /// Trimmed copy of scripts/conformance/offline/fixtures/sarvam_mla/sarvamai__sarvam-105b.json.
    /// Drops auto_map, attn_implementation and transformers_version. Values match the file.
    static func sarvam105bJSON() -> String {
        """
        {
          "architectures": ["SarvamMLAForCausalLM"],
          "attention_dropout": 0.0,
          "default_theta": 10000.0,
          "dtype": "float32",
          "embedding_dropout": 0.0,
          "eos_token_id": 1,
          "first_k_dense_replace": 1,
          "head_dim": 576,
          "hidden_act": "silu",
          "hidden_size": 4096,
          "initializer_range": 0.006,
          "intermediate_size": 16384,
          "kv_lora_rank": 512,
          "max_position_embeddings": 131072,
          "model_type": "sarvam_mla",
          "moe_intermediate_size": 2048,
          "moe_router_enable_expert_bias": true,
          "num_attention_heads": 64,
          "num_experts": 128,
          "num_experts_per_tok": 8,
          "num_hidden_layers": 32,
          "num_shared_experts": 1,
          "output_dropout": 0.0,
          "output_router_logits": false,
          "pad_token_id": 0,
          "q_head_dim": 192,
          "qk_nope_head_dim": 128,
          "qk_rope_head_dim": 64,
          "rms_norm_eps": 1e-06,
          "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "deepseek_yarn"
          },
          "rope_theta": 10000.0,
          "routed_scaling_factor": 2.5,
          "tie_word_embeddings": false,
          "use_cache": true,
          "use_qk_norm": true,
          "v_head_dim": 128,
          "vocab_size": 262144
        }
        """
    }

    /// Tiny config with the same key layout. Used for validation and sanitize tests.
    static func tinyJSON(
        hiddenLayers: Int = 2, numExperts: Int = 4, headDim: Int = 40, qHeadDim: Int = 24
    ) -> String {
        """
        {
          "model_type": "sarvam_mla",
          "vocab_size": 64,
          "hidden_size": 32,
          "intermediate_size": 64,
          "moe_intermediate_size": 16,
          "num_hidden_layers": \(hiddenLayers),
          "num_attention_heads": 2,
          "max_position_embeddings": 128,
          "rms_norm_eps": 1e-06,
          "rope_theta": 10000.0,
          "kv_lora_rank": 32,
          "qk_nope_head_dim": 16,
          "qk_rope_head_dim": 8,
          "v_head_dim": 16,
          "q_head_dim": \(qHeadDim),
          "head_dim": \(headDim),
          "use_qk_norm": true,
          "first_k_dense_replace": 1,
          "num_experts": \(numExperts),
          "num_experts_per_tok": 2,
          "num_shared_experts": 1,
          "routed_scaling_factor": 2.5,
          "moe_router_enable_expert_bias": true
        }
        """
    }
}
