// SharpAI ParamsJson test fixtures. Tiny shapes only. No network, no real weights.
// realEagleParamsJSON is copied verbatim from the EAGLE drafter's params.json on Hugging Face
// (mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle). The tiny fixtures are invented for shape.

import Foundation

enum SharpAIParamsJsonTestFixtures {

    /// Verbatim params.json of mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle (1041 bytes).
    static let realEagleParamsJSON = """
        {
          "dim": 7168,
          "n_layers": 1,
          "head_dim": 192,
          "hidden_dim": 16384,
          "n_heads": 128,
          "n_kv_heads": 128,
          "rope_theta": 10000.0,
          "norm_eps": 1e-06,
          "vocab_size": 131072,
          "tied_embeddings": false,
          "max_position_embeddings": 294912,
          "max_seq_len": 16384,
          "llama_4_scaling": {
            "original_max_position_embeddings": 8192,
            "beta": 0.1
          },
          "q_lora_rank": 1536,
          "qk_rope_head_dim": 64,
          "qk_nope_head_dim": 128,
          "kv_lora_rank": 512,
          "v_head_dim": 128,
          "quantization": {
            "qformat_weight": "fp8_e4m3",
            "qscheme_act": "TENSOR"
          },
          "yarn": {
            "original_max_position_embeddings": 8192,
            "factor": 36,
            "apply_scale": false,
            "beta": 32,
            "alpha": 1
          },
          "moe": {
            "expert_parallel": 1,
            "expert_model_parallel": 1,
            "route_every_n": 1,
            "first_k_dense_replace": 0,
            "num_experts": 128,
            "num_experts_per_tok": 4,
            "num_expert_groups": 1,
            "num_expert_groups_per_tok": 1,
            "routed_scale": 1.0,
            "expert_hidden_dim": 4096,
            "num_shared_experts": 1
          }
        }
        """

    static let tinyParamsJSON = """
        {
          "dim": 32,
          "n_layers": 1,
          "n_heads": 4,
          "n_kv_heads": 4,
          "vocab_size": 64,
          "norm_eps": 1e-05,
          "rope_theta": 10000.0,
          "max_position_embeddings": 64,
          "q_lora_rank": 16,
          "kv_lora_rank": 12,
          "qk_nope_head_dim": 8,
          "qk_rope_head_dim": 4,
          "v_head_dim": 8,
          "moe": {
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "expert_hidden_dim": 16,
            "num_shared_experts": 1,
            "routed_scale": 1.0,
            "norm_topk_prob": true
          },
          "llama_4_scaling": {
            "beta": 0.1,
            "original_max_position_embeddings": 8
          },
          "tied_embeddings": false
        }
        """

    static let minimalParamsJSON = """
        {
          "dim": 32,
          "n_layers": 1,
          "n_heads": 4,
          "vocab_size": 64,
          "q_lora_rank": 16,
          "kv_lora_rank": 12,
          "qk_nope_head_dim": 8,
          "qk_rope_head_dim": 4,
          "v_head_dim": 8,
          "moe": {
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "expert_hidden_dim": 16
          }
        }
        """
}
