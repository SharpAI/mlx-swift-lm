// SharpAI fork overlay: tiny qwen4_exp config for building and running the model in tests.
// Same layout as the released config (3 linear layers, 1 full layer, PLE on layer 1, hc streams,
// n-gram PLE, QSA indexer, MoE) at toy sizes. Not part of upstream.

enum SharpAIQwen4ExpTinyFixtures {

    static let configJSON = """
        {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "text_config": {
                "attention_bias": false,
                "eos_token_id": 2,
                "bos_token_id": 1,
                "hc_count": 2,
                "hc_lowrank": 4,
                "head_dim": 8,
                "heads_per_ngram": 2,
                "hidden_act": "silu",
                "hidden_size": 32,
                "indexer_budget": 8,
                "indexer_compress_ratio": 2,
                "indexer_head_dim": 8,
                "indexer_kv_heads": 1,
                "indexer_n_heads": 2,
                "full_attention_interval": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 8,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_value_head_dim": 8,
                "make_ngram_vocab_size_divisible_by": 8,
                "model_type": "qwen4_exp_text",
                "moe_intermediate_size": 16,
                "ngram_size": 3,
                "ngram_vocab_size_base": 50,
                "num_attention_heads": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "num_hidden_layers": 4,
                "num_key_value_heads": 2,
                "output_gate_type": "sigmoid",
                "partial_rotary_factor": 0.5,
                "ple_conv_kernel_size": 4,
                "ple_embed_dim": 32,
                "ple_layer_ids": [2],
                "rms_norm_eps": 1e-06,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [1, 1, 2],
                    "partial_rotary_factor": 0.5,
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                },
                "shared_expert_intermediate_size": 16,
                "vocab_size": 64
            }
        }
        """
}
