// SharpAI fork overlay test fixture for step3p5. Not part of upstream.
//
// A tiny synthetic config with the same key names as the real checkpoint. No weights, no network.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation

enum SharpAIStep3p5TestFixture {

    /// Three layers. Layer 0 is dense and full attention with 4 heads. Layer 1 is a sparse MoE
    /// block with sliding attention and 6 heads. Layer 2 is a sparse MoE block, full, 4 heads.
    /// Head dim 8. Layer 3 is an MTP layer that the target model ignores.
    static var baseOverrides: [String: Any] {
        return [
        "architectures": ["Step3p5ForCausalLM"],
        "model_type": "step3p5",
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 3,
        "vocab_size": 100,
        "num_attention_heads": 4,
        "num_attention_groups": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "attention_other_setting": [
            "attention_type": "sliding_attention",
            "num_attention_heads": 6,
            "num_attention_groups": 2,
            "head_dim": 8,
        ],
        "layer_types": ["full_attention", "sliding_attention", "full_attention", "full_attention"],
        "sliding_window": 4,
        "rope_theta": [10000.0, 10000.0, 5000000.0, 5000000.0],
        "partial_rotary_factors": [0.5, 1.0, 0.5, 0.5],
        "rope_scaling": [
            "rope_type": "llama3",
            "factor": 2.0,
            "original_max_position_embeddings": 32,
            "low_freq_factor": 1.0,
            "high_freq_factor": 32.0,
        ],
        "yarn_only_types": ["full_attention"],
        "use_head_wise_attn_gate": true,
        "att_impl_type": "GQA",
        "hidden_act": "silu",
        "moe_layers_enum": "1,2",
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "moe_intermediate_size": 16,
        "share_expert_dim": 16,
        "moe_router_activation": "sigmoid",
        "moe_router_scaling_factor": 3.0,
        "norm_expert_weight": true,
        "use_moe_router_bias": true,
        "need_fp32_gate": true,
        "swiglu_limits": [0.0, 7.0, 7.0, 0.0],
        "swiglu_limits_shared": [0.0, 16.0, 0.0, 0.0],
        "num_nextn_predict_layers": 1,
        "tie_word_embeddings": false,
        ]
    }

    /// The base config with `overrides` applied. Use `NSNull()` to write a JSON null.
    static func json(_ overrides: [String: Any] = [:]) -> String {
        var config = baseOverrides
        for (key, value) in overrides {
            config[key] = value
        }
        let data = try! JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }
}
