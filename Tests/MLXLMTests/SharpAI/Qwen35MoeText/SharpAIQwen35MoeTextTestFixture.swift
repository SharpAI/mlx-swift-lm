// SharpAI fork overlay test fixture for qwen3_5_moe_text. Not part of upstream.

import Foundation

enum SharpAIQwen35MoeTextTestFixture {

    /// Tiny text-only MoE config. Hidden 64, 2 layers, 4 experts with top-2 routing.
    static func tinyJSON(
        numExperts: Int = 4, numExpertsPerTok: Int = 2, modelType: String = "qwen3_5_moe_text",
        architectures: String = "\"Qwen3_5MoeForCausalLM\""
    ) -> String {
        """
        {
            "model_type": "\(modelType)",
            "architectures": [\(architectures)],
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 100,
            "full_attention_interval": 2,
            "num_experts": \(numExperts),
            "num_experts_per_tok": \(numExpertsPerTok),
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32
        }
        """
    }
}
