// SharpAI fork overlay test fixtures for laguna. Not part of upstream.
//
// Inline JSON only. XS.2 values come from scripts/conformance/offline/fixtures/laguna.

import Foundation

enum SharpAILagunaTestFixture {

    /// Laguna-XS.2 target config (40 layers). Layer 0 is full attention with a dense MLP.
    static func xs2JSON() -> String {
        let layerTypes = (0 ..< 40)
            .map { $0 % 4 == 0 ? "\"full_attention\"" : "\"sliding_attention\"" }
            .joined(separator: ", ")
        let mlpLayerTypes = (0 ..< 40)
            .map { $0 == 0 ? "\"dense\"" : "\"sparse\"" }
            .joined(separator: ", ")
        let headsPerLayer = (0 ..< 40)
            .map { $0 % 4 == 0 ? "48" : "64" }
            .joined(separator: ", ")

        return """
            {
              "architectures": ["LagunaForCausalLM"],
              "model_type": "laguna",
              "vocab_size": 100352,
              "hidden_size": 2048,
              "intermediate_size": 8192,
              "num_hidden_layers": 40,
              "num_attention_heads": 48,
              "num_key_value_heads": 8,
              "head_dim": 128,
              "max_position_embeddings": 262144,
              "attention_bias": false,
              "rms_norm_eps": 1e-06,
              "num_experts": 256,
              "num_experts_per_tok": 8,
              "moe_intermediate_size": 512,
              "shared_expert_intermediate_size": 512,
              "tie_word_embeddings": false,
              "hidden_act": "silu",
              "gating": true,
              "sliding_window": 512,
              "rope_parameters": {
                "full_attention": {
                  "rope_theta": 500000.0,
                  "rope_type": "yarn",
                  "factor": 64.0,
                  "original_max_position_embeddings": 4096,
                  "beta_slow": 1.0,
                  "beta_fast": 64.0,
                  "attention_factor": 1.4158883083359672,
                  "partial_rotary_factor": 0.5
                },
                "sliding_attention": {
                  "rope_type": "default",
                  "rope_theta": 10000.0,
                  "partial_rotary_factor": 1.0
                }
              },
              "layer_types": [\(layerTypes)],
              "mlp_layer_types": [\(mlpLayerTypes)],
              "num_attention_heads_per_layer": [\(headsPerLayer)],
              "moe_apply_router_weight_on_input": false,
              "moe_routed_scaling_factor": 2.5
            }
            """
    }

    /// Laguna-XS.2-DFlash drafter config. It shares model_type "laguna" but is not the target.
    static let dflashJSON = """
        {
          "architectures": ["DFlashLagunaForCausalLM"],
          "model_type": "laguna",
          "vocab_size": 100352,
          "hidden_size": 2048,
          "intermediate_size": 8192,
          "num_hidden_layers": 5,
          "num_attention_heads": 64,
          "num_key_value_heads": 8,
          "head_dim": 128,
          "max_position_embeddings": 262144,
          "attention_bias": false,
          "rms_norm_eps": 1e-06,
          "rope_theta": 500000.0,
          "sliding_window": 512,
          "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
          "gating": "per-head",
          "num_experts": 0
        }
        """

    /// Two layers: layer 0 is full attention with a dense MLP. Layer 1 is sliding attention
    /// with a sparse MoE. Per-layer heads are 4 and 6. The gate is per head.
    static func tinyJSON(
        tied: Bool = false, gating: String = "true", routerSoftcap: Double = 0
    ) -> String {
        """
        {
          "model_type": "laguna",
          "vocab_size": 100,
          "hidden_size": 32,
          "intermediate_size": 64,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_attention_heads_per_layer": [4, 6],
          "num_key_value_heads": 2,
          "head_dim": 8,
          "rms_norm_eps": 1e-06,
          "max_position_embeddings": 128,
          "hidden_act": "silu",
          "tie_word_embeddings": \(tied),
          "attention_bias": false,
          "gating": \(gating),
          "sliding_window": 4,
          "layer_types": ["full_attention", "sliding_attention"],
          "rope_parameters": {
            "full_attention": {
              "rope_type": "yarn",
              "rope_theta": 500000.0,
              "factor": 4.0,
              "original_max_position_embeddings": 64,
              "beta_fast": 32.0,
              "beta_slow": 1.0,
              "attention_factor": 1.1386294,
              "partial_rotary_factor": 0.5
            },
            "sliding_attention": {
              "rope_type": "default",
              "rope_theta": 10000.0,
              "partial_rotary_factor": 1.0
            }
          },
          "mlp_layer_types": ["dense", "sparse"],
          "num_experts": 4,
          "num_experts_per_tok": 2,
          "moe_intermediate_size": 16,
          "shared_expert_intermediate_size": 16,
          "moe_routed_scaling_factor": 2.5,
          "moe_apply_router_weight_on_input": false,
          "moe_router_logit_softcapping": \(routerSoftcap)
        }
        """
    }
}
