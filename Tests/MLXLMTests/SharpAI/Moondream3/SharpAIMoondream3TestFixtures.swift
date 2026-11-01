// SharpAI fork overlay test fixtures for moondream3. Not part of upstream.

import Foundation

@testable import MLXVLM

enum SharpAIMoondream3TestFixtures {

    /// Tiny config. Text: width 16, 2 heads, 2 layers, MoE from layer 1 with 4 experts.
    /// Vision: 28 px crop, 14 px patches (2x2 grid), width 8, 2 layers, no overlap.
    /// Image prefix is 1 + 2*2 = 5 positions.
    static let tinyJSON = """
        {
            "model_type": "moondream3",
            "architectures": ["Moondream3ForCausalLM"],
            "config": {
                "text": {
                    "dim": 16,
                    "ff_dim": 32,
                    "n_layers": 2,
                    "vocab_size": 32,
                    "max_context": 64,
                    "n_heads": 2,
                    "n_kv_heads": 2,
                    "prefix_attn": 5,
                    "rope_theta": 10000.0,
                    "moe": {
                        "start_layer": 1,
                        "n_experts": 4,
                        "n_experts_per_tok": 2,
                        "expert_inner_dim": 8
                    }
                },
                "vision": {
                    "enc_dim": 8,
                    "enc_patch_size": 14,
                    "enc_n_layers": 2,
                    "enc_ff_dim": 16,
                    "enc_n_heads": 2,
                    "proj_inner_dim": 12,
                    "crop_size": 28,
                    "max_crops": 4,
                    "overlap_margin": 0
                },
                "tokenizer": {
                    "answer_id": 3
                }
            }
        }
        """

    static func decode(_ json: String = tinyJSON) throws -> SharpAIMoondream3Configuration {
        try JSONDecoder().decode(SharpAIMoondream3Configuration.self, from: Data(json.utf8))
    }
}
