// SharpAI fork overlay test fixtures for kanana-1.5-v. Not part of upstream.

import Foundation
import MLXLMCommon

@testable import MLXVLM

enum SharpAIKanana15VTestFixtures {

    /// Tiny config. Text: Llama width 16, 1 layer, 2 heads, 1 KV head, vocab 32, untied head.
    /// Vision: 4x4 patch grid, patch 2, temporal 2, width 8, 2 heads, depth 1.
    static let tinyJSON = """
        {
            "architectures": ["KananaVForConditionalGeneration"],
            "model_type": "kanana-1.5-v",
            "text_config": {
                "model_type": "kanana-1.5-3b-instruct",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rms_norm_eps": 1e-05,
                "vocab_size": 32,
                "max_position_embeddings": 64,
                "rope_theta": 8000000.0,
                "tie_word_embeddings": false
            },
            "vision_config": {
                "model_type": "kanana-1.5-v-visual-encoder",
                "depth": 1,
                "embed_dim": 8,
                "hidden_size": 16,
                "num_heads": 2,
                "mlp_ratio": 4,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "in_channels": 3,
                "hidden_act": "quick_gelu"
            },
            "projector_config": {
                "projector_type": "dynamic-c-abs",
                "depth": 2,
                "mlp_depth": 2,
                "encoder_hidden_size": 8,
                "hidden_size": 16,
                "output_hidden_size": 16,
                "merge_size": 2,
                "pos_emb": true,
                "pos_emb_size": 4
            }
        }
        """

    static func decode(_ json: String = tinyJSON) throws -> SharpAIKanana15VConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIKanana15VConfiguration.self, from: Data(json.utf8))
    }
}
