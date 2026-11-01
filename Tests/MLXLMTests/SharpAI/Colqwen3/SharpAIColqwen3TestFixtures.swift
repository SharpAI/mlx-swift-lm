import Foundation

@testable import MLXVLM

/// Tiny ColQwen3 config with random-weight-friendly shapes. Vision and text hidden sizes match.
enum SharpAIColqwen3TestFixtures {

    static let tinyConfigJSON = """
        {
          "architectures": ["ColQwen3"],
          "model_type": "colqwen3",
          "embed_dim": 8,
          "image_token_id": 9,
          "video_token_id": 10,
          "vision_start_token_id": 11,
          "vision_end_token_id": 12,
          "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000,
            "tie_word_embeddings": true,
            "vocab_size": 32,
            "rope_scaling": {
              "mrope_interleaved": true,
              "mrope_section": [1, 1, 2],
              "rope_type": "default"
            }
          },
          "vision_config": {
            "model_type": "qwen3_vl",
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "out_hidden_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "in_channels": 3,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": []
          }
        }
        """

    static var tinyConfigData: Data { Data(tinyConfigJSON.utf8) }

    static func tinyConfiguration() throws -> SharpAIColqwen3Configuration {
        try JSONDecoder().decode(SharpAIColqwen3Configuration.self, from: tinyConfigData)
    }
}
