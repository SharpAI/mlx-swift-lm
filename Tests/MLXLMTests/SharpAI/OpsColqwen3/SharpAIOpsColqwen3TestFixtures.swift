import Foundation

@testable import MLXVLM

/// Tiny ops_colqwen3 config with random-weight-friendly shapes, plus the real checkpoint's
/// config-level values for decode-only tests. Vision and text hidden sizes match.
enum SharpAIOpsColqwen3TestFixtures {

    static let tinyConfigJSON = """
        {
          "architectures": ["OpsColQwen3Model"],
          "model_type": "ops_colqwen3",
          "dims": 8,
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

    /// Config-level shape of the OpenSearch-AI/Ops-Colqwen3-4B checkpoint, with a 2-layer
    /// text stack and 1-block vision tower so it stays small to decode.
    static let realShapedConfigJSON = """
        {
          "architectures": ["OpsColQwen3Model"],
          "model_type": "ops_colqwen3",
          "dims": 2560,
          "image_token_id": 151655,
          "video_token_id": 151656,
          "vision_start_token_id": 151652,
          "vision_end_token_id": 151653,
          "tie_word_embeddings": true,
          "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 2,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000,
            "tie_word_embeddings": true,
            "vocab_size": 151936,
            "hidden_act": "silu",
            "attention_bias": false,
            "rope_scaling": {
              "mrope_interleaved": true,
              "mrope_section": [24, 20, 20],
              "rope_type": "default"
            }
          },
          "vision_config": {
            "model_type": "qwen3_vl",
            "depth": 1,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "out_hidden_size": 2560,
            "num_heads": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "num_position_embeddings": 2304,
            "hidden_act": "gelu_pytorch_tanh",
            "deepstack_visual_indexes": [5, 11, 17]
          }
        }
        """

    static var tinyConfigData: Data { Data(tinyConfigJSON.utf8) }

    static func tinyConfiguration() throws -> SharpAIOpsColqwen3Configuration {
        try JSONDecoder().decode(SharpAIOpsColqwen3Configuration.self, from: tinyConfigData)
    }
}
