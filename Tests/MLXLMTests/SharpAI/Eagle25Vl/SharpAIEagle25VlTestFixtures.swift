// SharpAI fork overlay test fixtures for eagle_2_5_vl. Not part of upstream.

import Foundation

@testable import MLXVLM

enum SharpAIEagle25VlTestFixtures {

    /// Tiny config. Vision: 28x28 image, 14-pixel patches (2x2 grid), width 8, 2 layers.
    /// Text: Qwen2 width 16, 2 layers, 4 query heads, 2 KV heads, vocab 32.
    /// Pixel shuffle 0.5 gives one image token per tile.
    static let tinyJSON = """
        {
            "model_type": "eagle_2_5_vl",
            "architectures": ["Eagle2_5_VLForConditionalGeneration"],
            "downsample_ratio": 0.5,
            "select_layer": -1,
            "dynamic_image_size": true,
            "use_thumbnail": true,
            "min_dynamic_patch": 1,
            "max_dynamic_patch": 4,
            "img_context_token_id": 31,
            "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_channels": 3,
                "image_size": 28,
                "patch_size": 14,
                "layer_norm_eps": 1e-6,
                "hidden_act": "gelu_pytorch_tanh"
            },
            "text_config": {
                "model_type": "qwen2",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }
        }
        """

    static func decode(_ json: String = tinyJSON) throws -> SharpAIEagle25VlConfiguration {
        try JSONDecoder().decode(SharpAIEagle25VlConfiguration.self, from: Data(json.utf8))
    }
}
