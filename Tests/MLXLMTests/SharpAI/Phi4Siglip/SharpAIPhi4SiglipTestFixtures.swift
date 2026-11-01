// SharpAI fork overlay test fixtures for phi4-siglip. Not part of upstream.

import Foundation

@testable import MLXVLM

enum SharpAIPhi4SiglipTestFixtures {

    /// Values from the offline fixture for microsoft/Phi-4-reasoning-vision-15B. The fixture has
    /// no patch size, position count or image token index, so the defaults apply.
    static let fixtureJSON = """
        {
            "architectures": ["Phi4ForCausalLMV"],
            "attention_bias": false,
            "bos_token_id": 100257,
            "eos_token_id": 100265,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "intermediate_size": 17920,
            "max_num_patches": 3600,
            "min_num_patches": 256,
            "max_position_embeddings": 32768,
            "mm_hidden_size": 1152,
            "mm_projector_type": "mlp2x_gelu",
            "mm_vision_tower": "google/siglip2-so400m-patch16-naflex",
            "model_type": "phi4-siglip",
            "num_attention_heads": 40,
            "num_hidden_layers": 40,
            "num_key_value_heads": 10,
            "original_max_position_embeddings": 32768,
            "partial_rotary_factor": 1.0,
            "pad_token_id": 100349,
            "rms_norm_eps": 1e-05,
            "rope_scaling": null,
            "rope_theta": 500000,
            "tie_word_embeddings": false,
            "vocab_size": 100352,
            "vision_config": {
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "model_type": "siglip2_vision_model",
                "num_attention_heads": 16,
                "num_hidden_layers": 27
            }
        }
        """

    /// Tiny config. Vision: 2-pixel patches, 2x2 position table, width 8, 2 layers, 2 heads.
    /// Text: width 16, 2 layers, 4 query heads, 2 KV heads, vocab 32. Image token id 31.
    static let tinyJSON = """
        {
            "model_type": "phi4-siglip",
            "architectures": ["Phi4ForCausalLMV"],
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "rope_scaling": null,
            "tie_word_embeddings": false,
            "mm_hidden_size": 8,
            "mm_projector_type": "mlp2x_gelu",
            "max_num_patches": 16,
            "min_num_patches": 1,
            "image_token_index": 31,
            "vision_config": {
                "model_type": "siglip2_vision_model",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_channels": 3,
                "patch_size": 2,
                "num_patches": 4,
                "layer_norm_eps": 1e-6,
                "hidden_act": "gelu_pytorch_tanh"
            }
        }
        """

    static func decode(_ json: String = tinyJSON) throws -> SharpAIPhi4SiglipConfiguration {
        try JSONDecoder().decode(SharpAIPhi4SiglipConfiguration.self, from: Data(json.utf8))
    }
}
