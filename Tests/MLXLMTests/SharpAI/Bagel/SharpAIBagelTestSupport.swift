// SharpAI fork overlay test fixtures for bagel. Not part of upstream.

import Foundation

enum SharpAIBagelTestFixtures {

    /// Tiny bagel config. Text width 16, 2 layers, 4 query heads, 2 KV heads, vocab 32.
    /// Vision width 8, 1 layer, 28x28 image with 14-pixel patches (2x2 grid).
    static let tinyJSON = """
        {
            "architectures": ["BagelForConditionalGeneration"],
            "model_type": "bagel",
            "connector_act": "gelu_pytorch_tanh",
            "vit_max_num_patch_per_side": 2,
            "image_token_index": 31,
            "llm_config": {
                "model_type": "qwen2",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false,
                "qk_norm": true
            },
            "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_channels": 3,
                "image_size": 28,
                "patch_size": 14,
                "layer_norm_eps": 1e-6
            }
        }
        """
}
