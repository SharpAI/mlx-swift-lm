// SharpAI fork overlay test fixtures for nemotron_parse. Not part of upstream.

import Foundation

enum SharpAINemotronParseTestFixtures {

    /// Reduced-size model: the encoder dimensions come from the optional keys (absent in the real
    /// checkpoint, which uses ViT-H). A 64x64 image gives 4x4 patches and 5 encoder tokens.
    static let tinyJSON = """
        {
            "model_type": "nemotron_parse",
            "architectures": ["NemotronParseForConditionalGeneration"],
            "encoder": {
                "model_type": "radio",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "mlp_dim": 128
            },
            "decoder": {
                "d_model": 64,
                "decoder_layers": 1,
                "decoder_attention_heads": 4,
                "decoder_ffn_dim": 128,
                "encoder_ffn_dim": 128,
                "vocab_size": 128
            },
            "decoder_start_token_id": 2,
            "pad_token_id": 1
        }
        """

    /// `preprocessor_config.json` of nvidia/NVIDIA-Nemotron-Parse-v1.1, as published.
    static let hubPreprocessorJSON = """
        {
          "feature_extractor_type": "NemotronParseImageProcessor",
          "image_processor_type": "NemotronParseImageProcessor",
          "processor_class": "NemotronParseProcessor",
          "do_normalize": true,
          "image_mean": [0.48145466, 0.4578275, 0.40821073],
          "image_std": [0.26862954, 0.26130258, 0.27577711],
          "do_rescale": true,
          "rescale_factor": 0.00392156862745098,
          "size": { "height": 2048, "width": 1648, "longest_edge": [2048, 1648] },
          "final_size": [2048, 1648]
        }
        """
}
