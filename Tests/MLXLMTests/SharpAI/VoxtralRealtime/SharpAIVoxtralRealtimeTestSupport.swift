// SharpAI fork overlay test support for voxtral_realtime. Not part of upstream.
// Tiny shapes only. Random weights, no checkpoint.

import Foundation
import MLXLMCommon
import MLXVLM

enum SharpAIVoxtralRealtimeTestConfig {

    /// Tiny config. Audio width 8 times downsample 4 gives a projector input of 32.
    static func tiny(numKvHeads: Int = 1, topHiddenSize: Int = 8) -> String {
        """
        {
            "model_type": "voxtral_realtime",
            "hidden_size": \(topHiddenSize),
            "projector_hidden_act": "gelu",
            "audio_length_per_tok": 8,
            "default_num_delay_tokens": 6,
            "downsample_factor": 4,
            "ada_rms_norm_t_cond_dim": 4,
            "audio_config": {
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "intermediate_size": 16,
                "hidden_act": "silu",
                "num_mel_bins": 4,
                "max_position_embeddings": 64,
                "sliding_window": 16,
                "rms_norm_eps": 1e-5,
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
            },
            "text_config": {
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": \(numKvHeads),
                "head_dim": 4,
                "intermediate_size": 16,
                "hidden_act": "silu",
                "vocab_size": 32,
                "max_position_embeddings": 64,
                "sliding_window": 16,
                "rms_norm_eps": 1e-5,
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
                "tie_word_embeddings": true
            }
        }
        """
    }

    static func decodeTiny(numKvHeads: Int = 1, topHiddenSize: Int = 8) throws
        -> SharpAIVoxtralRealtimeConfiguration
    {
        try JSONDecoder().decode(
            SharpAIVoxtralRealtimeConfiguration.self,
            from: Data(tiny(numKvHeads: numKvHeads, topHiddenSize: topHiddenSize).utf8))
    }
}
