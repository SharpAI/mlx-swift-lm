// SharpAI fork overlay test helper for moss_transcribe_diarize. Inline JSON only, no weights.

import Foundation
@testable import MLXVLM

enum SharpAIMossTranscribeDiarizeTestConfig {

    /// Tiny config. Audio token id is 30. A 15-frame clip gives 8 conv frames and 4 audio tokens
    /// (downsample factor 2).
    static func tiny(activation: String = "gelu", downsample: Int = 2) -> String {
        """
        {
          "model_type": "moss_transcribe_diarize",
          "audio_token_id": 30,
          "audio_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_mel_bins": 4,
            "max_source_positions": 32,
            "activation_function": "\(activation)",
            "downsample_factor": \(downsample)
          },
          "text_config": {
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "rope_theta": 10000,
            "tie_word_embeddings": true
          }
        }
        """
    }

    static func decode(_ json: String) throws -> SharpAIMossTranscribeDiarizeConfiguration {
        try JSONDecoder().decode(
            SharpAIMossTranscribeDiarizeConfiguration.self, from: Data(json.utf8))
    }
}
