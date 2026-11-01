// SharpAI fork overlay: moss_audio checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keys keep their vLLM/HF names. Rotary buffers are dropped, as is `audio_encoder.embed_positions`:
// the encoder builds its sinusoidal table itself (vLLM's load mapping drops that key too).

import Foundation
import MLX

public enum SharpAIMossAudioSanitize {

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter {
            !$0.key.contains("rotary_emb.inv_freq")
                && !$0.key.hasPrefix("audio_encoder.embed_positions")
        }
    }
}
