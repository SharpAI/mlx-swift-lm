// SharpAI fork overlay: moss_audio checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keys keep their vLLM/HF names. Rotary buffers are dropped, as is `audio_encoder.embed_positions`:
// the encoder builds its sinusoidal table itself (vLLM's load mapping drops that key too).
// The three audio-encoder Conv2d weights are PyTorch [O, I, kH, kW] and are moved to MLX [O, kH, kW, I].

import Foundation
import MLX

public enum SharpAIMossAudioSanitize {

    private static let convKeys: Set<String> = [
        "audio_encoder.conv1.weight", "audio_encoder.conv2.weight", "audio_encoder.conv3.weight",
    ]

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") || key.hasPrefix("audio_encoder.embed_positions") {
                continue
            }
            // PyTorch layout has a 3x3 kernel in the last two axes; MLX layout ends in the input channels.
            if convKeys.contains(key), value.ndim == 4, value.dim(2) == 3, value.dim(3) == 3 {
                result[key] = value.transposed(0, 2, 3, 1)
            } else {
                result[key] = value
            }
        }
        return result
    }
}
