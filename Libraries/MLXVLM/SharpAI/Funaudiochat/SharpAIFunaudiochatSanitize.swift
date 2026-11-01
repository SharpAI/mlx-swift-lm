// SharpAI fork overlay: funaudiochat key sanitization. Not part of upstream mlx-swift-lm.

import Foundation
import MLX

/// Checkpoint key sanitization for `funaudiochat`.
/// Verified against: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/funaudiochat.py
/// (`hf_to_vllm_mapper`: `orig_to_new_prefix={"audio_invert_tower.": None}`, so those weights are dropped on load).
/// Dropped here: rotary buffers, and every `audio_invert_tower.` key (the CRQ / invert tower).
/// UNVERIFIED: remapping of `language_model.*` (checkpoint) to the Qwen3 module names, and of the
/// `continuous_audio_tower.*` and `audio_tower.*` (discrete encoder) keys, is not written yet.
public enum SharpAIFunaudiochatSanitize {

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter {
            !$0.key.contains("rotary_emb.inv_freq") && !$0.key.hasPrefix("audio_invert_tower.")
        }
    }
}
