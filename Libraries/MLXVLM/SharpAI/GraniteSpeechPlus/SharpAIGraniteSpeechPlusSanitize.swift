// SharpAI fork overlay: granite_speech_plus checkpoint key and layout sanitization. Not part of
// upstream mlx-swift-lm.
//
// Checkpoint layout (ibm-granite/granite-speech-4.1-2b-plus, model.safetensors.index.json): the text
// stack is `language_model.model.model.*` (362 keys, tied head, so no `lm_head`), while `encoder.*`
// and `projector.*` sit at the top level. The transformers layout uses `model.language_model.*`,
// `model.encoder.*` and `model.projector.*`, so those prefixes are mapped too. The Swift module tree
// is `language_model.model.*`, `language_model.lm_head`, `encoder.*` and `projector.*`. Keys already in
// the Swift form are left unchanged.
// Conv weights change from PyTorch layout (out, in / groups, k) to MLX layout (out, k, in / groups).
// Verified against: https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite_speech/modeling_granite_speech.py
// Key names were read from the index through a summarizing fetch, not as raw bytes. Re-check them
// against the downloaded file before relying on the mapping.

import Foundation
import MLX

public enum SharpAIGraniteSpeechPlusSanitizer {

    /// HF prefix to Swift prefix. Order matters only for keys that share a prefix.
    static let prefixMap: [(from: String, to: String)] = [
        (from: "language_model.model.model.", to: "language_model.model."),
        (from: "model.language_model.", to: "language_model.model."),
        (from: "lm_head.", to: "language_model.lm_head."),
        (from: "model.encoder.", to: "encoder."),
        (from: "model.projector.", to: "projector."),
    ]

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights where !shouldDrop(key) {
            let renamed = remapKey(key)
            sanitized[renamed] = transposeConvWeight(renamed, value)
        }
        return sanitized
    }

    /// Drops buffers the Swift model computes itself: RoPE frequencies, the non-persistent Shaw
    /// distance table, and BatchNorm counters.
    static func shouldDrop(_ key: String) -> Bool {
        key.contains("rotary_emb.inv_freq")
            || key.hasSuffix("attention_dists")
            || key.hasSuffix("num_batches_tracked")
    }

    static func remapKey(_ key: String) -> String {
        for mapping in prefixMap where key.hasPrefix(mapping.from) {
            return mapping.to + String(key.dropFirst(mapping.from.count))
        }
        return key
    }

    /// PyTorch conv weights are (out, in / groups, k). MLX wants (out, k, in / groups).
    static func transposeConvWeight(_ key: String, _ value: MLXArray) -> MLXArray {
        guard key.hasSuffix(".weight"), value.ndim == 3 else {
            return value
        }
        return value.transposed(0, 2, 1)
    }
}
