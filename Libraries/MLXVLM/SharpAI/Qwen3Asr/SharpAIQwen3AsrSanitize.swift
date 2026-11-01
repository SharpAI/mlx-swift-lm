// SharpAI fork overlay: qwen3_asr checkpoint key and layout sanitization. Not part of upstream
// mlx-swift-lm.
//
// Hub keys carry a "thinker." prefix. The Swift modules use the keys without it. Transformers
// keys (model.audio_tower, model.language_model, multi_modal_projector) are mapped too, so
// either export loads. Conv weights go from PyTorch (out, in, kH, kW) to MLX (out, kH, kW, in).
// UNVERIFIED: the hub key names come from vLLM's weight mapper (thinker.audio_tower.proj1 and
// proj2 for the projector). No checkpoint was read here.

import Foundation
import MLX

public enum SharpAIQwen3AsrSanitizer {

    /// Prefix rewrites, first match wins. Applied after the "thinker." prefix is removed.
    static let prefixRules: [(from: String, to: String)] = [
        ("model.audio_tower.", "audio_tower."),
        ("model.multi_modal_projector.linear_1.", "audio_tower.proj1."),
        ("model.multi_modal_projector.linear_2.", "audio_tower.proj2."),
        ("model.language_model.", "model."),
        ("language_model.model.", "model."),
        ("language_model.lm_head.", "lm_head."),
    ]

    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIQwen3AsrConfiguration
    ) throws -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            if shouldDrop(key) {
                continue
            }
            var renamed = remapKey(key)
            if configuration.forcedAligner {
                renamed = renamedForAligner(renamed)
            } else if configuration.tieWordEmbeddings && renamed.hasPrefix("lm_head.") {
                // The output projection reuses the token embedding.
                continue
            }
            sanitized[renamed] = transposeConvWeight(
                renamed, value,
                downsampleHiddenSize: configuration.audioConfiguration.downsampleHiddenSize)
        }

        return sanitized
    }

    /// Buffers that the Swift model computes inline.
    static func shouldDrop(_ key: String) -> Bool {
        key.contains("positional_embedding") || key.contains("rotary_emb.inv_freq")
    }

    static func remapKey(_ key: String) -> String {
        var name = key
        let thinker = "thinker."
        if name.hasPrefix(thinker) {
            name = String(name.dropFirst(thinker.count))
        }
        for rule in prefixRules where name.hasPrefix(rule.from) {
            return rule.to + String(name.dropFirst(rule.from.count))
        }
        return name
    }

    /// The forced aligner's `lm_head` is its classification head.
    static func renamedForAligner(_ key: String) -> String {
        let head = "lm_head."
        guard key.hasPrefix(head) else {
            return key
        }
        return "classifier." + String(key.dropFirst(head.count))
    }

    /// PyTorch conv weights are (out, in, kH, kW). MLX wants (out, kH, kW, in).
    /// A weight whose last dim already equals its input channels is in MLX layout. Skip it.
    static func transposeConvWeight(
        _ key: String, _ value: MLXArray, downsampleHiddenSize: Int
    ) -> MLXArray {
        guard key.hasSuffix(".weight"), key.contains("conv2d"), value.ndim == 4 else {
            return value
        }
        let inChannels = key.contains("conv2d1.") ? 1 : downsampleHiddenSize
        if value.shape[3] == inChannels {
            return value
        }
        return value.transposed(0, 2, 3, 1)
    }
}
