// SharpAI fork overlay: audioflamingo3 key sanitization. Not part of upstream mlx-swift-lm.

import Foundation
import MLX
import MLXLMCommon

/// Maps checkpoint names onto `SharpAIAudioflamingo3` module paths.
public enum SharpAIAudioflamingo3Sanitize {

    public static func sanitize(
        weights: [String: MLXArray], config: SharpAIAudioflamingo3Configuration
    ) -> [String: MLXArray] {
        var renamed: [String: MLXArray] = [:]
        renamed.reserveCapacity(weights.count)
        for (key, value) in weights {
            renamed[checkpointName(key)] = value
        }

        var result = filterLMHeadWeights(
            from: renamed, tiedWordEmbeddings: config.textConfiguration.tieWordEmbeddings)
        result = result.filter { !$0.key.contains("rotary_emb.inv_freq") }

        // PyTorch conv weights are [out, in, kernel]. MLX expects [out, kernel, in].
        let convInputChannels: [(name: String, inChannels: Int)] = [
            ("audio_tower.conv1.weight", config.audioConfiguration.numMelBins),
            ("audio_tower.conv2.weight", config.audioConfiguration.hiddenSize),
        ]
        for (name, inChannels) in convInputChannels {
            guard let weight = result[name], weight.ndim == 3 else { continue }
            if weight.dim(1) == inChannels, weight.dim(2) != inChannels {
                result[name] = weight.transposed(0, 2, 1)
            }
        }
        return result
    }

    /// Renames transformers-style prefixes to this model's paths.
    /// UNVERIFIED: the `model.*` checkpoint layout is not on disk. It is mapped from the
    /// transformers v5 naming, which the vLLM reference does not use.
    static func checkpointName(_ key: String) -> String {
        let rules: [(from: String, to: String)] = [
            ("model.language_model.", "language_model.model."),
            ("model.audio_tower.", "audio_tower."),
            ("model.multi_modal_projector.", "multi_modal_projector."),
            ("lm_head.", "language_model.lm_head."),
        ]
        for rule in rules where key.hasPrefix(rule.from) {
            return rule.to + String(key.dropFirst(rule.from.count))
        }
        return key
    }
}
