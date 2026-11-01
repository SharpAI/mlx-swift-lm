// SharpAI fork overlay: moss_transcribe_diarize checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon

/// Maps checkpoint names onto `SharpAIMossTranscribeDiarize` module paths.
public enum SharpAIMossTranscribeDiarizeSanitize {

    public static func sanitize(
        weights: [String: MLXArray], config: SharpAIMossTranscribeDiarizeConfiguration
    ) -> [String: MLXArray] {
        var renamed: [String: MLXArray] = [:]
        renamed.reserveCapacity(weights.count)
        for (key, value) in weights where !key.contains("rotary_emb.inv_freq") {
            renamed[checkpointName(key)] = value
        }

        var result = filterLMHeadWeights(
            from: renamed, tiedWordEmbeddings: config.textConfiguration.tieWordEmbeddings)

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

    /// Renames transformers-style prefixes to this model's paths. Keys already in this layout
    /// pass through unchanged.
    /// UNVERIFIED: the raw Hub key layout is not on disk. The rules below are guesses that
    /// follow the sibling SharpAI audio models.
    static func checkpointName(_ key: String) -> String {
        let rules: [(from: String, to: String)] = [
            ("model.audio_tower.", "audio_tower."),
            ("model.encoder.", "audio_tower."),
            ("model.multi_modal_projector.", "multi_modal_projector."),
            ("model.language_model.", "language_model.model."),
            ("lm_head.", "language_model.lm_head."),
            ("model.", "language_model.model."),
        ]
        for rule in rules where key.hasPrefix(rule.from) {
            return rule.to + String(key.dropFirst(rule.from.count))
        }
        return key
    }
}
