// SharpAI fork overlay: kanana-1.5-v checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Drops rotary inv_freq and position_id keys, drops lm_head when embeddings are tied, and moves the
// Conv3d patch-embed weight from PyTorch [out, in, T, H, W] to MLX [out, T, H, W, in].
// UNVERIFIED: no real checkpoint was read, so the key prefixes are assumed.
// Projector keys are not remapped. A checkpoint that has them fails the noUnusedKeys check on load.

import Foundation
import MLX

public enum SharpAIKanana15VSanitizer {

    static let patchEmbedSuffix = "patch_embed.proj.weight"

    /// Returns nil for keys the model never reads.
    static func remappedName(
        _ key: String, configuration: SharpAIKanana15VConfiguration
    ) -> String? {
        if key.hasSuffix(".position_ids") || key.contains("rotary_emb.inv_freq") {
            return nil
        }
        if key == "language_model.lm_head.weight", configuration.tieWordEmbeddings {
            return nil
        }
        return key
    }

    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIKanana15VConfiguration
    ) throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(weights.count)
        let inChannels = configuration.visionConfiguration.inChannels

        for (key, value) in weights {
            guard let name = remappedName(key, configuration: configuration) else { continue }

            var tensor = value
            if name.hasSuffix(patchEmbedSuffix), tensor.ndim == 5, tensor.dim(1) == inChannels {
                tensor = tensor.transposed(0, 2, 3, 4, 1)
            }
            result[name] = tensor
        }
        return result
    }
}
