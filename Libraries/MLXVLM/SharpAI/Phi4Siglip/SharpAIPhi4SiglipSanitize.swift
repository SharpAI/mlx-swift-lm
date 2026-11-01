// SharpAI fork overlay: phi4-siglip checkpoint key sanitization. Not part of upstream.
//
// Module keys already match the checkpoint layout: `model.embed_tokens`, `model.layers`,
// `model.norm`, `model.vision_tower.vision_tower.vision_model.*`, `model.mm_projector.0|2`, and
// `lm_head`. This pass drops tensors the model never reads and fixes the patch projection layout.
// UNVERIFIED: no real checkpoint index was read, so the prefixes are not confirmed.

import Foundation
import MLX

public enum SharpAIPhi4SiglipSanitizer {

    // UNVERIFIED: checkpoint prefix is guessed; no weight index was read.
    static let visionPrefix = "model.vision_tower.vision_tower.vision_model."

    /// Returns the model key for one checkpoint key, or nil when the model never reads it.
    static func remappedName(_ key: String, configuration: SharpAIPhi4SiglipConfiguration) -> String?
    {
        if key.hasSuffix(".position_ids") || key.contains("rotary_emb.inv_freq") {
            return nil
        }
        // The SigLIP attention pooling head is not used for image features.
        if key.hasPrefix(visionPrefix + "head.") {
            return nil
        }
        if key.hasPrefix("lm_head."), configuration.tieWordEmbeddings {
            return nil
        }
        return key
    }

    /// Applies ``remappedName(_:configuration:)`` to every key. A Conv2d patch weight becomes the
    /// flat `[out, p * p * C]` Linear weight, with patch rows ordered `(row, col, channel)`.
    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIPhi4SiglipConfiguration
    ) throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(weights.count)
        let patchKey = visionPrefix + "embeddings.patch_embedding.weight"

        for (key, value) in weights {
            guard let name = remappedName(key, configuration: configuration) else { continue }

            var tensor = value
            if name == patchKey, tensor.ndim == 4 {
                let outputs = tensor.dim(0)
                tensor = tensor.transposed(0, 2, 3, 1).reshaped(outputs, -1)
            }

            guard result[name] == nil else {
                throw SharpAIPhi4SiglipError.duplicateWeightKey(name)
            }
            result[name] = tensor
        }
        return result
    }
}
