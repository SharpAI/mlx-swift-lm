// SharpAI fork overlay: isaac checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Output names are the parameter paths of SharpAIIsaacModel: `vision_tower.*`,
// `multimodal_projector.*`, `text_model.*` and `lm_head.weight`.
// The real Perceptron/Isaac-0.1 checkpoint (safetensors header, checked) uses
// `model.vision_tower.*`, `model.multimodal_projector.*`, `model.text_model.*` and
// `lm_head.weight`. The bare `model.embed_tokens/layers/norm` and `vision_model` spellings
// are accepted as well; their exact use in other checkpoints is not verified.
// The shipped Isaac-0.1 model.safetensors.index.json names sharded files that are not in the
// repo, so it was not used as a key source.

import Foundation
import MLX

public enum SharpAIIsaacSanitizer {

    /// (checkpoint prefix, parameter prefix), first match wins.
    static let renames: [(from: String, to: String)] = [
        ("model.vision_tower.", "vision_tower."),
        ("vision_tower.", "vision_tower."),
        ("model.vision_model.", "vision_tower."),
        ("vision_model.", "vision_tower."),
        ("model.multimodal_projector.", "multimodal_projector."),
        ("model.text_model.", "text_model."),
        ("model.embed_tokens.", "text_model.embed_tokens."),
        ("model.layers.", "text_model.layers."),
        ("model.norm.", "text_model.norm."),
    ]

    /// Returns the remapped key, or nil for keys the overlay never reads.
    static func remappedName(_ key: String) -> String? {
        if key.hasSuffix(".position_ids") || key.contains("rotary_emb.inv_freq") {
            return nil
        }
        for rename in renames where key.hasPrefix(rename.from) {
            return rename.to + String(key.dropFirst(rename.from.count))
        }
        return key
    }

    /// Applies ``remappedName(_:)`` to every key. Throws when two keys map to one name.
    public static func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(weights.count)
        for (key, value) in weights {
            guard let name = remappedName(key) else { continue }
            guard result[name] == nil else {
                throw SharpAIIsaacError.duplicateWeightKey(name)
            }
            result[name] = value
        }
        return result
    }
}
