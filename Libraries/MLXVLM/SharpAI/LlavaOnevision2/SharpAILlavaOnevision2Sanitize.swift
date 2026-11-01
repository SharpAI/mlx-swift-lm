// SharpAI fork overlay: llava_onevision2 checkpoint key sanitization. Not part of upstream.
//
// Checkpoint layout (lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct, model.safetensors.index.json):
// - model.language_model.* -> model.* (Qwen3 text decoder)
// - model.visual.*         -> kept as-is (onevision_encoder tower and merger)
// - lm_head.weight         -> kept as-is
// Two vision-side renames are needed for the Swift parameter paths:
// - the patch conv weight [out, C, p, p] is flattened to [out, C * p * p];
// - the merger's `mlp.0` and `mlp.2` linears become `fc1` and `fc2`.
// Keys under `vision_tower.` / `multi_modal_projector.` are not in this checkpoint and are dropped
// if a different export uses them.

import Foundation
import MLX

public enum SharpAILlavaOnevision2Sanitizer {

    static let languagePrefixes: [(from: String, to: String)] = [
        (from: "model.language_model.", to: "model."),
        (from: "language_model.model.", to: "model."),
        (from: "language_model.lm_head.", to: "lm_head."),
    ]

    static let visionRenames: [(from: String, to: String)] = [
        (from: "model.visual.merger.mlp.0.", to: "model.visual.merger.fc1."),
        (from: "model.visual.merger.mlp.2.", to: "model.visual.merger.fc2."),
    ]

    static let patchEmbeddingKey = "model.visual.embeddings.patch_embedding.weight"

    static let droppedPrefixes = [
        "model.vision_tower.",
        "vision_tower.",
        "model.multi_modal_projector.",
        "multi_modal_projector.",
    ]

    /// Drops non-checkpoint vision names, maps language keys onto the Qwen3 names, and renames
    /// the vision merger and patch-embedding weights to the Swift parameter paths.
    public static func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") {
                continue
            }
            if droppedPrefixes.contains(where: { key.hasPrefix($0) }) {
                continue
            }

            var newKey = key
            for mapping in languagePrefixes where newKey.hasPrefix(mapping.from) {
                newKey = mapping.to + String(newKey.dropFirst(mapping.from.count))
                break
            }
            for mapping in visionRenames where newKey.hasPrefix(mapping.from) {
                newKey = mapping.to + String(newKey.dropFirst(mapping.from.count))
                break
            }

            var tensor = value
            if newKey == patchEmbeddingKey, tensor.ndim == 4 {
                tensor = tensor.reshaped(tensor.dim(0), -1)
            }

            guard sanitized[newKey] == nil else {
                throw SharpAILlavaOnevision2Error.duplicateWeightKey(newKey)
            }
            sanitized[newKey] = tensor
        }

        return sanitized
    }
}
