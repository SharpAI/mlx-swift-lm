// SharpAI fork overlay: LlamaNemotronVL checkpoint key remap. Not part of upstream mlx-swift-lm.

import Foundation
import MLX

/// Maps LlamaNemotronVL checkpoint keys onto `SharpAILlamaNemotronVl` module paths.
public enum SharpAILlamaNemotronVlSanitize {

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        sanitized.reserveCapacity(weights.count)
        for (key, value) in weights {
            guard let name = mappedName(key) else { continue }
            if name.hasSuffix("patch_embedding.weight"), value.ndim == 4, isTorchConvLayout(value) {
                sanitized[name] = value.transposed(0, 2, 3, 1)
            } else {
                sanitized[name] = value
            }
        }
        return sanitized
    }

    /// Module path for one checkpoint key, or nil to drop it.
    ///
    /// Handled: `language_model.model.` (standard Llama export) is dropped to `language_model.`.
    /// `mlp1.0/1/3` map to `norm`, `fc1`, `fc2`. The SigLIP pooling head (`vision_model.vision_model.head.*`), the reranker/LM-head,
    /// RoPE caches, and C-RADIO normalization buffers are dropped.
    static func mappedName(_ key: String) -> String? {
        if key.contains("rotary_emb.inv_freq") || key.contains("position_ids")
            || key.contains("norm_mean") || key.contains("norm_std")
        {
            return nil
        }
        if key.hasPrefix("vision_model.vision_model.head.") {
            return nil
        }
        if key.hasPrefix("lm_head.") || key.hasPrefix("language_model.lm_head.") {
            return nil
        }

        var body = key
        let modelPrefix = "language_model.model."
        if body.hasPrefix(modelPrefix) {
            body = "language_model." + String(body.dropFirst(modelPrefix.count))
        }

        let projectorRenames = [
            "mlp1.0.": "mlp1.norm.",
            "mlp1.1.": "mlp1.fc1.",
            "mlp1.3.": "mlp1.fc2.",
        ]
        for (from, to) in projectorRenames where body.hasPrefix(from) {
            return to + String(body.dropFirst(from.count))
        }
        return body
    }

    /// PyTorch conv layout is `[out, in, kH, kW]`, so dim 1 (input channels) is smaller than
    /// dim 3 (kernel width). MLX layout is `[out, kH, kW, in]`.
    static func isTorchConvLayout(_ array: MLXArray) -> Bool {
        array.dim(1) < array.dim(3)
    }
}
