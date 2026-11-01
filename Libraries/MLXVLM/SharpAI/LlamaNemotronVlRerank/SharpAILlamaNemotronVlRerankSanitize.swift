// SharpAI fork overlay: llama_nemotron_vl_rerank weight sanitize. Not part of upstream mlx-swift-lm.
//
// Checked against the safetensors header of nvidia/llama-nemotron-rerank-vl-1b-v2. Checkpoint keys
// already match the module paths of SharpAILlamaNemotronVlRerankModel, except for three cases:
//   - `model.mlp1.0/1/3.*` (nn.Sequential indices) map to `norm`, `fc1`, `fc2`.
//   - The patch-embedding conv is PyTorch [out, in, kH, kW] and is transposed to MLX [out, kH, kW, in].
//   - `head.*` (the unused SigLIP attention-pooling head) and buffers are dropped.

import Foundation
import MLX

public enum SharpAILlamaNemotronVlRerankSanitize {

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
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

    private static let projectorRenames: [(from: String, to: String)] = [
        ("model.mlp1.0.", "model.mlp1.norm."),
        ("model.mlp1.1.", "model.mlp1.fc1."),
        ("model.mlp1.3.", "model.mlp1.fc2."),
    ]

    /// Module path for one checkpoint key, or nil to drop it.
    static func mappedName(_ key: String) -> String? {
        if key.contains("rotary_emb.inv_freq") || key.contains("position_ids")
            || key.hasPrefix("lm_head.")
        {
            return nil
        }
        // The reference reads last_hidden_state and never runs this attention-pooling head.
        if key.hasPrefix("model.vision_model.vision_model.head.") {
            return nil
        }
        for (from, to) in projectorRenames where key.hasPrefix(from) {
            return to + String(key.dropFirst(from.count))
        }
        return key
    }

    /// PyTorch conv layout is `[out, in, kH, kW]`, so dim 1 (input channels) is smaller than
    /// dim 3 (kernel width). MLX layout is `[out, kH, kW, in]`.
    static func isTorchConvLayout(_ array: MLXArray) -> Bool {
        array.dim(1) < array.dim(3)
    }
}
