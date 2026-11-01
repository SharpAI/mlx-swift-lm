// SharpAI fork overlay: inkling_mm_model checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Checkpoint key names are kept (Verified against: https://huggingface.co/thinkingmachines/Inkling/resolve/main/model.safetensors.index.json).
// They are the names the model module tree is built with, so no renaming happens here.
//
// - `model.mtp.*` is dropped. transformers ignores it on load
//   (`_keys_to_ignore_on_load_unexpected = [r"model\.mtp\..*"]` in modeling_inkling.py), and the MTP head is not built.
// - Short-convolution weights are stored as `[channels, 1, kernel]` (the PyTorch depthwise `Conv1d`
//   layout, shapes from the safetensors headers). The model uses `[channels, kernel, 1]`, so those are
//   transposed here. Arrays already in the MLX layout (`shape[1] != 1`) are left as they are.
// - MoE experts are already stacked (`experts.w13_weight`, `experts.w2_weight`), and no stacking is needed.
//   The fused gate/up rows interleave (even rows gate, odd rows up), as transformers' conversion
//   (`Interleave` followed by `Chunk` in conversion_mapping.py) reads them. The model splits them at forward time.
//   Verified against: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/conversion_mapping.py
// - Vision and audio tensors are kept. Their modules are built in this overlay.
//
// Not done yet: ModelOpt NVFP4 expert tensors. The published base checkpoint is bf16, so the NVFP4
// packed codes and their scale tensors are not decoded.

import Foundation
import MLX

public enum SharpAIInklingMmModelSanitizer {

    public static let mtpPrefix = "model.mtp."

    /// Drops model.mtp.* keys and moves checkpoint short-conv weights to the MLX layout.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (key, value) in weights where !key.hasPrefix(mtpPrefix) {
            if key.hasSuffix("_sconv.weight"), value.ndim == 3, value.dim(1) == 1 {
                // [channels, 1, kernel] -> [channels, kernel, 1]
                out[key] = value.transposed(0, 2, 1)
            } else {
                out[key] = value
            }
        }
        return out
    }
}
