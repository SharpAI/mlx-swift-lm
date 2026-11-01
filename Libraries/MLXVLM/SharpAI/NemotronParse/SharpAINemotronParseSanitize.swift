// SharpAI fork overlay: nemotron_parse weight sanitize. Not part of upstream mlx-swift-lm.
//
// Key names are kept as the checkpoint writes them (HF `NemotronParseForConditionalGeneration`
// layout: `encoder.model_encoder.radio_model.*`, `encoder.conv1/conv2/sum_proj`, `decoder.*`,
// `lm_head.weight`). The names and shapes were read from the safetensors headers of
// nvidia/NVIDIA-Nemotron-Parse-v1.2 and nvidia/NVIDIA-Nemotron-Parse-v1.1. Changes:
//   - RADIO `input_conditioner.norm_mean` / `norm_std` buffers are dropped (the processor normalizes).
//   - `encoder.model_encoder.radio_model.summary_idxs` (I64 buffer) is dropped. The port reads the
//     indices from `encoder.args.teachers` in the configuration.
//   - `decoder.embed_positions.weight` is dropped. The reference decoder adds no positions, and vLLM
//     skips the same key (`if name.startswith("embed_positions"): continue`).
//   - `encoder.conv1.weight` ([out, in, 1]) and `encoder.conv2.weight` ([out, in, 1, 4]) are moved
//     from PyTorch layout to MLX channel-last layout.
//   - every other key passes through unchanged.

import MLX

public enum SharpAINemotronParseSanitize {

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        sanitized.reserveCapacity(weights.count)
        for (key, value) in weights {
            if isDroppedBuffer(key) {
                continue
            }
            if key == "encoder.conv1.weight", value.ndim == 3 {
                // [out, in, k] -> [out, k, in]
                sanitized[key] = value.transposed(0, 2, 1)
            } else if key == "encoder.conv2.weight", value.ndim == 4 {
                // [out, in, kH, kW] -> [out, kH, kW, in]
                sanitized[key] = value.transposed(0, 2, 3, 1)
            } else {
                sanitized[key] = value
            }
        }
        return sanitized
    }

    /// Buffers that are not module parameters in the port (see the header comment).
    static func isDroppedBuffer(_ key: String) -> Bool {
        key.hasSuffix("input_conditioner.norm_mean")
            || key.hasSuffix("input_conditioner.norm_std")
            || key.hasSuffix("radio_model.summary_idxs")
            || key.hasPrefix("decoder.embed_positions")
    }
}
