// SharpAI fork overlay: kolibri1 checkpoint key mapping. Not part of upstream mlx-swift-lm.
//
// Handles three checkpoint layouts:
// - Fused experts (the transformers layout): mlp.experts.gate_up_proj [E, 2I, H] and
//   mlp.experts.down_proj [E, H, I]. gate_up_proj is split in half along axis 1, with the
//   gate half first, into switch_mlp.gate_proj and switch_mlp.up_proj.
// - Per-expert keys (mlp.experts.E.<proj>.<suffix>): stacked into switch_mlp.<proj>. A group
//   is stacked only when every expert index is present.
// - FP8 block scales (<name>.weight with <name>.weight_scale_inv, block size 128): the weight
//   is multiplied by its scale block. The scale key is removed.
//
// Not covered: decoding raw float8_e4m3 bytes into floats. The dequant step expects a weight
// that is already a float array (see FP8Linear.swift for the raw-byte decode used elsewhere).

import Foundation
import MLX

public enum SharpAIKolibri1Sanitizer {

    public static let fp8BlockSize = 128

    /// Applies the mappings above to a checkpoint dictionary.
    public static func sanitize(
        weights: [String: MLXArray], config: SharpAIKolibri1Configuration
    ) -> [String: MLXArray] {
        var newWeights = dequantizeBlockScales(weights, blockSize: fp8BlockSize)

        for l in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            splitFusedExperts(in: &newWeights, prefix: prefix)
            for proj in ["gate_proj", "down_proj", "up_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let expertKeys = (0 ..< config.numExperts).map {
                        "\(prefix).mlp.experts.\($0).\(proj).\(suffix)"
                    }
                    guard expertKeys.allSatisfy({ newWeights[$0] != nil }) else { continue }
                    let joined = expertKeys.map { newWeights[$0]! }
                    newWeights["\(prefix).mlp.switch_mlp.\(proj).\(suffix)"] = stacked(joined)
                    for key in expertKeys {
                        newWeights[key] = nil
                    }
                }
            }
        }

        return newWeights
    }

    /// Splits fused expert tensors for one layer into the switch_mlp layout.
    /// The fused tensors are bare parameters in the checkpoint (`mlp.experts.gate_up_proj`,
    /// no `.weight` suffix), matching the transformers layout.
    static func splitFusedExperts(in weights: inout [String: MLXArray], prefix: String) {
        let fusedKey = "\(prefix).mlp.experts.gate_up_proj"
        if let fused = weights[fusedKey], fused.ndim == 3 {
            let half = fused.dim(1) / 2
            weights["\(prefix).mlp.switch_mlp.gate_proj.weight"] = fused[0..., ..<half, 0...]
            weights["\(prefix).mlp.switch_mlp.up_proj.weight"] = fused[0..., half..., 0...]
            weights[fusedKey] = nil
        }
        let downKey = "\(prefix).mlp.experts.down_proj"
        if let down = weights[downKey] {
            weights["\(prefix).mlp.switch_mlp.down_proj.weight"] = down
            weights[downKey] = nil
        }
    }

    /// Dequantizes `<name>.weight` with `<name>.weight_scale_inv` (one scale per
    /// blockSize x blockSize tile). Pairs with no weight are left alone.
    public static func dequantizeBlockScales(
        _ weights: [String: MLXArray], blockSize: Int
    ) -> [String: MLXArray] {
        var newWeights = weights

        for (key, scaleInv) in weights where key.hasSuffix("weight_scale_inv") {
            let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
            guard let weight = weights[weightKey] else { continue }

            let (m, n) = (weight.dim(0), weight.dim(1))
            let padBottom = (blockSize - m % blockSize) % blockSize
            let padSide = (blockSize - n % blockSize) % blockSize

            var paddedWeight = MLX.padded(
                weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            paddedWeight = paddedWeight.reshaped(
                [(m + padBottom) / blockSize, blockSize, (n + padSide) / blockSize, blockSize])
            let scaled = paddedWeight * scaleInv[0..., .newAxis, 0..., .newAxis]
            newWeights[weightKey] = scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
            newWeights[key] = nil
        }

        return newWeights
    }
}
