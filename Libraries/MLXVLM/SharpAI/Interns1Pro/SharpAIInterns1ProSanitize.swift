// SharpAI fork overlay: interns1_pro checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Verified against: https://huggingface.co/internlm/Intern-S1-Pro/resolve/main/model.safetensors.index.json
// - Checkpoint names are model.language_model.layers.N.{self_attn,mlp,...}, plus
//   model.language_model.rotary_emb.{sin_coef,cos_coef}, model.language_model.embed_tokens.weight,
//   model.language_model.norm.weight and lm_head.weight. No inv_freq tensor is stored.
// - Routed experts are per-expert tensors: mlp.experts.E.{gate,up,down}_proj.{weight,weight_scale_inv}.
// - FP8 block scales are named <weight>.weight_scale_inv (block 128x128, config quantization_config).
//   Attention projections are FP8 too. The routers (mlp.gate.weight) and the norms are not.
//
// Mapping:
// - rotary_emb.inv_freq is dropped. The FoPE coefficients sin_coef and cos_coef are kept: they are
//   learned parameters, and the checkpoint stores them.
// - <name>.weight is multiplied by <name>.weight_scale_inv (tile-wise), and the scale key is removed.
// - When numExperts is given, mlp.experts.E.<proj>.weight is stacked into mlp.switch_mlp.<proj>.weight.

import Foundation
import MLX

public enum SharpAIInterns1ProSanitizer {

    /// FP8 weight block edge length, from quantization_config.weight_block_size [128, 128].
    public static let fp8BlockSize = 128

    /// Applies the mappings above. Expert stacking runs only when `numExperts` is given, so a
    /// caller that has no config still gets per-expert keys back unchanged.
    public static func sanitize(
        _ weights: [String: MLXArray], numExperts: Int? = nil
    ) -> [String: MLXArray] {
        var result = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
        result = dequantizeBlockScales(result, blockSize: fp8BlockSize)
        if let numExperts {
            stackExperts(&result, numExperts: numExperts)
        }
        return result
    }

    /// Multiplies each `<name>.weight` by its `<name>.weight_scale_inv` tile scales and removes the
    /// scale key. A pair whose scale shape does not match the tile grid is left untouched, so the
    /// loader reports it instead of silently producing wrong values.
    public static func dequantizeBlockScales(
        _ weights: [String: MLXArray], blockSize: Int
    ) -> [String: MLXArray] {
        var result = weights
        let suffix = "_scale_inv"
        for (key, scaleInv) in weights where key.hasSuffix("weight" + suffix) {
            let weightKey = String(key.dropLast(suffix.count))
            guard let weight = weights[weightKey], weight.ndim == 2, scaleInv.ndim == 2 else {
                continue
            }
            let (m, n) = (weight.dim(0), weight.dim(1))
            let rowTiles = (m + blockSize - 1) / blockSize
            let colTiles = (n + blockSize - 1) / blockSize
            guard scaleInv.dim(0) == rowTiles, scaleInv.dim(1) == colTiles else { continue }

            let padBottom = rowTiles * blockSize - m
            let padSide = colTiles * blockSize - n
            var padded = weight
            if padBottom > 0 || padSide > 0 {
                padded = MLX.padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            }
            let tiled = padded.reshaped([rowTiles, blockSize, colTiles, blockSize])
            let scaled = tiled * scaleInv[0..., .newAxis, 0..., .newAxis]
            result[weightKey] = scaled.reshaped([rowTiles * blockSize, colTiles * blockSize])[0 ..< m, 0 ..< n]
            result[key] = nil
        }
        return result
    }

    /// Stacks `<prefix>.mlp.experts.0 ... N-1 .<proj>.weight` into `<prefix>.mlp.switch_mlp.<proj>.weight`.
    /// A group is stacked only when every expert index is present.
    static func stackExperts(_ weights: inout [String: MLXArray], numExperts: Int) {
        let prefixes = Set(
            weights.keys.compactMap { key -> String? in
                guard let range = key.range(of: ".mlp.experts.0.") else { return nil }
                return String(key[..<range.lowerBound])
            })
        for prefix in prefixes.sorted() {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                let expertKeys = (0 ..< numExperts).map { "\(prefix).mlp.experts.\($0).\(proj).weight" }
                guard expertKeys.allSatisfy({ weights[$0] != nil }) else { continue }
                let joined = expertKeys.map { weights[$0]! }
                weights["\(prefix).mlp.switch_mlp.\(proj).weight"] = MLX.stacked(joined)
                for key in expertKeys {
                    weights[key] = nil
                }
            }
        }
    }
}
