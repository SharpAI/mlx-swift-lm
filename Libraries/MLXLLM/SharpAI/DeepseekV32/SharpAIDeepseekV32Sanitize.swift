// SharpAI fork overlay: deepseek_v32 checkpoint sanitize with MTP retention. Not part of
// upstream mlx-swift-lm.
//
// Same rules as DeepseekV32Model.sanitize, plus: when `retainMTP` is set, the blocks at
// model.layers.<num_hidden_layers + d> are kept. Their embed_tokens and shared_head.head
// copies are dropped, because the main embedding and lm_head are reused.
// UNVERIFIED: whether real checkpoints ship those copies or rely on sharing.

import Foundation
import MLX

enum SharpAIDeepseekV32Sanitizer {

    static func sanitize(
        _ weights: [String: MLXArray], configuration: DeepseekV32Configuration, retainMTP: Bool
    ) -> [String: MLXArray] {
        let numMain = configuration.numHiddenLayers
        let mtpCount = retainMTP ? configuration.numNextnPredictLayers : 0

        var out = dequantizeScaleInv(weights)
        stackExperts(
            into: &out, layerCount: numMain + mtpCount,
            expertCount: configuration.nRoutedExperts ?? 1)
        return out.filter { key, _ in keep(key, numMain: numMain, mtpCount: mtpCount) }
    }

    private static func keep(_ key: String, numMain: Int, mtpCount: Int) -> Bool {
        if key.contains("rotary_emb.inv_freq") || key.contains(".attn.compressor.")
            || key.contains("weight_scale_inv")
        {
            return false
        }
        if isPerExpertKey(key) {
            return false
        }
        if let layer = layerIndex(key), layer >= numMain {
            guard layer < numMain + mtpCount else { return false }
            if key.contains(".embed_tokens.") || key.contains(".shared_head.head.") {
                return false
            }
        }
        return true
    }

    /// `model.layers.N` → N. Nil for keys outside the layer stack.
    private static func layerIndex(_ key: String) -> Int? {
        guard key.hasPrefix("model.layers.") else { return nil }
        let parts = key.split(separator: ".")
        guard parts.count >= 3 else { return nil }
        return Int(parts[2])
    }

    /// `….mlp.experts.<int>.…` — consumed by stackExperts, which has no module to load into.
    private static func isPerExpertKey(_ key: String) -> Bool {
        let parts = key.split(separator: ".")
        guard parts.count >= 3 else { return false }
        for i in 0 ..< (parts.count - 2)
        where parts[i] == "mlp" && parts[i + 1] == "experts" && Int(parts[i + 2]) != nil {
            return true
        }
        return false
    }

    private static func dequantizeScaleInv(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = weights
        for (key, scaleInv) in weights where key.contains("weight_scale_inv") {
            let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
            if let weight = weights[weightKey] {
                out[weightKey] = dequant(weight: weight, scaleInv: scaleInv)
            }
        }
        return out
    }

    /// 128x128 block-wise FP8 dequantization.
    private static func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
        let bs = 128
        let (m, n) = (weight.dim(0), weight.dim(1))
        let padBottom = (bs - m % bs) % bs
        let padSide = (bs - n % bs) % bs

        let paddedWeight = padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
        let blocks = paddedWeight.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
        let scaled = blocks * scaleInv[0..., .newAxis, 0..., .newAxis]
        return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
    }

    /// Stacks per-expert weights into `switch_mlp`, using the HF names for projections.
    private static func stackExperts(
        into out: inout [String: MLXArray], layerCount: Int, expertCount: Int
    ) {
        for layer in 0 ..< layerCount {
            let prefix = "model.layers.\(layer)"
            for proj in ["gate_proj", "down_proj", "up_proj"] {
                for kind in ["weight", "scales", "biases"] {
                    guard out["\(prefix).mlp.experts.0.\(proj).\(kind)"] != nil else { continue }
                    let joined = (0 ..< expertCount).compactMap {
                        out["\(prefix).mlp.experts.\($0).\(proj).\(kind)"]
                    }
                    guard joined.count == expertCount else { continue }
                    out["\(prefix).mlp.switch_mlp.\(proj).\(kind)"] = stacked(joined)
                }
            }
        }
    }
}
