// SharpAI fork overlay: glm5_next checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keeps the HF checkpoint names. The remap here is limited to:
//   - fp8 block-wise dequant (weight_scale_inv, 128x128 blocks, from quantization_config)
//   - per-expert stacking into switch_mlp
//   - dropping rotary buffers, scale tensors and nextn (MTP) layers unless retainMTP is set
// UNVERIFIED: the expert key names (mlp.experts.<e>.<proj>) and the scale_inv suffix are
// assumed from the glm_moe_dsa / DeepSeek checkpoint layout. The fixture lists no expert keys.
// Vision keys (visual.*) pass through unchanged.

import Foundation
import MLX

public enum SharpAIGlm5NextSanitizer {

    public static func sanitize(
        _ weights: [String: MLXArray],
        numHiddenLayers: Int,
        numNextnPredictLayers: Int,
        expertCount: Int,
        retainMTP: Bool
    ) -> [String: MLXArray] {
        let mtpCount = retainMTP ? numNextnPredictLayers : 0
        var out = dequantizeScaleInv(weights)
        stackExperts(into: &out, layerCount: numHiddenLayers + mtpCount, expertCount: expertCount)
        return out.filter { key, _ in
            keep(key, numMain: numHiddenLayers, mtpCount: mtpCount)
        }
    }

    private static func keep(_ key: String, numMain: Int, mtpCount: Int) -> Bool {
        if key.contains("rotary_emb.inv_freq") || key.contains("weight_scale_inv") {
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

    /// `model.layers.N` gives N. Nil for keys outside the layer stack.
    private static func layerIndex(_ key: String) -> Int? {
        guard key.hasPrefix("model.layers.") else { return nil }
        let parts = key.split(separator: ".")
        guard parts.count >= 3 else { return nil }
        return Int(parts[2])
    }

    /// `....mlp.experts.<int>....`, consumed by stackExperts.
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

    /// Stacks per-expert weights into `switch_mlp`, keeping the HF projection names.
    private static func stackExperts(
        into out: inout [String: MLXArray], layerCount: Int, expertCount: Int
    ) {
        guard expertCount > 0 else { return }
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
