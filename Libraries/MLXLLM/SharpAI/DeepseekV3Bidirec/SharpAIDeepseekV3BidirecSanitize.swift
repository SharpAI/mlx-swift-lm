// SharpAI fork overlay: deepseek_v3_bidirec checkpoint key sanitization. Not part of upstream
// mlx-swift-lm.
//
// Key names are the DeepSeek-V3 HF names, so checkpoint keys load unchanged. The steps follow
// DeepseekV3Model.sanitize: drop rotary buffers, dequantize FP8 block scales, drop
// multi-token-prediction layers past num_hidden_layers, and stack per-expert weights into
// switch_mlp.
//
// UNVERIFIED: whether the checkpoint has MTP layers at index num_hidden_layers is not known.
// Those layers are dropped because the model has no module for them.

import Foundation
import MLX

enum SharpAIDeepseekV3BidirecError: LocalizedError {
    case missingExpertWeight(String)

    var errorDescription: String? {
        switch self {
        case .missingExpertWeight(let key):
            return "Missing expert weight during deepseek_v3_bidirec sanitize: \(key)"
        }
    }
}

enum SharpAIDeepseekV3BidirecSanitizer {

    static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIDeepseekV3BidirecConfiguration
    ) throws -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        for scaleKey in sanitized.keys.filter({ $0.hasSuffix(".weight_scale_inv") }) {
            guard let scaleInv = sanitized.removeValue(forKey: scaleKey) else { continue }
            let weightKey = String(scaleKey.dropLast("_scale_inv".count))
            guard let weight = sanitized[weightKey] else { continue }
            sanitized[weightKey] = dequantize(weight, scaleInv: scaleInv)
        }

        let layerCount = configuration.numHiddenLayers
        sanitized = sanitized.filter { key, _ in
            !isMultiTokenPredictionKey(key, layerCount: layerCount)
        }

        guard let numExperts = configuration.nRoutedExperts else {
            return sanitized
        }

        for layer in 0 ..< layerCount {
            let prefix = "model.layers.\(layer)"
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                for kind in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(proj).\(kind)"
                    guard sanitized[firstKey] != nil else { continue }

                    let toJoin = try (0 ..< numExperts).map { expert -> MLXArray in
                        let key = "\(prefix).mlp.experts.\(expert).\(proj).\(kind)"
                        guard let value = sanitized.removeValue(forKey: key) else {
                            throw SharpAIDeepseekV3BidirecError.missingExpertWeight(key)
                        }
                        return value
                    }
                    sanitized["\(prefix).mlp.switch_mlp.\(proj).\(kind)"] = stacked(toJoin)
                }
            }
        }

        return sanitized
    }

    /// Dequantizes an FP8 weight stored in 128x128 blocks with one inverse scale per block.
    private static func dequantize(_ weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
        let bs = 128
        let rows = weight.dim(0)
        let cols = weight.dim(1)
        let padBottom = (bs - rows % bs) % bs
        let padSide = (bs - cols % bs) % bs

        let paddedWeight = padded(
            weight, widths: [.init((0, padBottom)), .init((0, padSide))])
        let blocked = paddedWeight.reshaped([
            (rows + padBottom) / bs, bs, (cols + padSide) / bs, bs,
        ])
        let scaled = blocked * scaleInv[0..., .newAxis, 0..., .newAxis]
        return scaled.reshaped([rows + padBottom, cols + padSide])[0 ..< rows, 0 ..< cols]
    }

    /// Multi-token-prediction layers sit at `model.layers.<num_hidden_layers>` and beyond.
    private static func isMultiTokenPredictionKey(_ key: String, layerCount: Int) -> Bool {
        guard key.hasPrefix("model.layers.") else { return false }
        let parts = key.split(separator: ".")
        guard parts.count >= 3, let layerIdx = Int(parts[2]) else { return false }
        return layerIdx >= layerCount
    }
}
