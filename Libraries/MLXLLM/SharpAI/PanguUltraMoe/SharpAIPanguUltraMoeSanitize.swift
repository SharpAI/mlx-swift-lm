// SharpAI fork overlay: pangu_ultra_moe checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keys keep their HF names, except that routed experts (`mlp.experts.N.*`) are stacked into
// `mlp.switch_mlp.*` for SwitchGLU. Dropped: rotary buffers, the tied LM head, and the multi-token-prediction layers. The
// published checkpoint stores the MTP layer as model.layers.<num_hidden_layers> (index 61 of
// 62 shards), and modeling_openpangu_moe.py never runs it.

import Foundation
import MLX

public enum SharpAIPanguUltraMoeSanitizer {

    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIPanguUltraMoeConfiguration
    ) -> [String: MLXArray] {
        var sanitized = weights.filter { key, _ in
            !key.contains("rotary_emb.inv_freq")
                && !isMultiTokenPredictionKey(key, hiddenLayers: configuration.hiddenLayers)
        }

        if configuration.tieWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
        }

        if let numExperts = configuration.routedExperts {
            for l in 0 ..< configuration.hiddenLayers {
                let prefix = "model.layers.\(l).mlp"
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    for k in ["weight", "scales", "biases"] {
                        guard sanitized["\(prefix).experts.0.\(proj).\(k)"] != nil else {
                            continue
                        }
                        // Missing experts are skipped here. A short checkpoint fails at load time.
                        let toJoin = (0 ..< numExperts).compactMap { e in
                            sanitized.removeValue(forKey: "\(prefix).experts.\(e).\(proj).\(k)")
                        }
                        sanitized["\(prefix).switch_mlp.\(proj).\(k)"] = MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitized
    }

    /// True for keys under `model.layers.<i>` with i at or past the main stack.
    static func isMultiTokenPredictionKey(_ key: String, hiddenLayers: Int) -> Bool {
        let parts = key.split(separator: ".")
        guard parts.count >= 3, parts[0] == "model", parts[1] == "layers",
            let layer = Int(parts[2])
        else {
            return false
        }
        return layer >= hiddenLayers
    }
}
