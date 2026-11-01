// SharpAI fork overlay: checkpoint key sanitization for the "unknown" entry. Not part of upstream.
//
// EagleCohere: strips the "model." prefix so keys match SharpAIEagleCohereDraftModel, drops
// buffers and the LM head, and shifts layer indices by `layerOffset`.
// PanguProMoEV2: drops buffers and the tied LM head. Expert stacking is not handled (gap).
// UNVERIFIED: the layer offset and the expert key layout come from vLLM, not a checkpoint.

import Foundation
import MLX

public enum SharpAIUnknownSanitizer {

    public static func eagleCohere(
        _ weights: [String: MLXArray],
        configuration: SharpAIEagleCohereConfiguration,
        layerOffset: Int = 0
    ) -> [String: MLXArray] {
        let hiddenLayers = configuration.base.hiddenLayers
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") || key.hasPrefix("lm_head.") {
                continue
            }
            if configuration.tieWordEmbeddings && key.hasPrefix("model.embed_tokens.") {
                continue
            }

            var name = key
            if name.hasPrefix("model.") {
                name = String(name.dropFirst("model.".count))
            }

            if name.hasPrefix("layers.") {
                let rest = name.dropFirst("layers.".count)
                guard let dot = rest.firstIndex(of: "."),
                    let index = Int(rest[..<dot]),
                    index >= layerOffset, index - layerOffset < hiddenLayers
                else {
                    continue
                }
                name = "layers.\(index - layerOffset)" + String(rest[dot...])
            }

            sanitized[name] = value
        }

        return sanitized
    }

    public static func panguProMoEV2(
        _ weights: [String: MLXArray], configuration: SharpAIPanguProMoEV2Configuration
    ) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        if configuration.tieWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
        }

        return sanitized
    }
}
