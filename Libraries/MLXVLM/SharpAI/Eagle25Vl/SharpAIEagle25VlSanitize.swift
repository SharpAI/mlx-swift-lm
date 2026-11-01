// SharpAI fork overlay: eagle_2_5_vl checkpoint key sanitization. Not part of upstream.
//
// Keys come from vLLM's module tree. The projector `mlp1` is a Sequential, so its checkpoint
// indices 0, 1 and 3 map to the named modules norm, linear_1 and linear_2. Index 2 is GELU and
// has no weights.
// UNVERIFIED: no real checkpoint index was read, so the prefixes are not confirmed.

import Foundation
import MLX

public enum SharpAIEagle25VlSanitizer {

    static let visionPrefix = "vision_model.vision_model."
    static let encoderLayerPrefix = "vision_model.vision_model.encoder.layers."

    /// Remaps checkpoint keys and drops unused ones. Returns nil for keys the model never reads.
    static func remappedName(
        _ key: String, configuration: SharpAIEagle25VlConfiguration
    ) -> String? {
        if key.hasSuffix(".position_ids") || key.contains("rotary_emb.inv_freq") {
            return nil
        }
        if key.hasPrefix(visionPrefix + "head.") {
            return nil
        }
        if key.hasPrefix(visionPrefix + "post_layernorm."),
            !configuration.appliesVisionPostLayerNorm
        {
            return nil
        }
        if key.hasPrefix(encoderLayerPrefix) {
            let rest = String(key.dropFirst(encoderLayerPrefix.count))
            let indexText = String(rest.prefix(while: { $0 != "." }))
            if let index = Int(indexText), index >= configuration.visionLayerCount {
                return nil
            }
        }
        if key.hasPrefix("language_model.lm_head."),
            configuration.textConfiguration.tieWordEmbeddings
        {
            return nil
        }
        if key.hasPrefix("mlp1.0.") {
            return "mlp1.norm." + String(key.dropFirst("mlp1.0.".count))
        }
        if key.hasPrefix("mlp1.1.") {
            return "mlp1.linear_1." + String(key.dropFirst("mlp1.1.".count))
        }
        if key.hasPrefix("mlp1.3.") {
            return "mlp1.linear_2." + String(key.dropFirst("mlp1.3.".count))
        }
        return key
    }

    /// Applies ``remappedName(_:configuration:)`` to every key and fixes the patch-embedding layout.
    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIEagle25VlConfiguration
    ) throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(weights.count)
        let channels = configuration.visionConfiguration.channels
        let patchKey = visionPrefix + "embeddings.patch_embedding.weight"

        for (key, value) in weights {
            guard let name = remappedName(key, configuration: configuration) else { continue }

            var tensor = value
            // PyTorch Conv2d is [out, in, kH, kW]. MLX wants [out, kH, kW, in].
            if name == patchKey, tensor.ndim == 4, tensor.dim(1) == channels {
                tensor = tensor.transposed(0, 2, 3, 1)
            }

            guard result[name] == nil else {
                throw SharpAIEagle25VlError.duplicateWeightKey(name)
            }
            result[name] = tensor
        }
        return result
    }
}
