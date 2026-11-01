// SharpAI fork overlay: ModernVBERT checkpoint key remap. Not part of upstream mlx-swift-lm.

import Foundation
import MLX

/// Maps ModernVBERT checkpoint keys onto `SharpAIModernvbert` module paths.
///
/// Both `ModernVBertModel` (no prefix) and `ColModernVBertForRetrieval` (`model.` prefix)
/// land on the same tree: `model.{vision_model,connector,text_model}.*` and
/// `embedding_proj_layer.*`.
public enum SharpAIModernvbertSanitize {

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var renamed = [String: MLXArray]()
        renamed.reserveCapacity(weights.count)
        for (key, value) in weights {
            guard let name = mappedName(key) else { continue }
            renamed[name] = value
        }
        return patchEmbeddingLayout(renamed)
    }

    /// Module path for one checkpoint key, or nil to drop it.
    ///
    /// UNVERIFIED: the prefixes are not checked against a real ModernVBERT file. Handled:
    /// an optional `model.` wrapper, a doubled `vision_model.vision_model.` wrapper, the
    /// masked-LM head, and the SigLIP pooling head.
    static func mappedName(_ key: String) -> String? {
        if key.contains("rotary_emb.inv_freq") || key.contains("position_id") {
            return nil
        }

        var body = key
        if body.hasPrefix("model.") {
            body = String(body.dropFirst("model.".count))
        }
        if body.hasPrefix("vision_model.vision_model.") {
            body = String(body.dropFirst("vision_model.".count))
        }

        let dropped = ["lm_head.", "decoder.", "head.", "vision_model.head."]
        if dropped.contains(where: { body.hasPrefix($0) }) {
            return nil
        }

        if body.hasPrefix("embedding_proj_layer.") {
            return body
        }
        if body.hasPrefix("custom_text_proj.") {
            return "embedding_proj_layer." + String(body.dropFirst("custom_text_proj.".count))
        }

        let backboneRoots = ["vision_model.", "connector.", "text_model."]
        guard backboneRoots.contains(where: { body.hasPrefix($0) }) else {
            return key
        }
        return "model." + body
    }

    /// Moves the SigLIP patch conv weight to MLX layout `[out, kH, kW, in]`.
    private static func patchEmbeddingLayout(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var fixed = [String: MLXArray]()
        fixed.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key.contains("patch_embedding.weight"), value.ndim == 4, !isMLXLayout(value) {
                fixed[key] = value.transposed(0, 2, 3, 1)
            } else {
                fixed[key] = value
            }
        }
        return fixed
    }

    private static func isMLXLayout(_ array: MLXArray) -> Bool {
        let (outChannels, kH, kW) = (array.dim(0), array.dim(1), array.dim(2))
        return outChannels >= kH && outChannels >= kW && kH == kW
    }
}
