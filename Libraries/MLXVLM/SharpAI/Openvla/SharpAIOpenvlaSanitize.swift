// SharpAI fork overlay: OpenVLA checkpoint key remap. Not part of upstream mlx-swift-lm.

import Foundation
import MLX

/// Maps OpenVLA checkpoint keys onto `SharpAIOpenvla` module paths.
public enum SharpAIOpenvlaSanitize {

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var renamed = [String: MLXArray]()
        renamed.reserveCapacity(weights.count)
        for (key, value) in weights {
            guard let name = mappedName(key) else { continue }
            renamed[name] = patchLayout(name, value)
        }
        return renamed
    }

    /// Module path for one checkpoint key, or nil to drop it.
    ///
    /// UNVERIFIED: the prefixes are not checked against a real OpenVLA file. Handled: an
    /// optional `model.` wrapper, prismatic's `llm_backbone.llm.` / `llm_backbone.` names, and
    /// the timm names inside each vision backbone.
    static func mappedName(_ key: String) -> String? {
        if key.contains("rotary_emb.inv_freq") {
            return nil
        }

        var body = key
        if body.hasPrefix("model.") {
            body = String(body.dropFirst("model.".count))
        }

        let llmPrefixes = ["llm_backbone.llm.", "llm_backbone."]
        if let prefix = llmPrefixes.first(where: { body.hasPrefix($0) }) {
            body = "language_model." + String(body.dropFirst(prefix.count))
        }

        // The final encoder norms and SigLIP's attention-pooling head are not used by the
        // penultimate-block features, so they are dropped.
        let unusedVisionParts = [
            "vision_backbone.featurizer.norm.",
            "vision_backbone.fused_featurizer.norm.",
            "vision_backbone.fused_featurizer.head.",
            "vision_backbone.fused_featurizer.attn_pool.",
        ]
        if unusedVisionParts.contains(where: { body.hasPrefix($0) }) {
            return nil
        }

        let roots = ["vision_backbone.", "projector.", "language_model."]
        if roots.contains(where: { body.hasPrefix($0) }) {
            return body
        }
        if body.hasPrefix("lm_head.") {
            return "language_model." + body
        }
        return key
    }

    /// Moves the patch conv weight from PyTorch `[out, in, kH, kW]` to MLX `[out, kH, kW, in]`.
    static func patchLayout(_ name: String, _ value: MLXArray) -> MLXArray {
        guard name.hasSuffix("patch_embed.proj.weight"), value.ndim == 4, !isMLXLayout(value)
        else {
            return value
        }
        return value.transposed(0, 2, 3, 1)
    }

    private static func isMLXLayout(_ array: MLXArray) -> Bool {
        let (outChannels, kH, kW) = (array.dim(0), array.dim(1), array.dim(2))
        return outChannels >= kH && outChannels >= kW && kH == kW
    }
}
