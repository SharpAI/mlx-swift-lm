// SharpAI fork overlay: qianfan_ocr checkpoint key sanitization. Not part of upstream mlx-swift-lm.

import Foundation
import MLX
import MLXLMCommon

/// Maps checkpoint names onto `SharpAIQianfanOcr` module paths.
public enum SharpAIQianfanOcrSanitize {

    public static func sanitize(
        weights: [String: MLXArray], config: SharpAIQianfanOcrConfiguration
    ) -> [String: MLXArray] {
        var renamed: [String: MLXArray] = [:]
        renamed.reserveCapacity(weights.count)
        for (key, value) in weights where !key.contains("rotary_emb.inv_freq") {
            let name = checkpointName(key)
            if name.contains(".attn.qkv.") {
                // Fused qkv is stacked on dim 0 as [q; k; v].
                for (index, part) in ["q_proj", "k_proj", "v_proj"].enumerated() {
                    let target = name.replacingOccurrences(
                        of: ".attn.qkv.", with: ".attention.\(part).")
                    renamed[target] = third(value, index)
                }
            } else {
                renamed[name] = value
            }
        }

        var result = filterLMHeadWeights(
            from: renamed, tiedWordEmbeddings: config.textConfiguration.tieWordEmbeddings)

        // PyTorch conv weights are [out, in, kH, kW]. MLX expects [out, kH, kW, in].
        let patchKey = "vision_tower.embeddings.patch_embeddings.projection.weight"
        if let weight = result[patchKey], weight.ndim == 4,
            weight.dim(1) == config.visionConfiguration.numChannels
        {
            result[patchKey] = weight.transposed(0, 2, 3, 1)
        }
        return result
    }

    /// Rows `[index * n, (index + 1) * n)` of a stacked weight, with `n = dim(0) / 3`.
    static func third(_ value: MLXArray, _ index: Int) -> MLXArray {
        let size = value.dim(0) / 3
        return value[(index * size) ..< ((index + 1) * size)]
    }

    /// Renames the prefixes of the original Hub layout and of transformers' names to this model's paths.
    /// Hub keys: `vision_model.*`, `mlp1.*`, `language_model.model.*`, `language_model.lm_head.*`.
    /// UNVERIFIED: the raw Hub key layout is not on disk. Names follow transformers' conversion
    /// mapping for qianfan_ocr. Keys already in this layout pass through unchanged.
    static func checkpointName(_ key: String) -> String {
        let prefixRules: [(from: String, to: String)] = [
            ("model.vision_tower.", "vision_tower."),
            ("vision_model.", "vision_tower."),
            ("model.multi_modal_projector.", "multi_modal_projector."),
            ("mlp1.0.", "multi_modal_projector.layer_norm."),
            ("mlp1.1.", "multi_modal_projector.linear_1."),
            ("mlp1.3.", "multi_modal_projector.linear_2."),
            ("model.language_model.", "language_model.model."),
            ("lm_head.", "language_model.lm_head."),
        ]
        var name = key
        for rule in prefixRules where name.hasPrefix(rule.from) {
            name = rule.to + String(name.dropFirst(rule.from.count))
            break
        }
        guard name.hasPrefix("vision_tower.") else { return name }

        if name == "vision_tower.embeddings.class_embedding" {
            return "vision_tower.embeddings.cls_token"
        }
        if name == "vision_tower.embeddings.position_embedding" {
            return "vision_tower.embeddings.position_embeddings"
        }

        let innerRules: [(from: String, to: String)] = [
            ("vision_tower.encoder.layers.", "vision_tower.layers."),
            ("vision_tower.embeddings.patch_embedding.",
                "vision_tower.embeddings.patch_embeddings.projection."),
            (".attn.proj.", ".attention.projection_layer."),
            (".norm1.", ".layernorm_before."),
            (".norm2.", ".layernorm_after."),
        ]
        for rule in innerRules {
            name = name.replacingOccurrences(of: rule.from, with: rule.to)
        }
        if name.hasSuffix(".ls1") {
            name = String(name.dropLast(4)) + ".lambda_1"
        }
        if name.hasSuffix(".ls2") {
            name = String(name.dropLast(4)) + ".lambda_2"
        }
        return name
    }
}
