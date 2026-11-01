// SharpAI fork overlay: step5_mtp checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Sources: vLLM Step5MTP.load_weights in vllm/models/step5/amd/mtp.py and the stacked-parameter
// mapping in vllm/models/step5/amd/model.py, both from vllm-project/vllm PR #61837 (unmerged, read
// through the PR diff). The mtp_block insertion, shared_head.output to shared_head.head rename,
// and dropping of tok_embeddings and rotary inv_freq follow that loader.
// Fused projections: q/k/v stack into qkv_proj and gate/up stack into gate_up_proj, in shard order.
// Expert weights are not stacked here. The ".bias" skip needs the built module's parameter names
// and is not done here.
// UNVERIFIED: whether real VL checkpoints use the "model.language_model.layers.N." prefix for MTP
// layers. specLayerIndex accepted it, so mappedKey normalises it to "model.layers.N." before the
// mtp_block rewrite; no source shows the real layout.

import Foundation
import MLX

public enum SharpAIStep5MtpSanitizer {

    static let specWeightNames = ["embed_tokens", "enorm", "hnorm", "eh_proj", "shared_head"]

    /// Maps checkpoint keys to MTP-module keys. Returns nil for keys the MTP module drops.
    public static func mappedKey(
        _ rawKey: String, configuration: SharpAIStep5MtpConfiguration
    ) -> String? {
        if rawKey.contains("rotary_emb.inv_freq") {
            return nil
        }
        var name = rawKey.hasPrefix("layers.") ? "model." + rawKey : rawKey
        let languageModelLayers = "model.language_model.layers."
        if name.hasPrefix(languageModelLayers) {
            name = "model.layers." + name.dropFirst(languageModelLayers.count)
        }
        let layer = specLayerIndex(of: name, configuration: configuration)
        if !name.contains("embed_tokens") && layer == nil {
            return nil
        }
        name = rewriteSpecLayerName(name, layer: layer)
        if name.contains("tok_embeddings") {
            return nil
        }
        if layer != nil {
            name = name.replacingOccurrences(of: ".transformer.", with: ".")
        }
        name = name.replacingOccurrences(of: "shared_head.output", with: "shared_head.head")
        if name.contains("embed_tokens") {
            return "model.embed_tokens.weight"
        }
        return name
    }

    /// Sanitizes checkpoint weights for the MTP module. Duplicate targets keep the first sorted key.
    /// Fused projections are stacked last, so the result matches the loaded parameter layout.
    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIStep5MtpConfiguration
    ) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for key in weights.keys.sorted() {
            guard let mapped = mappedKey(key, configuration: configuration),
                result[mapped] == nil,
                let value = weights[key]
            else {
                continue
            }
            result[mapped] = value
        }
        return stackFusedProjections(result)
    }

    /// (checkpoint parts in shard order, fused name), from vLLM's stacked_params_mapping.
    static let fusedGroups: [(parts: [String], fused: String)] = [
        (["q_proj", "k_proj", "v_proj"], "qkv_proj"),
        (["gate_proj", "up_proj"], "gate_up_proj"),
    ]

    /// Parent modules whose projections are stacked. Routed experts are never stacked here.
    static let stackableParents = ["self_attn.", "mlp.", "share_expert."]

    /// Concatenates q/k/v into qkv_proj and gate/up into gate_up_proj along the output axis.
    /// A group is stacked only when every part is present with the same suffix (weight or bias).
    public static func stackFusedProjections(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights
        var handled = Set<String>()
        for (parts, fused) in fusedGroups {
            guard let first = parts.first else { continue }
            let marker = "." + first + "."
            for key in weights.keys.sorted() {
                guard let range = key.range(of: marker) else { continue }
                let parent = String(key[..<range.lowerBound]) + "."
                let suffix = String(key[range.upperBound...])
                guard !parent.contains("experts."),
                    stackableParents.contains(where: { parent.hasSuffix("." + $0) }),
                    !handled.contains(parent + suffix)
                else { continue }
                let sources = parts.map { parent + $0 + "." + suffix }
                let arrays = sources.compactMap { weights[$0] }
                guard arrays.count == parts.count else { continue }
                sources.forEach { result[$0] = nil }
                result[parent + fused + "." + suffix] = concatenated(arrays, axis: 0)
                handled.insert(parent + suffix)
            }
        }
        return result
    }

    /// Returns the absolute layer index for an MTP-layer key, or nil.
    public static func specLayerIndex(
        of key: String, configuration: SharpAIStep5MtpConfiguration
    ) -> Int? {
        let count = configuration.nextnPredictLayers
        let base = configuration.hiddenLayers
        for offset in 0 ..< max(0, count) {
            let index = base + offset
            let prefixes = [
                "model.layers.\(index).",
                "layers.\(index).",
                "model.language_model.layers.\(index).",
            ]
            if prefixes.contains(where: { key.hasPrefix($0) }) {
                return index
            }
        }
        return nil
    }

    static func rewriteSpecLayerName(_ name: String, layer: Int?) -> String {
        guard let layer = layer else {
            return name
        }
        if specWeightNames.contains(where: { name.contains($0) }) {
            return name
        }
        return name.replacingOccurrences(
            of: "model.layers.\(layer).", with: "model.layers.\(layer).mtp_block.")
    }
}
