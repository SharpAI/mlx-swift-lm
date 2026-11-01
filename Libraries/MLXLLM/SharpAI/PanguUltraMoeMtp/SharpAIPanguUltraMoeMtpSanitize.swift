// SharpAI fork overlay: pangu_ultra_moe_mtp checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Layer-indexed rules follow vLLM's openpangu_mtp.py `_rewrite_spec_layer_name` and `get_spec_layer`:
// keys `model.layers.{N}.*` with N in [num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)
// are MTP keys. `embed_tokens` is shared with the main model and moves to `model.embed_tokens`.
// `enorm`, `hnorm`, `eh_proj` and `shared_head` stay under the spec layer. All other tensors of
// that layer belong to the transformer block and move under `mtp_block`.
// The prefix table below is still empty: the layer-indexed rules are handled by `rewriteKey`.
// Verified against: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/openpangu_mtp.py
// (get_spec_layer and _rewrite_spec_layer_name, as quoted by the fetch; the layer-61 key names in
// the checkpoint index are not confirmed, see SharpAIPanguUltraMoeMtpModel.swift).

import Foundation
import MLX

public enum SharpAIPanguUltraMoeMtpSanitizer {

    /// Plain HF prefix -> module prefix rewrites. Empty: the MTP rules are layer-indexed, see `rewriteKey`.
    public static let keyMapping: [(from: String, to: String)] = []

    private static let layerPrefix = "model.layers."
    private static let specLayerKeywords = ["enorm.", "hnorm.", "eh_proj.", "shared_head."]

    /// Splits `model.layers.{N}.rest` into `(N, rest)`. Returns nil for other keys.
    static func parseLayerKey(_ key: String) -> (index: Int, rest: String)? {
        guard key.hasPrefix(layerPrefix) else { return nil }
        let after = key.dropFirst(layerPrefix.count)
        guard let dot = after.firstIndex(of: ".") else { return nil }
        guard let index = Int(after[..<dot]) else { return nil }
        return (index, String(after[after.index(after: dot)...]))
    }

    /// True when the key belongs to an MTP spec layer (layer index >= num_hidden_layers).
    public static func isMtpKey(
        _ key: String, numHiddenLayers: Int, numNextnPredictLayers: Int
    ) -> Bool {
        guard let parsed = parseLayerKey(key) else { return false }
        return parsed.index >= numHiddenLayers
            && parsed.index < numHiddenLayers + numNextnPredictLayers
    }

    /// Rewrites one checkpoint key to its module key. Non-MTP keys are returned unchanged.
    public static func rewriteKey(
        _ key: String, numHiddenLayers: Int, numNextnPredictLayers: Int
    ) -> String {
        guard isMtpKey(key, numHiddenLayers: numHiddenLayers,
            numNextnPredictLayers: numNextnPredictLayers),
            let parsed = parseLayerKey(key)
        else { return key }
        if parsed.rest.hasPrefix("embed_tokens.") {
            return "model." + parsed.rest
        }
        if specLayerKeywords.contains(where: { parsed.rest.hasPrefix($0) }) {
            return key
        }
        return "\(layerPrefix)\(parsed.index).mtp_block.\(parsed.rest)"
    }

    /// Applies the first matching prefix rewrite to each key. With an empty table this is a no-op.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        guard !keyMapping.isEmpty else { return weights }
        var out = [String: MLXArray]()
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            var newKey = key
            for (from, to) in keyMapping where newKey.hasPrefix(from) {
                newKey = to + String(newKey.dropFirst(from.count))
                break
            }
            out[newKey] = value
        }
        return out
    }

    /// Keeps only the MTP spec-layer tensors and rewrites them to the names the model uses.
    ///
    /// Main-model tensors are dropped, because this module holds only the MTP namespace. Rotary
    /// buffers are dropped. `shared_head.head` is dropped: the head is the main model's LM head,
    /// passed to `draftLogits` (vLLM's compute_logits does the same with `shared_head.head`).
    public static func sanitizeMTP(
        _ weights: [String: MLXArray], numHiddenLayers: Int, numNextnPredictLayers: Int
    ) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        for (key, value) in weights {
            guard !key.contains("rotary_emb.inv_freq"),
                isMtpKey(key, numHiddenLayers: numHiddenLayers,
                    numNextnPredictLayers: numNextnPredictLayers),
                !key.contains(".shared_head.head.")
            else { continue }
            let rewritten = rewriteKey(
                key, numHiddenLayers: numHiddenLayers,
                numNextnPredictLayers: numNextnPredictLayers)
            out[rewritten] = value
        }
        return out
    }
}
