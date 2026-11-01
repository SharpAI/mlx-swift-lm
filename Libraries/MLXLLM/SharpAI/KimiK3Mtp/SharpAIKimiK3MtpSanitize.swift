//
//  SharpAIKimiK3MtpSanitize.swift
//  MLXLLM
//
//  Key remapping for `kimi_k3_mtp` checkpoints. Not part of upstream.
//  The released index lists no MTP tensors (UNVERIFIED for upstream drafts), so the table is empty
//  and sanitize passes every key through unchanged until it is filled in.
//

import Foundation
import MLX

public enum SharpAIKimiK3MtpSanitize {

    /// HF checkpoint prefix -> module prefix. Empty until the MTP tensor names are known (UNVERIFIED).
    public static let keyMapping: [(from: String, to: String)] = []

    /// Maps one checkpoint key of MTP spec layer `specLayer` to its module key, following vLLM's
    /// `KimiK3MTP` loader (vllm/models/kimi_k3/nvidia/mtp.py, docs.vllm.ai):
    /// - skips `rotary_emb.inv_freq`, strips the leading `language_model.` prefix
    ///   (Verified against: https://docs.vllm.ai/en/latest/api/vllm/models/kimi_k3/nvidia/mtp/
    ///   load_weights summary; the enorm/hnorm/eh_proj/embed_tokens rules remain UNVERIFIED);
    /// - with `renamePackedWeights` (unpacked expert layout), `.weight_packed` becomes `.weight`;
    /// - keys that are not under `model.layers.{specLayer}.` return nil (not an MTP layer weight);
    /// - `embed_tokens` is hoisted to `model.` (shared embedding);
    /// - `enorm`, `hnorm`, `eh_proj`, `shared_head` stay under `model.layers.{specLayer}.`;
    /// - every other weight goes under `model.layers.{specLayer}.mtp_block.`.
    /// The spec layer index is supplied by the caller: vLLM's `get_spec_layer_idx_from_weight_name`
    /// was not available, so it is not reproduced here (UNVERIFIED for real checkpoints).
    public static func specLayerKey(
        _ checkpointKey: String, specLayer: Int, renamePackedWeights: Bool = false
    ) -> String? {
        if checkpointKey.contains("rotary_emb.inv_freq") { return nil }
        var name = checkpointKey
        let languagePrefix = "language_model."
        if name.hasPrefix(languagePrefix) {
            name = String(name.dropFirst(languagePrefix.count))
        }
        if renamePackedWeights {
            name = name.replacingOccurrences(of: ".weight_packed", with: ".weight")
        }
        let layerPrefix = "model.layers.\(specLayer)."
        guard name.hasPrefix(layerPrefix) else { return nil }
        let rest = String(name.dropFirst(layerPrefix.count))
        if rest.hasPrefix("embed_tokens") {
            return "model." + rest
        }
        for marker in ["enorm", "hnorm", "eh_proj", "shared_head"] where rest.contains(marker) {
            return layerPrefix + rest
        }
        return layerPrefix + "mtp_block." + rest
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
}
