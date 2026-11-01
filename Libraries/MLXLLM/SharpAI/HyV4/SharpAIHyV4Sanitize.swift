//
//  SharpAIHyV4Sanitize.swift
//  MLXLLM
//
//  Checkpoint key filter for `hy_v4`. Not part of upstream mlx-swift-lm.
//  Drops rotary buffers, the reference's `model.mtp_layers.*` (ignored on load), and
//  decoder layers at index >= num_hidden_layers (MTP). Expert key remapping lives in
//  SharpAIHyV4Model.sanitize.
//  UNVERIFIED: the key layout (model.layers.<i>.*) and the MTP layer index
//  (num_hidden_layers + d) follow the DeepSeek-style convention. No real checkpoint
//  index was read.
//

import Foundation
import MLX

enum SharpAIHyV4Sanitizer {

    static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIHyV4Configuration
    ) -> [String: MLXArray] {
        let numMain = configuration.hiddenLayers
        return weights.filter { key, _ in
            keep(key, numMain: numMain)
        }
    }

    static func keep(_ key: String, numMain: Int) -> Bool {
        if key.contains("rotary_emb.inv_freq") {
            return false
        }
        // The transformers reference ignores these keys on load (_keys_to_ignore_on_load_unexpected).
        if key.hasPrefix("model.mtp_layers.") {
            return false
        }
        // UNVERIFIED: MTP drop rule assumes MTP keys sit at index >= num_hidden_layers. No checkpoint index on disk confirms it.
        if let layer = layerIndex(key), layer >= numMain {
            return false
        }
        return true
    }

    static func layerIndex(_ key: String) -> Int? {
        let prefix = "model.layers."
        guard key.hasPrefix(prefix) else { return nil }
        let digits = key.dropFirst(prefix.count).prefix { $0.isNumber }
        return Int(digits)
    }
}
