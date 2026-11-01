// SharpAI fork overlay: dots3_note_mtp checkpoint key mapping. Not part of upstream.
//
// Checkpoint layout (dots-studio/dots3-note-prev model.safetensors.index.json, verified):
//   model.layers.{N}.{enorm,hnorm,eh_proj}.weight        N = num_hidden_layers (46)
//   model.layers.{N}.shared_head.norm.weight
//   model.layers.{N}.{self_attn,mlp,input_layernorm,post_attention_layernorm}.*   -> decoder
//   model.mtp.embed_tokens.weight                         -> embed_tokens (MTP-specific)
// There is no `shared_head.head` tensor. The head is shared with the target lm_head, as sglang does.
// Matches sglang's NextN loader (dots3_common/modeling.py `load_weights`, is_nextn branch):
// `model.mtp.embed_tokens.weight` -> `embed_tokens`, `enorm/hnorm/eh_proj/shared_head.norm` at the
// head top level, every other layer tensor under the decoder.

import Foundation
import MLX

public enum SharpAIDots3NoteMtpSanitize {

    /// Checkpoint key of the MTP-specific input embedding.
    public static let mtpEmbeddingKey = "model.mtp.embed_tokens.weight"

    /// Head-level keys that sit outside the decoder layer, relative to the MTP layer prefix.
    public static let headTopLevelKeys: Set<String> = [
        "enorm.weight",
        "hnorm.weight",
        "eh_proj.weight",
        "shared_head.norm.weight",
    ]

    /// Maps one checkpoint key to its `SharpAIDots3NoteMtpHead` parameter key, or nil when the
    /// tensor is not owned by the MTP head (target-model weights, lower layers, or the shared head).
    public static func headKey(_ checkpointKey: String, numHiddenLayers: Int) -> String? {
        if checkpointKey == mtpEmbeddingKey {
            return "embed_tokens.weight"
        }
        // Dot boundary: layer 46 must not match layer 460.
        let layerPrefix = "model.layers.\(numHiddenLayers)."
        guard checkpointKey.hasPrefix(layerPrefix) else { return nil }
        let rest = String(checkpointKey.dropFirst(layerPrefix.count))
        if headTopLevelKeys.contains(rest) {
            return rest
        }
        // The head is shared with the target lm_head; the checkpoint carries no such tensor.
        if rest.hasPrefix("shared_head.head.") {
            return nil
        }
        return "decoder." + rest
    }

    /// Returns the head's weights keyed for `SharpAIDots3NoteMtpHead`. Target-model keys are dropped.
    public static func sanitize(
        _ weights: [String: MLXArray], numHiddenLayers: Int
    ) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            guard let headParameterKey = headKey(key, numHiddenLayers: numHiddenLayers) else {
                continue
            }
            out[headParameterKey] = value
        }
        return out
    }
}
