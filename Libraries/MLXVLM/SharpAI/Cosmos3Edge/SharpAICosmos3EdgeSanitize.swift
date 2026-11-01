// SharpAI fork overlay: cosmos3_edge checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Source for the vision and projector names: the vLLM `Cosmos3EdgeForConditionalGeneration.hf_to_vllm_mapper`
// (https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/cosmos3_edge/). Its prefix rules
// `model.projector.` -> `visual.projector.` and `model.visual.` -> `visual.encoder.` are applied here.
// The projector and vision key names were checked against the real index of nvidia/Cosmos3-Edge
// (vision_encoder/model.safetensors: model.projector.linear_fc1.weight and so on).
//
// UNVERIFIED: key remapping for the language tower. The nvidia/Cosmos3-Edge transformer shards use
// diffusers-style names (layers.N.self_attn.to_q, layers.N.mlp.up_proj), and the vLLM mapping for them
// is not ported here. Those keys keep their names.

import Foundation
import MLX

public enum SharpAICosmos3EdgeSanitizer {

    /// Anchored prefix renames for the projector and vision tower (startsWith, as in vLLM's WeightsMapper).
    static let prefixRenames: [(from: String, to: String)] = [
        ("model.projector.", "visual.projector."),
        ("model.visual.", "visual.encoder."),
    ]

    /// Drops non-weight rotary buffers, and renames the projector and vision prefixes. All other keys keep their names.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") {
                continue
            }
            var newKey = key
            for rename in prefixRenames where key.hasPrefix(rename.from) {
                newKey = rename.to + String(key.dropFirst(rename.from.count))
                break
            }
            result[newKey] = value
        }
        return result
    }
}
