// SharpAI fork overlay: cosmos3_omni checkpoint key sanitization. Not part of upstream.
//
// Mirrors Qwen3VL.sanitize: `model.visual.` -> `vision_tower.`, `model.language_model.` ->
// `language_model.model.`, `lm_head.` -> `language_model.lm_head.`.
// Also maps the root-level understanding layout of nvidia/Cosmos3-Nano (checkpoint index
// `model.safetensors.index.json`, 984 non-`_moe_gen` keys): `blocks.`, `patch_embed.`,
// `pos_embed.`, `merger.`, `deepstack_merger_list.` -> `vision_tower.`; `layers.`,
// `embed_tokens.`, `norm.` -> `language_model.model.`.
// The generation experts are verified as `*_moe_gen` suffixes (181 keys in the same index,
// e.g. `layers.0.mlp_moe_gen.down_proj.weight`). Keys with the marker are rejected, not dropped,
// so an unexpected layout fails loudly.
// Attention names are NOT renamed here, so the key layout test keeps `self_attn.to_q`. The second
// step `qwen3VLAttentionNames` maps them to the Qwen3-VL names that `Qwen3VL` builds (`q_proj`,
// `k_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm`). Verified by the tiny checkpoint key and shape
// list (https://huggingface.co/hf-tiny-v2/tiny-random-Cosmos3OmniForConditionalGeneration/resolve/main/model.safetensors):
// the mapped names and shapes match `Qwen3VL.parameters()`. No upstream conversion table was found.
// NOT handled here: diffusion-side keys (`action_*`, `audio_*`, `proj_in`/`proj_out`,
// `time_embedder.*`) pass through unchanged; vision weights live in the `vision_encoder/` subfolder.

import Foundation
import MLX

public enum SharpAICosmos3OmniSanitizer {

    static let generationMarker = "_moe_gen"

    static let renameRules: [(from: String, to: String)] = [
        ("model.visual.", "vision_tower."),
        ("model.language_model.", "language_model.model."),
        ("lm_head.", "language_model.lm_head."),
        ("blocks.", "vision_tower.blocks."),
        ("patch_embed.", "vision_tower.patch_embed."),
        ("pos_embed.", "vision_tower.pos_embed."),
        ("merger.", "vision_tower.merger."),
        ("deepstack_merger_list.", "vision_tower.deepstack_merger_list."),
        ("layers.", "language_model.model.layers."),
        ("embed_tokens.", "language_model.model.embed_tokens."),
        ("norm.", "language_model.model.norm."),
    ]

    /// Drops position ids, renames understanding weights, and fixes the patch embed layout.
    public static func sanitize(
        _ weights: [String: MLXArray], visionInChannels: Int = 3,
        tieWordEmbeddings: Bool = false
    ) throws -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            if key.contains("position_ids") {
                continue
            }
            if key.contains(generationMarker) {
                throw SharpAICosmos3OmniError.generationWeightNotSupported(key)
            }

            let newKey = remap(key)

            if tieWordEmbeddings && newKey.contains(".lm_head.") {
                continue
            }

            var newValue = value
            if newKey.contains("patch_embed.proj.weight"),
                value.ndim == 5, value.dim(-1) != visionInChannels
            {
                newValue = value.transposed(0, 2, 3, 4, 1)
            }

            guard sanitized[newKey] == nil else {
                throw SharpAICosmos3OmniError.duplicateWeightKey(newKey)
            }
            sanitized[newKey] = newValue
        }

        return sanitized
    }

    static let attentionRenameRules: [(from: String, to: String)] = [
        (".self_attn.to_q.", ".self_attn.q_proj."),
        (".self_attn.to_k.", ".self_attn.k_proj."),
        (".self_attn.to_v.", ".self_attn.v_proj."),
        (".self_attn.to_out.", ".self_attn.o_proj."),
        (".self_attn.norm_q.", ".self_attn.q_norm."),
        (".self_attn.norm_k.", ".self_attn.k_norm."),
    ]

    /// Renames text attention keys from the checkpoint layout (`to_q`, `to_k`, `to_v`, `to_out`,
    /// `norm_q`, `norm_k`) to the Qwen-VL names that `Qwen3VL` builds. Run after `sanitize`.
    /// Throws `duplicateWeightKey` if two keys map to the same name.
    public static func qwen3VLAttentionNames(_ weights: [String: MLXArray]) throws
        -> [String: MLXArray]
    {
        var renamed = [String: MLXArray]()
        for (key, value) in weights {
            var newKey = key
            for rule in attentionRenameRules where newKey.contains(rule.from) {
                newKey = newKey.replacingOccurrences(of: rule.from, with: rule.to)
            }
            guard renamed[newKey] == nil else {
                throw SharpAICosmos3OmniError.duplicateWeightKey(newKey)
            }
            renamed[newKey] = value
        }
        return renamed
    }

    static func remap(_ key: String) -> String {
        for rule in renameRules where key.hasPrefix(rule.from) {
            return rule.to + String(key.dropFirst(rule.from.count))
        }
        return key
    }
}
