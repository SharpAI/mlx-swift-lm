// SharpAI fork overlay: cosmos3_omni checkpoint key sanitization. Not part of upstream.
//
// Mirrors Qwen3VL.sanitize: `model.visual.` -> `vision_tower.`, `model.language_model.` ->
// `language_model.model.`, `lm_head.` -> `language_model.lm_head.`.
// Also maps the root-level understanding layout of nvidia/Cosmos3-Nano (checkpoint index
// `model.safetensors.index.json`, 1165 keys: 814 transformer + 351 vision): `blocks.`, `patch_embed.`,
// `pos_embed.`, `merger.`, `deepstack_merger_list.` -> `vision_tower.`; `layers.`,
// `embed_tokens.`, `norm.` -> `language_model.model.`.
// The generation tower (Mixture-of-Transformers second stream, NOT routed experts) is mapped, not
// rejected, into two namespaces that the wrapper model owns: `gen.*` (layers and final norm) and
// `gen_heads.*` (modality heads). Mapping, checkpoint -> post-sanitize:
//   layers.N.self_attn.{add_q_proj,add_k_proj,add_v_proj,to_add_out,norm_added_q,norm_added_k}.weight
//     -> gen.layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight
//   layers.N.mlp_moe_gen.{gate,up,down}_proj.weight -> gen.layers.N.mlp.{gate,up,down}_proj.weight
//   layers.N.{input_layernorm,post_attention_layernorm}_moe_gen.weight -> gen.layers.N.<name>.weight
//   norm_moe_gen.weight -> gen.norm.weight
//   the 18 head keys (proj_in, proj_out, time_embedder.linear_{1,2}, audio_*, action_*) keep their
//     names under `gen_heads.`
// Verified against https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_cosmos3.py
// (attention gen projections lines 280-289, decoder layer 355-371, final norms 664, heads 441-490)
// and NVIDIA/cosmos-framework cosmos_framework/model/generator/mot/unified_mot.py:538-625.
// Verified against transformer_cosmos3.py:60-100 (Cosmos3AttnProcessor applies add_q/k/v_proj and
// norm_added_q/k to the gen stream, to_add_out to its output). Only bias-free keys (`.weight`) are
// mapped (Nano `attention_bias` false); a `.bias` key fails loudly.
// An unrecognized key throws `unrecognizedWeightKey` instead of passing through: the previous
// pass-through let the 216 `add_*`/`to_add_out`/`norm_added_*` keys reach the Qwen3VL load
// unrenamed (they did not contain `_moe_gen`, so the old rejection also missed them).
// Attention names are NOT renamed here, so the key layout test keeps `self_attn.to_q`. The second
// step `qwen3VLAttentionNames` maps them to the Qwen3-VL names that `Qwen3VL` builds (`q_proj`,
// `k_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm`). Verified by the tiny checkpoint key and shape
// list (https://huggingface.co/hf-tiny-v2/tiny-random-Cosmos3OmniForConditionalGeneration/resolve/main/model.safetensors):
// the mapped names and shapes match `Qwen3VL.parameters()`. No upstream conversion table was found.
// Vision weights live in the `vision_encoder/` subfolder (see SharpAICosmos3OmniWeightFiles).

import Foundation
import MLX

public enum SharpAICosmos3OmniSanitizer {

    static let generationPrefix = "gen."
    static let generationHeadPrefix = "gen_heads."

    /// Per-layer suffix rules of the generation tower: checkpoint suffix -> post-sanitize suffix.
    static let generationLayerRules: [String: String] = [
        "self_attn.add_q_proj.weight": "self_attn.q_proj.weight",
        "self_attn.add_k_proj.weight": "self_attn.k_proj.weight",
        "self_attn.add_v_proj.weight": "self_attn.v_proj.weight",
        "self_attn.to_add_out.weight": "self_attn.o_proj.weight",
        "self_attn.norm_added_q.weight": "self_attn.q_norm.weight",
        "self_attn.norm_added_k.weight": "self_attn.k_norm.weight",
        "mlp_moe_gen.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp_moe_gen.up_proj.weight": "mlp.up_proj.weight",
        "mlp_moe_gen.down_proj.weight": "mlp.down_proj.weight",
        "input_layernorm_moe_gen.weight": "input_layernorm.weight",
        "post_attention_layernorm_moe_gen.weight": "post_attention_layernorm.weight",
    ]

    /// The 18 head keys of the transformer index (video flow I/O 8, audio 5, action 5).
    static let generationHeadKeys: Set<String> = [
        "proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias",
        "time_embedder.linear_1.weight", "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight", "time_embedder.linear_2.bias",
        "audio_modality_embed", "audio_proj_in.weight", "audio_proj_in.bias",
        "audio_proj_out.weight", "audio_proj_out.bias",
        "action_modality_embed", "action_proj_in.fc.weight", "action_proj_in.bias.weight",
        "action_proj_out.fc.weight", "action_proj_out.bias.weight",
    ]

    private static let languageModelPrefix = "model.language_model."

    /// Maps a generation tower key, or returns nil when the key is not one.
    /// Head keys of a disabled head (`sound_gen` / `action_gen` false) are not mapped; they then
    /// fail as unrecognized, since the model has no such parameters.
    static func mapGeneration(_ key: String, soundGen: Bool = true, actionGen: Bool = true)
        -> String?
    {
        if generationHeadKeys.contains(key) {
            if key.hasPrefix("audio_") && !soundGen { return nil }
            if key.hasPrefix("action_") && !actionGen { return nil }
            return generationHeadPrefix + key
        }
        let name = key.hasPrefix(languageModelPrefix)
            ? String(key.dropFirst(languageModelPrefix.count)) : key
        if name == "norm_moe_gen.weight" {
            return generationPrefix + "norm.weight"
        }
        guard name.hasPrefix("layers.") else { return nil }
        let rest = name.dropFirst("layers.".count)
        guard let dot = rest.firstIndex(of: "."), !rest[..<dot].isEmpty,
            rest[..<dot].allSatisfy(\.isASCII), rest[..<dot].allSatisfy(\.isNumber),
            let mapped = generationLayerRules[String(rest[rest.index(after: dot)...])]
        else { return nil }
        return generationPrefix + "layers." + String(rest[..<dot]) + "." + mapped
    }

    /// True for keys that can only belong to the generation tower, mapped or not.
    static func looksLikeGeneration(_ key: String) -> Bool {
        if key.contains("_moe_gen") || key.contains(".add_q_proj.") || key.contains(".add_k_proj.")
            || key.contains(".add_v_proj.") || key.contains(".to_add_out.")
            || key.contains(".norm_added_")
        {
            return true
        }
        return ["proj_in.", "proj_out.", "time_embedder.", "audio_", "action_"].contains {
            key.hasPrefix($0)
        }
    }

    /// Splits sanitized weights into the understanding keys and the `gen.*` / `gen_heads.*` keys.
    public static func partitionGeneration(_ weights: [String: MLXArray]) -> (
        understanding: [String: MLXArray], generation: [String: MLXArray]
    ) {
        var understanding = [String: MLXArray]()
        var generation = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix(generationPrefix) || key.hasPrefix(generationHeadPrefix) {
                generation[key] = value
            } else {
                understanding[key] = value
            }
        }
        return (understanding, generation)
    }

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

    /// Drops position ids, renames understanding weights, maps the generation tower to `gen.*` and
    /// `gen_heads.*`, and fixes the patch embed layout.
    ///
    /// Throws `unrecognizedWeightKey` for a key that matches no known layout and
    /// `duplicateWeightKey` when two keys map to one name.
    public static func sanitize(
        _ weights: [String: MLXArray], visionInChannels: Int = 3,
        tieWordEmbeddings: Bool = false, soundGen: Bool = true, actionGen: Bool = true
    ) throws -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            if key.contains("position_ids") {
                continue
            }

            if let generationKey = mapGeneration(key, soundGen: soundGen, actionGen: actionGen) {
                guard sanitized[generationKey] == nil else {
                    throw SharpAICosmos3OmniError.duplicateWeightKey(generationKey)
                }
                sanitized[generationKey] = value
                continue
            }
            if looksLikeGeneration(key) {
                throw SharpAICosmos3OmniError.unrecognizedWeightKey(key)
            }

            let newKey = remap(key)
            guard newKey.hasPrefix("vision_tower.") || newKey.hasPrefix("language_model.") else {
                throw SharpAICosmos3OmniError.unrecognizedWeightKey(key)
            }

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
