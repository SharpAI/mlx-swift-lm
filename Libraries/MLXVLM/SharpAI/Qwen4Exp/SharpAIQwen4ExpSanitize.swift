// SharpAI fork overlay: qwen4_exp checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Rules, from the released checkpoint index (Qwen/Qwen3.8-Flash-Next, model.safetensors.index.json):
//   - drop rotary buffers (rotary_emb.inv_freq), vision keys (model.visual.*) and MTP keys (mtp.*).
//     The text path does not port the vision tower or the MTP head.
//   - drop the PLE buffers layer_multipliers, ngram_heads_vocab_sizes and ngram_heads_offsets. They
//     are recomputed from the config, and the port's values match the checkpoint (see tests).
//   - concatenate ple_embedding.ngram_embedding.shard_0..127.weight along axis 0 into one table.
//     Each shard is [2500012, 160] and the padded vocab is 320001536 = 128 * 2500012.
//   - split mlp.experts.gate_up_proj (E, 2I, H) into switch_mlp gate_proj and up_proj (E, I, H),
//     and map mlp.experts.down_proj (E, H, I) to switch_mlp.down_proj.
//   - move conv1d weights from torch (C, 1, k) to MLX (C, k, 1).
// Norm weights are kept as stored: the module applies (1 + w), as the reference does.
// Verified against: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py

import Foundation
import MLX

public enum SharpAIQwen4ExpSanitizer {

    private static let shardMarker = ".ngram_embedding.shard_"
    private static let pleBufferSuffixes = [
        ".ple_embedding.layer_multipliers",
        ".ple_embedding.ngram_heads_vocab_sizes",
        ".ple_embedding.ngram_heads_offsets",
    ]

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = weights

        // Drop the unported families and buffers recomputed from the config.
        for key in Array(out.keys) {
            if key.contains("rotary_emb.inv_freq") || key.hasPrefix("model.visual.")
                || key.hasPrefix("mtp.") || pleBufferSuffixes.contains(where: { key.hasSuffix($0) })
            {
                out[key] = nil
            }
        }

        // Concatenate the n-gram table shards in index order.
        var shardGroups: [String: [(Int, String)]] = [:]
        for key in out.keys where key.contains(shardMarker) && key.hasSuffix(".weight") {
            guard let markerRange = key.range(of: shardMarker) else { continue }
            let prefix = String(key[..<markerRange.lowerBound])
            let indexText = key[markerRange.upperBound...].dropLast(".weight".count)
            guard let index = Int(indexText) else { continue }
            shardGroups[prefix, default: []].append((index, key))
        }
        for (prefix, entries) in shardGroups {
            let ordered = entries.sorted { $0.0 < $1.0 }.compactMap { out.removeValue(forKey: $0.1) }
            if !ordered.isEmpty {
                out[prefix + ".ngram_embedding.weight"] = concatenated(ordered, axis: 0)
            }
        }

        // Split the fused expert tensors into SwitchGLU weights.
        for key in Array(out.keys) where key.hasSuffix(".mlp.experts.gate_up_proj") {
            guard let fused = out.removeValue(forKey: key) else { continue }
            let prefix = String(key.dropLast("experts.gate_up_proj".count))
            let intermediate = fused.dim(1) / 2
            out[prefix + "switch_mlp.gate_proj.weight"] = fused[0..., ..<intermediate, 0...]
            out[prefix + "switch_mlp.up_proj.weight"] = fused[0..., intermediate..., 0...]
        }
        for key in Array(out.keys) where key.hasSuffix(".mlp.experts.down_proj") {
            guard let down = out.removeValue(forKey: key) else { continue }
            let prefix = String(key.dropLast("experts.down_proj".count))
            out[prefix + "switch_mlp.down_proj.weight"] = down
        }

        // Depthwise conv weights: torch (C, 1, k) -> MLX (C, k, 1).
        for key in Array(out.keys) where key.contains("conv1d.weight") {
            guard let value = out[key], value.dim(-1) != 1 else { continue }
            out[key] = value.movedAxis(source: 2, destination: 1)
        }

        return out
    }
}
