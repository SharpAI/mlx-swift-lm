// SharpAI fork overlay: dots3_note checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keeps HF key names, except routed experts: per-expert `mlp.experts.N.<proj>.<rest>` keys are
// stacked into `mlp.switch_mlp.<proj>.<rest>`, the same convention as DeepseekV3.swift.
// Drops `mtp.`-prefixed keys (vLLM's dots3_note loader skips them too), tied `lm_head`, text layers
// at or past num_hidden_layers, and `.indexer.` keys on sliding_attention layers.
// Sources: vLLM dots3_note loader (Dots3NoteModel._adapt_weights) and the checkpoint index of
// dots-studio/dots3-note-prev (per-expert keys, indexer keys on layer 0, audio_encoder.* keys).
// Verified against: https://huggingface.co/dots-studio/dots3-note-prev/raw/main/model.safetensors.index.json
//   The vision tower keys use the `vision_encoder.` prefix (blocks, adapter, patch_embed,
//   post_trunk_norm). The audio keys use `audio_encoder.audio_adapter.*` and
//   `audio_encoder.dots_encoder.speech_encoder.layers.N.*`. Tower keys have no text-layer index.
// Verified against: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/dots3_note/nvidia/model.py
//   (Dots3NoteModel._adapt_weights drops `.indexer.` on sliding_attention layers below num_hidden_layers).

import Foundation
import MLX

public enum SharpAIDots3NoteSanitizer {

    static let towerSegments: Set<String> = [
        "visual", "vision", "vision_tower", "vision_encoder", "audio", "audio_tower", "audio_encoder",
        "whisper",
    ]

    static let expertProjections: Set<String> = ["gate_proj", "up_proj", "down_proj"]

    /// Drops MTP, tied `lm_head`, text-layer keys past `num_hidden_layers` and sliding-layer
    /// indexer keys, then stacks routed experts.
    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIDots3NoteConfiguration
    ) -> [String: MLXArray] {
        let kept = weights.filter { key, _ in
            let parts = key.split(separator: ".")
            if parts.first == "mtp" {
                return false
            }
            if configuration.tieWordEmbeddings, parts.contains("lm_head") {
                return false
            }
            if parts.contains(where: { towerSegments.contains(String($0)) }) {
                return true
            }
            if let index = layerIndex(of: key) {
                if index >= configuration.hiddenLayers {
                    return false
                }
                if parts.contains("indexer"),
                    configuration.attentionGeometry(at: index)?.layerType == .slidingAttention
                {
                    return false
                }
            }
            return true
        }
        return stackRoutedExperts(kept)
    }

    /// Returns N for keys shaped like `...layers.N.<rest>`, or nil.
    public static func layerIndex(of key: String) -> Int? {
        let parts = key.split(separator: ".")
        guard let at = parts.firstIndex(of: "layers"), at + 1 < parts.count else {
            return nil
        }
        return Int(parts[at + 1])
    }

    /// Stacks `mlp.experts.N.<proj>.<rest>` into `mlp.switch_mlp.<proj>.<rest>` for N in 0..<count.
    /// A group whose expert indices are not exactly 0..<count is left unchanged.
    static func stackRoutedExperts(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var groups: [String: [Int: String]] = [:]
        for key in weights.keys {
            let parts = key.split(separator: ".").map(String.init)
            guard let e = parts.firstIndex(of: "experts"), e >= 1, parts[e - 1] == "mlp",
                e + 3 < parts.count, let expert = Int(parts[e + 1]),
                expertProjections.contains(parts[e + 2])
            else {
                continue
            }
            let head = parts[0 ..< e].joined(separator: ".")
            let tail = parts[(e + 3)...].joined(separator: ".")
            let target = "\(head).switch_mlp.\(parts[e + 2]).\(tail)"
            groups[target, default: [:]][expert] = key
        }

        var out = weights
        for (target, sources) in groups {
            guard Set(sources.keys) == Set(0 ..< sources.count) else {
                continue
            }
            let joined = (0 ..< sources.count).map { weights[sources[$0]!]! }
            out[target] = stacked(joined)
            for source in sources.values {
                out[source] = nil
            }
        }
        return out
    }
}
