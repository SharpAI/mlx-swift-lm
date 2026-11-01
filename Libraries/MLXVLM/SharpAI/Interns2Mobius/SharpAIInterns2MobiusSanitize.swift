// SharpAI fork overlay: interns2_mobius checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Source: HF internlm/Intern-S2-Mobius modeling_interns2_mobius.py (block_idx rule) and the
// model.safetensors.index.json key summary. Keys keep their HF names. The routed MoE bank is
// stored as `meta_mlp.<block>` and is selected per layer by `bankIndex`.
// Per-layer shared-expert keys (`mlp.shared_expert.*`) contain "shared"; the bank keys do not.

import Foundation
import MLX

public enum SharpAIInterns2MobiusSanitizer {

    /// Drops non-weight rotary buffers. All other keys keep their HF names.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }

    /// Layer index from the first `layers.<n>` segment of a key, or nil.
    public static func layerIndex(of key: String) -> Int? {
        let parts = key.split(separator: ".")
        guard parts.count >= 2 else { return nil }
        for i in 0..<(parts.count - 1) where parts[i] == "layers" {
            if let index = Int(parts[i + 1]) {
                return index
            }
        }
        return nil
    }

    /// Index of the routed MoE bank (`meta_mlp.<index>`) that a decoder layer uses.
    /// Source: HF modeling_interns2_mobius.py, `block_idx = self.layer_idx % self.num_blocks`.
    public static func bankIndex(forLayer layer: Int, numBlocks: Int) -> Int {
        precondition(numBlocks > 0, "numBlocks must be positive")
        return layer % numBlocks
    }

    /// Groups per-layer shared-expert keys (`mlp.shared_expert.*`) by their layer index.
    /// Keys with no layer index are skipped.
    public static func sharedBankKeysByLayer(_ keys: [String]) -> [Int: [String]] {
        var grouped: [Int: [String]] = [:]
        for key in keys where key.contains("shared") {
            guard let layer = layerIndex(of: key) else { continue }
            grouped[layer, default: []].append(key)
        }
        return grouped.mapValues { $0.sorted() }
    }
}
