// SharpAI fork overlay: hrm_text prefix-LM attention mask. Not part of upstream mlx-swift-lm.
//
// Rule from transformers hrm_text (HrmTextModel.forward and the docs, v5.22.1): tokens with
// token_type_ids == 1 form one bidirectional block; all other positions are causal. The model
// maps token_type_ids to block ids (1 -> 0, otherwise -1) and passes them to create_causal_mask.
// The block applies only to prefill: once the cache is initialized, decode steps are causal.

import Foundation
import MLX

public enum SharpAIHrmTextPrefixMask {

    /// Returns a [length, length] bool mask. Query `i` may attend key `j` when `j < prefixLength`
    /// (bidirectional prefix) or `j <= i` (causal). `prefixLength` is clamped to 0...length.
    public static func prefixLMMask(length: Int, prefixLength: Int) -> MLXArray {
        let n = max(length, 0)
        let prefix = min(max(prefixLength, 0), n)
        let blockIds = (0 ..< n).map { $0 < prefix ? 0 : -1 }
        return attentionMask(queryLength: n, cacheOffset: 0, blockIds: blockIds)
    }

    /// Returns a bool mask of shape [queryLength, cacheOffset + queryLength]. Query `i` sits at
    /// absolute position `cacheOffset + i` and may attend key `j` when `j <= cacheOffset + i`.
    ///
    /// `blockIds` (one per query, prefill only) adds the bidirectional block: query `i` may also
    /// attend key `j` when `blockIds[i] >= 0` and `blockIds[j] == blockIds[i]`. It is ignored when
    /// `cacheOffset > 0`, matching the HF rule that the block applies only before the cache is
    /// initialized. A negative block id means no block.
    public static func attentionMask(
        queryLength: Int, cacheOffset: Int = 0, blockIds: [Int]? = nil
    ) -> MLXArray {
        let q = max(queryLength, 0)
        let offset = max(cacheOffset, 0)
        let s = offset + q
        let blocks: [Int]? = offset == 0 ? blockIds : nil
        if let blocks {
            precondition(blocks.count == q, "blockIds must have one entry per query")
        }

        var flat: [Int32] = []
        flat.reserveCapacity(q * s)
        for i in 0 ..< q {
            let position = offset + i
            for j in 0 ..< s {
                var visible = j <= position
                if let blocks, blocks[i] >= 0, blocks[j] == blocks[i] {
                    visible = true
                }
                flat.append(visible ? 1 : 0)
            }
        }
        return MLXArray(flat, [q, s]).asType(.bool)
    }
}
