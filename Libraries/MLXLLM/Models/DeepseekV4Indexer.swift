// DeepSeek-V4 sparse indexer (SL-35). SharpAI fork overlay: new file, not part of upstream.
//
// Layers with compress_ratio == 4 own a `V4Indexer` (checkpoint keys `layers.N.attn.indexer.{wq_b.weight,
// weights_proj.weight, compressor.*}`; the indexer has its own ratio-4 overlap compressor of width
// index_head_dim). For every query it scores the complete compressed blocks and keeps the top
// `index_topk`; the layer's attention then reads only those compressed entries (plus the local keys).
//
// Verified against the oracle (ConformanceV4IndexerTests): the indexer keys are the oracle's
// `softmaxPoolCompress` + `rmsNorm`; the selection equals the top-k of scores recomputed with the oracle's
// Hadamard rotation (`hadamardRotate`, the kpool rotation) applied to q AND k. This implementation skips
// the rotation: an orthonormal rotation of both operands leaves q.k unchanged, so only the CUDA kernel's
// fp8 quantisation (not modelled) would see it.
//
// INFERRED, not covered by any oracle (no V4 source offline):
//   1. score[t, b] = sum_h w[t, h] * relu(q_h[t] . k[b]), w = weights_proj(x) * nHeads^-0.5 * headDim^-0.5
//      (the repo's DeepseekV32Indexer formula).
//   2. q = wq_b(qNorm(wqA(x))) per index head; RoPE on the LAST `ropeDim` channels of q (position t) and
//      of k (position b * ratio), with the layer's rope.
//   3. only complete blocks (b + 1) * 4 - 1 <= t are candidates; k = min(index_topk, #candidates); a query
//      with no complete block reads no compressed entry.
//   4. no indexer sharing between layers (V4.1). With a KV cache the indexer keys come from the indexer
//      half of `V4CompressorStateCache` (batch size 1 only).
// upstream-watch: DeepseekV4.swift in mlx-swift-lm.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension SharpAIOverlay {
    final class V4Indexer: Module {
        static let ratio = 4

        let nHeads: Int
        let headDim: Int
        let topk: Int
        let ropeDim: Int

        @ModuleInfo(key: "wq_b") var wqB: Linear
        @ModuleInfo(key: "weights_proj") var weightsProj: Linear
        @ModuleInfo(key: "compressor") var compressor: V4Compressor

        init(hiddenSize: Int, qLoraRank: Int, nHeads: Int, headDim: Int, topk: Int, ropeDim: Int, eps: Float) {
            precondition(ropeDim <= headDim, "index_head_dim must be at least qk_rope_head_dim")
            self.nHeads = nHeads
            self.headDim = headDim
            self.topk = topk
            self.ropeDim = ropeDim
            self._wqB.wrappedValue = Linear(qLoraRank, nHeads * headDim, bias: false)
            self._weightsProj.wrappedValue = Linear(hiddenSize, nHeads, bias: false)
            self._compressor.wrappedValue = V4Compressor(
                hiddenSize: hiddenSize, headDim: headDim, ratio: Self.ratio, eps: eps)
        }

        /// Scores [B, L, nb]: -inf for blocks that are not complete at the query position.
        func scores(_ x: MLXArray, qr: MLXArray, rope: RoPELayer) -> MLXArray {
            let k = SharpAIOverlay.v4RopeCompressed(
                compressor(x), rope: rope, ropeDim: ropeDim, ratio: Self.ratio)  // [B, nb, headDim]
            return scores(x, qr: qr, rope: rope, keys: k, start: 0)
        }

        /// Same, with the indexer keys k [B, nb, headDim] (RoPE applied) given and the queries starting at
        /// absolute position `start` (cached decode).
        func scores(_ x: MLXArray, qr: MLXArray, rope: RoPELayer, keys k: MLXArray, start: Int) -> MLXArray {
            let b = x.dim(0), l = x.dim(1), nope = headDim - ropeDim
            let blocks = k.dim(1)
            var q = wqB(qr).reshaped(b, l, nHeads, headDim).transposed(0, 2, 1, 3)  // [B, H, L, D]
            q = concatenated([q[.ellipsis, ..<nope], rope(q[.ellipsis, nope...], offset: start)], axis: -1)
            var s = maximum(matmul(q, k.expandedDimensions(axis: 1).transposed(0, 1, 3, 2)), 0)  // [B, H, L, nb]
            let w = weightsProj(x) * (1 / Float(nHeads * headDim).squareRoot())
            s = (s * w.transposed(0, 2, 1).expandedDimensions(axis: -1)).sum(axis: 1)  // [B, L, nb]
            let visible = SharpAIOverlay.v4Visibility(length: l, blocks: blocks, ratio: Self.ratio, start: start)
            return which(visible.expandedDimensions(axis: 0), s, MLXArray(-Float.infinity))
        }

        /// Bool mask [B, L, nb]: the top-k complete blocks of every query.
        func select(_ x: MLXArray, qr: MLXArray, rope: RoPELayer) -> MLXArray {
            select(scores(x, qr: qr, rope: rope))
        }

        /// Cached form: keys from the state cache, queries starting at absolute position `start`.
        func select(_ x: MLXArray, qr: MLXArray, rope: RoPELayer, keys: MLXArray, start: Int) -> MLXArray {
            select(scores(x, qr: qr, rope: rope, keys: keys, start: start))
        }

        /// The selection step on a score matrix [B, L, nb] (-inf = not visible). Internal so the golden suite can feed it
        /// ties and rows with fewer than k visible entries directly.
        func select(_ s: MLXArray) -> MLXArray {
            let nb = s.dim(-1)
            let k = min(topk, nb)
            let top = argPartition(-s, kth: k - 1, axis: -1)[.ellipsis, ..<k]  // [B, L, k]
            let hit = (top.expandedDimensions(axis: -1) .== MLXArray(Int32(0) ..< Int32(nb))).any(axis: -2)
            return hit & (s .> -Float.infinity)
        }
    }
}
