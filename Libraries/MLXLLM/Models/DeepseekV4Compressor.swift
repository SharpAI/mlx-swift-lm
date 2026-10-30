// DeepSeek-V4 KV compressor (SL-35). SharpAI fork overlay: new file, not part of upstream.
//
// A layer with compress_ratio > 1 owns one `V4Compressor` (checkpoint keys
// `layers.N.attn.compressor.{ape,wkv.weight,wgate.weight,norm.weight}`, shapes from the vLLM loader
// layout in ConformanceSynthV4Layout). It turns every COMPLETE block of `ratio` tokens into one KV entry.
//
// Verified against the oracle `SharpAIOverlayGapReference.softmaxPoolCompress` + `rmsNorm`
// (ConformanceV4CompressorTests): the pooling (softmax over the block window of `wgate(x) + ape`, weighted
// sum of `wkv(x)`, previous block read through the second channel half when ratio == 4) and the RMSNorm.
//
// INFERRED, not covered by the oracle (no V4 reference source is available offline; written from the
// vLLM-derived notes in docs/research and the public CSA/HCA description). Each is a single place to change:
//   1. RoPE position of compressed entry b = b * ratio (first token of its block), applied to the last
//      `qk_rope_head_dim` channels with the layer's own rope (`ropeCompressed`).
//   2. Visibility: query t sees entry b iff the block is complete at t, (b + 1) * ratio - 1 <= t
//      (`visibility`). The local (uncompressed) keys keep the existing causal mask.
//   3. Compressed entries are appended after the local keys of the same layer and serve as both key and
//      value (k = v, as for the local keys).
//   4. The input of the compressor is the attention-normed hidden state (the same x as wq_a / wkv).
//   5. With a KV cache the compressor state lives in `V4CompressorStateCache` (DeepseekV4CompressorState.swift)
//      and the layer attends over the same entries as the full-sequence forward (batch size 1 only).
// Not implemented: the sliding-window restriction of the local keys (separate gap), fp8 quantisation.
// upstream-watch: DeepseekV4.swift in mlx-swift-lm.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension SharpAIOverlay {
    final class V4Compressor: Module {
        let ratio: Int
        let headDim: Int
        let overlap: Bool

        /// [ratio, coff * headDim]; coff = 2 when ratio == 4 (overlap), else 1.
        var ape: MLXArray
        @ModuleInfo(key: "wkv") var wkv: Linear
        @ModuleInfo(key: "wgate") var wgate: Linear
        @ModuleInfo(key: "norm") var norm: RMSNorm

        init(hiddenSize: Int, headDim: Int, ratio: Int, eps: Float) {
            precondition(ratio > 1, "a compressor needs compress_ratio > 1")
            self.ratio = ratio
            self.headDim = headDim
            self.overlap = ratio == 4
            let coff = overlap ? 2 : 1
            self.ape = zeros([ratio, coff * headDim])
            self._wkv.wrappedValue = Linear(hiddenSize, coff * headDim, bias: false)
            self._wgate.wrappedValue = Linear(hiddenSize, coff * headDim, bias: false)
            self._norm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        }

        /// x: [B, T, hidden] -> [B, T / ratio, headDim] (RMSNorm applied, no RoPE). Trailing tokens that
        /// do not fill a block are dropped.
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let b = x.dim(0), nb = x.dim(1) / ratio
            if nb == 0 { return zeros([b, 0, headDim], dtype: x.dtype) }
            let xs = x[0..., ..<(nb * ratio)]
            return norm(
                Self.pool(
                    kv: wkv(xs), score: wgate(xs), ape: ape, ratio: ratio, overlap: overlap,
                    headDim: headDim))
        }

        /// Batched, vectorised form of `SharpAIOverlayGapReference.softmaxPoolCompress`.
        /// kv, score: [B, nb * ratio, coff * headDim]; returns [B, nb, headDim].
        static func pool(
            kv: MLXArray, score: MLXArray, ape: MLXArray, ratio: Int, overlap: Bool, headDim: Int
        ) -> MLXArray {
            let b = kv.dim(0), c = kv.dim(2), nb = kv.dim(1) / ratio
            let k4 = kv.reshaped(b, nb, ratio, c)
            let s4 = score.reshaped(b, nb, ratio, c) + ape
            if !overlap {
                return (k4 * softmax(s4, axis: 2)).sum(axis: 2)
            }
            // Window of block i = previous block (first channel half) ++ this block (second half).
            // Block 0 has no previous block: score -inf, kv 0.
            let curK = k4[.ellipsis, headDim...], curS = s4[.ellipsis, headDim...]
            let prevK = concatenated(
                [zeros([b, 1, ratio, headDim], dtype: kv.dtype), k4[0..., ..<(nb - 1), 0..., ..<headDim]], axis: 1)
            let prevS = concatenated(
                [
                    full([b, 1, ratio, headDim], values: MLXArray(-Float.infinity), dtype: score.dtype),
                    s4[0..., ..<(nb - 1), 0..., ..<headDim],
                ], axis: 1)
            let k = concatenated([prevK, curK], axis: 2)
            let s = concatenated([prevS, curS], axis: 2)
            return (k * softmax(s, axis: 2)).sum(axis: 2)
        }
    }

    // MARK: attention with compressed entries

    /// Rotate the RoPE part (last `ropeDim` channels) of compressed entries c: [B, nb, headDim]; entry b
    /// gets position b * ratio (inferred, see the file header). Works for any `rope` by rotating a
    /// zero-padded sequence in which entry b sits at slot b * ratio.
    static func v4RopeCompressed(
        _ c: MLXArray, rope: RoPELayer, ropeDim: Int, ratio: Int, firstBlock: Int = 0
    ) -> MLXArray {
        let b = c.dim(0), nb = c.dim(1), nope = c.dim(2) - ropeDim
        let r = c[.ellipsis, nope...].reshaped(b, nb, 1, ropeDim)
        let padded = concatenated([r, zeros([b, nb, ratio - 1, ropeDim], dtype: c.dtype)], axis: 2)
            .reshaped(b, 1, nb * ratio, ropeDim)
        let rotated = rope(padded, offset: firstBlock * ratio).reshaped(b, nb, ratio, ropeDim)[0..., 0..., 0, 0...]
        return concatenated([c[.ellipsis, ..<nope], rotated], axis: -1)
    }

    /// [L, nb] bool: compressed entry b is complete at query position t (queries start at `start`).
    static func v4Visibility(length: Int, blocks: Int, ratio: Int, start: Int = 0) -> MLXArray {
        let t = MLXArray(Int32(start) ..< Int32(start + length)).reshaped(length, 1)
        let end = (MLXArray(Int32(0) ..< Int32(blocks)).reshaped(1, blocks) + 1) * Int32(ratio) - 1
        return end .<= t
    }

    /// Attention over local keys plus compressed entries. Without a cache localKeys has S == L; with a cache
    /// it holds all S cached positions and the L queries are the last L of them.
    /// queries: [B, H, L, D]; localKeys: [B, 1, S, D]; compressed: [B, nb, D] (normed, RoPE applied);
    /// compressedMask: [B or 1, L, nb] bool (which entries each query may read). k = v.
    /// Returns nil when the incoming local mask is neither causal nor boolean (caller keeps the old path).
    static func v4CompressedAttention(
        queries: MLXArray, localKeys: MLXArray, compressed: MLXArray, compressedMask: MLXArray,
        localMask: MLXFast.ScaledDotProductAttentionMaskMode, scale: Float, sinks: MLXArray?
    ) -> MLXArray? {
        let l = queries.dim(2), s = localKeys.dim(2)
        let local: MLXArray
        switch localMask {
        case .causal: local = createCausalMask(n: l, offset: s - l)
        case .none: local = MLXArray.ones([l, s], dtype: .bool)
        case .array(let m) where m.dtype == .bool: local = m
        default: return nil
        }
        let cm = compressedMask.expandedDimensions(axis: 1)  // [Bm, 1, L, nb]
        let lm = broadcast(local.reshaped([-1, 1, l, s]), to: [cm.dim(0), 1, l, s])
        let mask = concatenated([lm, cm], axis: -1)
        let keys = concatenated([localKeys, compressed.expandedDimensions(axis: 1)], axis: 2)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: keys, scale: scale, mask: .array(mask), sinks: sinks)
    }
}
