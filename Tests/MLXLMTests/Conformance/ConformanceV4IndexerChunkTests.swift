// SL-37 lane LIB: the indexer's score and select steps are chunked over the query axis so a long prompt cannot
// materialise [B, H, L, nb] floats / [B, L, k, nb] bools. A problem that fits one chunk takes the unchunked
// path; a chunked run must select exactly the same blocks.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

struct ConformanceV4IndexerChunkTests {
    typealias Ix = SharpAIOverlay.V4Indexer

    @Test("chunk planning bounds the per-step element count by the constant, at the 128k-context shape")
    func peakBounded() {
        // B = 1, H = 64, nb = 32768 (128k / 4), k = 512, a 2048-token prefill chunk
        let b = 1, h = 64, nb = 32768, k = 512, l = 2048
        let cap = Ix.maxIntermediateElements
        let scoreRows = Ix.chunkRows(perRow: b * h * nb, total: l, maxElements: cap)
        let selectRows = Ix.chunkRows(perRow: b * k * nb, total: l, maxElements: cap)
        #expect(scoreRows >= 1 && scoreRows < l, "score step must chunk at this shape")
        #expect(selectRows >= 1 && selectRows < l, "select step must chunk at this shape")
        #expect(scoreRows * b * h * nb <= cap, "score intermediate \(scoreRows * b * h * nb) > cap")
        #expect(selectRows * b * k * nb <= cap, "select intermediate \(selectRows * b * k * nb) > cap")
        // the unchunked tensors would have been far larger
        #expect(l * b * k * nb > 100 * cap)
        // a single row over the cap still makes progress
        #expect(Ix.chunkRows(perRow: cap * 4, total: 10, maxElements: cap) == 1)
        // a problem under the cap is one chunk
        #expect(Ix.chunkRows(perRow: 10, total: 7, maxElements: cap) == 7)
    }

    @Test("chunked select equals the unchunked selection on random scores, ties and short rows")
    func selectChunkedEqualsUnchunked() {
        let (b, l, nb) = (2, 37, 23)
        let ix = Ix(hiddenSize: 8, qLoraRank: 8, nHeads: 2, headDim: 8, topk: 5, ropeDim: 4, eps: 1e-6)
        var s = MLXRandom.normal([b, l, nb], key: MLXRandom.key(5))
        // ties and invisible entries: quantise, and blank the right part of the early rows
        s = round(s * Float(4)) / Float(4)
        let col = MLXArray(Int32(0) ..< Int32(nb)).reshaped(1, 1, nb)
        let row = MLXArray(Int32(0) ..< Int32(l)).reshaped(1, l, 1)
        s = which(col .> (row + 3), MLXArray(-Float.infinity), s)
        let whole = ix.select(s, maxElements: Int.max)
        for cap in [1, b * 5 * nb, b * 5 * nb * 3, b * 5 * nb * 16] {
            let chunked = ix.select(s, maxElements: cap)
            eval(whole, chunked)
            #expect(chunked.shape == whole.shape)
            #expect(chunked.dtype == whole.dtype)
            #expect(all(chunked .== whole).item(Bool.self), "cap \(cap)")
        }
        // the default entry point is the unchunked path at this size
        #expect(all(ix.select(s) .== whole).item(Bool.self))
    }

    @Test("chunked scores match the unchunked scores to float error and select the same blocks through the whole indexer")
    func scoresChunkedEqualsUnchunked() throws {
        let (attn, x) = try ConformanceV4IndexerTests.make(topk: 3, tokens: 41)
        let ix = try #require(attn.indexer)
        let qr = attn.qNorm(attn.wqA(x))
        let k = SharpAIOverlay.v4RopeCompressed(
            ix.compressor(x), rope: attn.rope, ropeDim: ix.ropeDim, ratio: Ix.ratio)
        for start in [0, 7] {
            let whole = ix.scores(x, qr: qr, rope: attn.rope, keys: k, start: start, maxElements: Int.max)
            let nb = k.dim(1)
            for rowsPerChunk in [1, 5, 16] {
                let cap = rowsPerChunk * ix.nHeads * nb
                let chunked = ix.scores(x, qr: qr, rope: attn.rope, keys: k, start: start, maxElements: cap)
                eval(whole, chunked)
                #expect(chunked.shape == whole.shape)
                // -inf positions are identical. The finite scores are NOT bit-identical (measured max abs diff 3.6e-7 on these
                // shapes), so they are compared with a tolerance.
                #expect(all((chunked .== -Float.infinity) .== (whole .== -Float.infinity)).item(Bool.self))
                let finite = whole .> -Float.infinity
                let diff = abs(which(finite, chunked, MLXArray(Float(0))) - which(finite, whole, MLXArray(Float(0))))
                #expect(diff.max().item(Float.self) < 1e-5, "start \(start) rows \(rowsPerChunk)")
                let a = ix.select(whole, maxElements: Int.max)
                let b = ix.select(chunked, maxElements: ix.topk * nb)  // one row per select chunk
                #expect(all(a .== b).item(Bool.self), "selection start \(start) rows \(rowsPerChunk)")
            }
        }
    }
}
