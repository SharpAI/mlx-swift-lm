// SL-35 step 4.2: DeepSeek-V4 sparse indexer on compress_ratio == 4 layers.
//
// The expected block selection is computed in the test from the pure-MLX oracle ops of GapReference:
//   keys   = softmaxPoolCompress (overlap, ratio 4, index_head_dim) + rmsNorm, RoPE at block start,
//   scores = sum over index heads of w[t,h] * relu(q_h[t] . k[b]),  w = weights_proj(x) / sqrt(nHeads * headDim)
// with BOTH q and k rotated by the kpool Hadamard (`hadamardRotate`). The implementation does not rotate:
// an orthonormal rotation of both operands leaves q.k unchanged (it only matters for the CUDA kernel's fp8
// quantisation), so the two must agree up to float error. That is the numeric comparison against the oracle.
//
// INFERRED (no V4 source offline): the score formula and scaling are those of the repo's DeepseekV32Indexer
// (relu, per-head weights, sum over heads, scale headDim^-0.5 * nHeads^-0.5); RoPE goes on the LAST
// qk_rope_head_dim channels of the index head (as for the main attention), q at its position, k at the
// block start; the indexer reads the q-lora residual qNorm(wqA(x)) and the attention-normed x.
// upstream-watch: DeepseekV4.swift in mlx-swift-lm.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

struct ConformanceV4IndexerTests {
    /// Oracle scores [L][nb] for batch 0 and the set of selected blocks per query.
    struct Expected {
        var scores: [[Float]]
        var selected: [[Bool]]  // [t][b]
        var visible: [[Bool]]
    }

    static func expected(attn: DeepseekV4Attention, x: MLXArray, topk: Int) throws -> Expected {
        let ix = try #require(attn.indexer)
        let L = x.dim(1), nh = ix.nHeads, hd = ix.headDim, rd = ix.ropeDim, nope = hd - rd
        let comp = ix.compressor
        let kv = matmul(x[0], comp.wkv.weight.T), score = matmul(x[0], comp.wgate.weight.T)
        let pooled = SharpAIOverlayGapReference.rmsNorm(
            SharpAIOverlayGapReference.softmaxPoolCompress(
                kv: kv, score: score, ape: comp.ape, ratio: 4, overlap: true, headDim: hd),
            weight: comp.norm.weight, eps: attn.config.rmsNormEps)
        let nb = pooled.dim(0)
        let keys = (0 ..< nb).map { b -> MLXArray in
            let e = pooled[b ..< (b + 1)]
            let r = attn.rope(e[0..., nope...].reshaped(1, 1, 1, -1), offset: b * 4).reshaped(1, -1)
            return concatenated([e[0..., ..<nope], r], axis: -1)
        }
        let k = SharpAIOverlayGapReference.hadamardRotate(concatenated(keys, axis: 0))  // [nb, hd]
        let qr = attn.qNorm(attn.wqA(x))
        var q = ix.wqB(qr).reshaped(1, L, nh, hd).transposed(0, 2, 1, 3)  // [1, nh, L, hd]
        q = concatenated([q[.ellipsis, ..<nope], attn.rope(q[.ellipsis, nope...], offset: 0)], axis: -1)
        let qh = SharpAIOverlayGapReference.hadamardRotate(q[0])  // [nh, L, hd]
        let w = ix.weightsProj(x)[0] * (1 / Float(nh * hd).squareRoot())  // [L, nh]
        var total = MLXArray.zeros([L, nb])
        for h in 0 ..< nh {
            total = total + maximum(matmul(qh[h], k.T), MLXArray(Float(0))) * w[0..., h ..< (h + 1)]
        }
        eval(total)
        let flat = total.asArray(Float.self)
        var scores = [[Float]](), selected = [[Bool]](), visible = [[Bool]]()
        for t in 0 ..< L {
            let row = Array(flat[(t * nb) ..< ((t + 1) * nb)])
            let vis = (0 ..< nb).map { ($0 + 1) * 4 - 1 <= t }
            let order = (0 ..< nb).filter { vis[$0] }.sorted { row[$0] > row[$1] }
            let chosen = Set(order.prefix(topk))
            scores.append(row)
            visible.append(vis)
            selected.append((0 ..< nb).map { chosen.contains($0) })
        }
        return Expected(scores: scores, selected: selected, visible: visible)
    }

    static func make(topk: Int, tokens: Int) throws -> (DeepseekV4Attention, MLXArray) {
        let cfg = try ConformanceV4CompressorTests.attentionConfig(ratios: [0, 4, 128, 0], indexTopk: topk)
        let attn = DeepseekV4Attention(config: cfg, layerIndex: 1)
        ConformanceV4CompressorTests.randomize(attn, seed: 17, scale: 0.4, skip: ["attn_sink"])
        let x = MLXRandom.normal([1, tokens, cfg.hiddenSize], key: MLXRandom.key(31))
        return (attn, x)
    }

    @Test("ratio-4 layers own an indexer with the checkpoint's parameter layout")
    func indexerModule() throws {
        let (attn, _) = try Self.make(topk: 2, tokens: 8)
        let ix = try #require(attn.indexer)
        let keys = Set(ix.parameters().flattened().map { $0.0 })
        #expect(keys == ["wq_b.weight", "weights_proj.weight", "compressor.ape", "compressor.wkv.weight",
                         "compressor.wgate.weight", "compressor.norm.weight"])
        #expect(ix.wqB.weight.shape == [2 * 8, 16] && ix.weightsProj.weight.shape == [2, 32])
        #expect(ix.compressor.ape.shape == [4, 16])
        // ratio 128 and ratio 0 layers have none.
        let cfg = attn.config
        #expect(DeepseekV4Attention(config: cfg, layerIndex: 2).indexer == nil)
        #expect(DeepseekV4Attention(config: cfg, layerIndex: 0).indexer == nil)
    }

    @Test("select() picks the oracle's top-k visible blocks per query")
    func selectMatchesOracle() throws {
        let topk = 2
        let (attn, x) = try Self.make(topk: topk, tokens: 24)  // 6 blocks
        let ix = try #require(attn.indexer)
        let e = try Self.expected(attn: attn, x: x, topk: topk)
        let sel = ix.select(x, qr: attn.qNorm(attn.wqA(x)), rope: attn.rope)
        eval(sel)
        #expect(sel.shape == [1, 24, 6])
        let got = sel[0].asArray(Bool.self)
        var wrong = 0, excluded = 0
        for t in 0 ..< 24 {
            for b in 0 ..< 6 {
                if got[t * 6 + b] != e.selected[t][b] { wrong += 1 }
                if e.visible[t][b] && !e.selected[t][b] { excluded += 1 }
            }
            let n = (0 ..< 6).filter { got[t * 6 + $0] }.count
            #expect(n == min(topk, e.visible[t].filter { $0 }.count), "row \(t) selects \(n)")
        }
        #expect(wrong == 0, "\(wrong) selection cells differ from the oracle")
        #expect(excluded > 0, "vacuous: top-k never excluded a visible block")
    }

    @Test("ratio-4 attention reads only the indexer-selected compressed entries")
    func attentionUsesSelection() throws {
        let topk = 2
        let (attn, x) = try Self.make(topk: topk, tokens: 24)
        let cfg = attn.config
        let e = try Self.expected(attn: attn, x: x, topk: topk)
        let comp = try #require(attn.compressor)
        let kv = matmul(x[0], comp.wkv.weight.T), score = matmul(x[0], comp.wgate.weight.T)
        let pooled = SharpAIOverlayGapReference.rmsNorm(
            SharpAIOverlayGapReference.softmaxPoolCompress(
                kv: kv, score: score, ape: comp.ape, ratio: 4, overlap: true, headDim: cfg.headDim),
            weight: comp.norm.weight, eps: cfg.rmsNormEps)
        let nope = cfg.nopeHeadDim
        let entries = (0 ..< pooled.dim(0)).map { b -> MLXArray in
            let en = pooled[b ..< (b + 1)]
            let r = attn.rope(en[0..., nope...].reshaped(1, 1, 1, -1), offset: b * 4).reshaped(1, -1)
            return concatenated([en[0..., ..<nope], r], axis: -1)
        }
        let expected = ConformanceV4CompressorTests.denseAttention(
            attn: attn, x: x, compressed: concatenated(entries, axis: 0), ratio: 4,
            select: { t, b in e.selected[t][b] })
        let allVisible = ConformanceV4CompressorTests.denseAttention(
            attn: attn, x: x, compressed: concatenated(entries, axis: 0), ratio: 4, select: { _, _ in true })
        let got = attn(x, mask: .causal, cache: nil)
        eval(expected, got, allVisible)
        let diff = abs(got - expected).max().item(Float.self)
        #expect(diff < 1e-4, "max |diff| = \(diff)")
        #expect(abs(expected - allVisible).max().item(Float.self) > 1e-3, "vacuous: selection does not change the output")
    }
}
