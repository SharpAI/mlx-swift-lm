// SL-35 step 4.1: DeepSeek-V4 per-layer KV compressor, numerically checked against the pure-MLX
// oracle `SharpAIOverlayGapReference.softmaxPoolCompress` (+ `rmsNorm`) on tiny shapes.
//
// What is verified against the oracle: the pooling (softmax over the window of score + ape, weighted sum
// of kv, overlap of the previous block through the second channel half for ratio 4) and the RMSNorm.
// What is NOT covered by the oracle and is inferred (see DeepseekV4Compressor.swift): the RoPE position
// given to a compressed entry, the visibility rule and the way compressed entries join the local keys.
// The attention-level test below pins that inferred behaviour against an independent dense computation.
// upstream-watch: DeepseekV4.swift in mlx-swift-lm.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

struct V4CompressorCase: CustomTestStringConvertible, Sendable {
    var ratio: Int
    var tokens: Int
    var testDescription: String { "ratio \(ratio), \(tokens) tokens" }
}

struct ConformanceV4CompressorTests {
    static let hidden = 12
    static let headDim = 6
    static let eps: Float = 1e-6

    static func randomCompressor(ratio: Int, seed: UInt64) -> SharpAIOverlay.V4Compressor {
        let c = SharpAIOverlay.V4Compressor(hiddenSize: hidden, headDim: headDim, ratio: ratio, eps: eps)
        let coff = ratio == 4 ? 2 : 1
        let k = MLXRandom.split(key: MLXRandom.key(seed), into: 4)
        c.update(parameters: ModuleParameters.unflattened([
            ("ape", MLXRandom.normal([ratio, coff * headDim], key: k[0])),
            ("wkv.weight", MLXRandom.normal([coff * headDim, hidden], key: k[1]) * 0.5),
            ("wgate.weight", MLXRandom.normal([coff * headDim, hidden], key: k[2]) * 0.5),
            ("norm.weight", 1 + MLXRandom.normal([headDim], key: k[3]) * 0.1),
        ]))
        eval(c)
        return c
    }

    @Test("compressor matches the softmaxPoolCompress + rmsNorm oracle", arguments: [
        V4CompressorCase(ratio: 4, tokens: 11),  // overlap, 2 complete blocks + 3 leftover tokens
        V4CompressorCase(ratio: 4, tokens: 4),  // overlap, exactly one block (nothing to overlap with)
        V4CompressorCase(ratio: 8, tokens: 19),  // no overlap, 2 blocks + 3 leftover
        V4CompressorCase(ratio: 3, tokens: 9),  // no overlap, 3 blocks, no leftover
    ])
    func matchesOracle(_ c: V4CompressorCase) throws {
        let comp = Self.randomCompressor(ratio: c.ratio, seed: 7)
        let b = 2
        let x = MLXRandom.normal([b, c.tokens, Self.hidden], key: MLXRandom.key(99))
        let out = comp(x)
        eval(out)
        let nb = c.tokens / c.ratio
        #expect(out.shape == [b, nb, Self.headDim])
        for i in 0 ..< b {
            let kv = matmul(x[i], comp.wkv.weight.T)
            let score = matmul(x[i], comp.wgate.weight.T)
            let pooled = SharpAIOverlayGapReference.softmaxPoolCompress(
                kv: kv, score: score, ape: comp.ape, ratio: c.ratio, overlap: c.ratio == 4,
                headDim: Self.headDim)
            let expected = SharpAIOverlayGapReference.rmsNorm(pooled, weight: comp.norm.weight, eps: Self.eps)
            let diff = abs(out[i] - expected).max().item(Float.self)
            #expect(diff < 1e-5, "batch \(i): max |diff| = \(diff)")
            #expect(expected.abs().max().item(Float.self) > 0.01, "oracle output is degenerate")
        }
    }

    @Test("fewer tokens than one block yields zero compressed entries")
    func shorterThanOneBlock() {
        let comp = Self.randomCompressor(ratio: 4, seed: 3)
        let out = comp(MLXRandom.normal([1, 3, Self.hidden], key: MLXRandom.key(1)))
        eval(out)
        #expect(out.shape == [1, 0, Self.headDim])
    }

    @Test("parameter keys follow the checkpoint layout: ape, wkv.weight, wgate.weight, norm.weight")
    func parameterKeys() {
        let comp = Self.randomCompressor(ratio: 4, seed: 3)
        let keys = Set(comp.parameters().flattened().map { $0.0 })
        #expect(keys == ["ape", "wkv.weight", "wgate.weight", "norm.weight"])
        #expect(comp.ape.shape == [4, 2 * Self.headDim])
        #expect(comp.wkv.weight.shape == [2 * Self.headDim, Self.hidden])
    }

    // MARK: attention integration (inferred behaviour, pinned against an independent dense computation)

    static func attentionConfig(ratios: [Int]) throws -> DeepseekV4Configuration {
        var o = try #require(try JSONSerialization.jsonObject(with: ConformanceDeepseekV4.tinyJSON()) as? [String: Any])
        o["compress_ratios"] = ratios
        return try JSONDecoder().decode(DeepseekV4Configuration.self, from: JSONSerialization.data(withJSONObject: o))
    }

    /// Randomise every parameter of `m` (seeded), keeping dtype float32.
    static func randomize(_ m: Module, seed: UInt64, scale: Float = 0.3, skip: Set<String> = []) {
        let flat = m.parameters().flattened().filter { !skip.contains($0.0) }
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: max(flat.count, 2))
        var out: [(String, MLXArray)] = []
        for (i, kv) in flat.enumerated() {
            out.append((kv.0, MLXRandom.normal(kv.1.shape, key: keys[i]) * scale))
        }
        m.update(parameters: ModuleParameters.unflattened(out))
        eval(m)
    }

    /// Dense reference of one attention layer for batch 1, no cache, no sinks. `compressed[b]` (already
    /// normed and roped) is visible to query t iff (b + 1) * ratio - 1 <= t and `select(t, b)`.
    static func denseAttention(
        attn: DeepseekV4Attention, x: MLXArray, compressed: MLXArray, ratio: Int,
        select: (Int, Int) -> Bool
    ) -> MLXArray {
        let L = x.dim(1), nh = attn.numHeads, hd = attn.headDim, nope = attn.nopeHeadDim
        var q = attn.wqB(attn.qNorm(attn.wqA(x))).reshaped(1, L, nh, hd).transposed(0, 2, 1, 3)
        q = q * rsqrt(q.square().mean(axis: -1, keepDims: true) + attn.eps)
        q = concatenated([q[.ellipsis, ..<nope], attn.rope(q[.ellipsis, nope...], offset: 0)], axis: -1)
        let kv = attn.kvNorm(attn.wkv(x)).reshaped(1, L, 1, hd).transposed(0, 2, 1, 3)
        let k = concatenated([kv[.ellipsis, ..<nope], attn.rope(kv[.ellipsis, nope...], offset: 0)], axis: -1)
        var rows: [MLXArray] = []
        for t in 0 ..< L {
            var keys = [k[0, 0, 0 ... t, 0...]]
            for b in 0 ..< compressed.dim(0) where (b + 1) * ratio - 1 <= t && select(t, b) {
                keys.append(compressed[b ..< (b + 1)])
            }
            let kk = concatenated(keys, axis: 0)  // [S, hd]
            let p = softmax(matmul(q[0, 0..., t, 0...], kk.T) * attn.scale, axis: -1)  // [nh, S]
            rows.append(matmul(p, kk))  // [nh, hd]
        }
        let out = stacked(rows, axis: 0).reshaped(1, L, nh * hd)  // [1, L, nh*hd]
        let gf = nh * hd / attn.oGroups
        let pieces = (0 ..< attn.oGroups).map { g in
            matmul(
                out[0..., 0..., (g * gf) ..< ((g + 1) * gf)],
                attn.woA.weight[(g * attn.oLoraRank) ..< ((g + 1) * attn.oLoraRank)].T)
        }
        return attn.woB(concatenated(pieces, axis: -1))
    }

    @Test("ratio-8 attention attends to the local keys plus the visible compressed entries")
    func attentionUsesCompressedEntries() throws {
        let cfg = try Self.attentionConfig(ratios: [0, 8, 128, 0])
        let attn = DeepseekV4Attention(config: cfg, layerIndex: 1)
        Self.randomize(attn, seed: 5, skip: ["attn_sink"])  // zero sink -> no sinks passed to SDPA
        let L = 20
        let x = MLXRandom.normal([1, L, cfg.hiddenSize], key: MLXRandom.key(21))
        let comp = try #require(attn.compressor)
        let kv = matmul(x[0], comp.wkv.weight.T), score = matmul(x[0], comp.wgate.weight.T)
        let pooled = SharpAIOverlayGapReference.rmsNorm(
            SharpAIOverlayGapReference.softmaxPoolCompress(
                kv: kv, score: score, ape: comp.ape, ratio: 8, overlap: false, headDim: cfg.headDim),
            weight: comp.norm.weight, eps: cfg.rmsNormEps)
        // inferred: RoPE position of compressed entry b is its first token, b * ratio.
        let nope = cfg.nopeHeadDim
        let entries = (0 ..< pooled.dim(0)).map { b -> MLXArray in
            let e = pooled[b ..< (b + 1)]
            let r = attn.rope(e[0..., nope...].reshaped(1, 1, 1, -1), offset: b * 8).reshaped(1, -1)
            return concatenated([e[0..., ..<nope], r], axis: -1)
        }
        let expected = Self.denseAttention(
            attn: attn, x: x, compressed: concatenated(entries, axis: 0), ratio: 8, select: { _, _ in true })
        let got = attn(x, mask: .causal, cache: nil)
        eval(expected, got)
        let diff = abs(got - expected).max().item(Float.self)
        #expect(diff < 1e-4, "max |diff| = \(diff)")
        // Non-vacuity: the compressed entries really change the result vs. a layer without compressor.
        let plain = DeepseekV4Attention(config: try Self.attentionConfig(ratios: [0, 0, 0, 0]), layerIndex: 1)
        plain.update(parameters: ModuleParameters.unflattened(attn.parameters().flattened().filter { !$0.0.hasPrefix("compressor.") }))
        let without = plain(x, mask: .causal, cache: nil)
        eval(without)
        #expect(abs(got[0, 7...] - without[0, 7...]).max().item(Float.self) > 1e-3)
        // causal: positions before the first block end (token 7) see no compressed entry at all.
        let early = abs(got[0, ..<7] - without[0, ..<7]).max().item(Float.self)
        let mag = abs(without[0, ..<7]).max().item(Float.self)
        #expect(early < 1e-5 * max(mag, 1), "early positions differ by \(early) (magnitude \(mag))")
    }
}
