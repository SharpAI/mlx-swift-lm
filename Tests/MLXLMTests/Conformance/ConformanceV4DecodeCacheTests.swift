// SL-35 lane CACHE: the V4 compressor / indexer state cache. Batch size 1 only; batch > 1 is untested.
//
// Reference = the full-sequence forward without a cache (the path the compressor / indexer oracle tests
// cover). Every cached run must reproduce its logits within 1e-4.
//
// Tiny models: layer 0 plain (ratio 0), layer 1 ratio 4 (overlap + indexer, topk 4), layer 2 ratio
// `plainRatio` (no overlap). T = 40 crosses 10 block boundaries at ratio 4 and 5 at ratio 8, and the
// indexer has more candidate blocks than topk, so the selection really prunes.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

enum ConformanceV4DecodeCache {
    static func model(ratios: [Int]) async throws -> DeepseekV4Model {
        var o = try JSONSerialization.jsonObject(with: ConformanceDeepseekV4.tinyJSON()) as! [String: Any]
        o["compress_ratios"] = ratios
        let data = try JSONSerialization.data(withJSONObject: o)
        let m = try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: "deepseek_v4")
        let model = try #require(m as? DeepseekV4Model)
        eval(model)
        return model
    }

    static func tokens(_ n: Int) -> MLXArray {
        MLXArray((0 ..< n).map { Int32(($0 * 7 + 3) % 64) }).reshaped(1, n)
    }

    /// Offsets of the KV cache and, where present, the compressor state of every layer.
    static func offsets(_ cache: [KVCache]) -> [Int] {
        cache.flatMap { c -> [Int] in
            let (local, state) = SharpAIOverlay.v4Split(c)
            return [local!.offset] + (state.map { [$0.offset] } ?? [])
        }
    }

    static func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        eval(a, b)
        return abs(a - b).max().item(Float.self)
    }

    /// Feed `t` in chunks of the given sizes (the last size repeats); returns logits [1, T, vocab].
    static func run(_ m: DeepseekV4Model, _ t: MLXArray, cache: [KVCache], chunks: [Int]) -> MLXArray {
        var outs: [MLXArray] = []
        var pos = 0, i = 0
        while pos < t.dim(1) {
            let n = min(chunks[min(i, chunks.count - 1)], t.dim(1) - pos)
            let o = m(t[0..., pos ..< (pos + n)], cache: cache)
            eval(o)
            outs.append(o)
            pos += n; i += 1
        }
        return concatenated(outs, axis: 1)
    }
}

struct ConformanceV4DecodeCacheTests {
    static let ratioSets: [[Int]] = [[0, 4, 8, 0], [0, 4, 4, 0], [0, 8, 8, 0]]

    @Test("V4 token-by-token decode with the cache reproduces the full-sequence logits", arguments: ratioSets)
    func tokenByTokenDecode(_ ratios: [Int]) async throws {
        let m = try await ConformanceV4DecodeCache.model(ratios: ratios)
        let t = ConformanceV4DecodeCache.tokens(40)
        let plain = m(t, cache: nil)
        let cached = ConformanceV4DecodeCache.run(m, t, cache: try m.newCache(parameters: nil), chunks: [1])
        #expect(ConformanceV4DecodeCache.maxDiff(plain, cached) < 1e-4, "ratios \(ratios)")
    }

    @Test("V4 prefill chunk followed by decode, and uneven chunks, reproduce the full-sequence logits")
    func chunkedPrefillThenDecode() async throws {
        let m = try await ConformanceV4DecodeCache.model(ratios: [0, 4, 8, 0])
        let t = ConformanceV4DecodeCache.tokens(37)
        let plain = m(t, cache: nil)
        for chunks in [[13, 1], [5, 3, 7, 1], [9]] {
            let cached = ConformanceV4DecodeCache.run(m, t, cache: try m.newCache(parameters: nil), chunks: chunks)
            #expect(ConformanceV4DecodeCache.maxDiff(plain, cached) < 1e-4, "chunks \(chunks)")
        }
    }

    @Test("V4 ratio-128 layer (production non-overlap ratio) decodes like the full sequence")
    func ratio128Decode() async throws {
        let m = try await ConformanceV4DecodeCache.model(ratios: [0, 4, 128, 0])
        let t = ConformanceV4DecodeCache.tokens(140)
        let plain = m(t, cache: nil)
        let cached = ConformanceV4DecodeCache.run(m, t, cache: try m.newCache(parameters: nil), chunks: [100, 1])
        #expect(ConformanceV4DecodeCache.maxDiff(plain, cached) < 1e-4)
    }

    @Test("V4 trim then re-decode equals never having decoded the trimmed tokens")
    func trimRewinds() async throws {
        let m = try await ConformanceV4DecodeCache.model(ratios: [0, 4, 8, 0])
        let t = ConformanceV4DecodeCache.tokens(30)
        let plain = m(t, cache: nil)
        for n in [1, 3, 4, 5, 6] {
            let cache = try m.newCache(parameters: nil)
            _ = ConformanceV4DecodeCache.run(m, t, cache: cache, chunks: [11, 1])
            #expect(cache.allSatisfy { $0.isTrimmable }, "trimmable")
            let limit = cache.map(\.maxTrimCount).min() ?? 0
            #expect(limit >= n, "maxTrimCount \(limit) < \(n)")
            for c in cache.map({ $0.copy() }) {
                let own = c.maxTrimCount
                #expect(c.trim(1000) <= own, "trim beyond maxTrimCount must clamp")
            }
            for c in cache { #expect(c.trim(n) == n) }
            #expect(ConformanceV4DecodeCache.offsets(cache).allSatisfy { $0 == 30 - n }, "offsets after trim")
            let again = ConformanceV4DecodeCache.run(m, t[0..., (30 - n)...], cache: cache, chunks: [2])
            #expect(
                ConformanceV4DecodeCache.maxDiff(plain[0..., (30 - n)...], again) < 1e-4, "trim \(n)")
        }
    }

    @Test("V4 cache entries survive a save/restore round trip through state and metaState")
    func serializationRoundTrip() async throws {
        let m = try await ConformanceV4DecodeCache.model(ratios: [0, 4, 8, 0])
        let t = ConformanceV4DecodeCache.tokens(30)
        let plain = m(t, cache: nil)
        let cache = try m.newCache(parameters: nil)
        _ = ConformanceV4DecodeCache.run(m, t[0..., ..<18], cache: cache, chunks: [18])
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("v4cache-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("c.safetensors")
        try savePromptCache(url: url, cache: cache)
        let (restored, _) = try loadPromptCache(url: url)
        let rest = ConformanceV4DecodeCache.run(m, t[0..., 18...], cache: restored, chunks: [1])
        #expect(ConformanceV4DecodeCache.maxDiff(plain[0..., 18...], rest) < 1e-4)
        // copy() is independent of the original
        let copy = cache.map { $0.copy() }
        _ = ConformanceV4DecodeCache.run(m, t[0..., 18...], cache: copy, chunks: [1])
        #expect(ConformanceV4DecodeCache.offsets(cache).allSatisfy { $0 == 18 })
    }

    @Test("V4 model without compress ratios keeps the plain caches and its outputs are unchanged")
    func noCompressRatiosIsUnchanged() async throws {
        let m = try await ConformanceV4DecodeCache.model(ratios: [0, 0, 0, 0])
        let cache = try m.newCache(parameters: nil)
        #expect(cache.count == 3)
        #expect(cache.allSatisfy { type(of: $0) == KVCacheSimple.self })
        let t = ConformanceV4DecodeCache.tokens(12)
        let a = ConformanceV4DecodeCache.run(m, t, cache: cache, chunks: [7, 1])
        let b = ConformanceV4DecodeCache.run(
            m, t, cache: (0 ..< 3).map { _ in KVCacheSimple() }, chunks: [7, 1])
        #expect(ConformanceV4DecodeCache.maxDiff(a, b) == 0)
    }
}
