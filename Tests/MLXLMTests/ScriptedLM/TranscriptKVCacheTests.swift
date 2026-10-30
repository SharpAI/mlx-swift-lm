// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXScriptedLM
import Testing

/// A custom cache without ``AuxiliaryKVCache``: cache policy treats it as an attention layer.
private final class PlainCustomCache: BaseKVCache {
    override var isTrimmable: Bool { true }
    override func copy() -> any KVCache { PlainCustomCache() }
}

struct TranscriptKVCacheTests {

    static func attention(_ count: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        let entry = MLXArray.zeros([1, 2, count, 64])
        _ = cache.update(keys: entry, values: entry)
        return cache
    }

    /// SL-37 clamp in `CacheList.trim` applies to every trimmable `AuxiliaryKVCache` child, a transcript included:
    /// its `maxTrimCount` is its offset, so a trim beyond the transcript length is clamped to it, every child
    /// rewinds by that same amount, and `trimPromptCache` refuses a request beyond it.
    @Test func `a cache list holding a transcript clamps trims to the transcript length`() throws {
        let kv = Self.attention(8)
        let transcript = TranscriptKVCache(tokens: [1, 2, 3, 4, 5])
        let list = CacheList(kv, transcript)
        list.offset = 5
        #expect(list.maxTrimCount == 5)
        #expect(list.isTrimmable)

        // within the limit: all children follow
        #expect(list.trim(2) == 2)
        #expect(kv.offset == 6)
        #expect(transcript.tokens == [1, 2, 3])
        #expect(list.offset == 3)

        // beyond the limit: clamped to the transcript length, not to the larger KV
        #expect(list.trim(7) == 3)
        #expect(kv.offset == 3, "KV rewound by the same 3 as the transcript")
        #expect(transcript.tokens.isEmpty)
        #expect(list.offset == 0)
        #expect(list.maxTrimCount == 0)
        #expect(list.trim(1) == 0)
        #expect(kv.offset == 3, "nothing left to rewind: untouched")
    }

    @Test func `trimPromptCache refuses beyond a transcript in a cache list and trims within it`() throws {
        let kv = Self.attention(8)
        let transcript = TranscriptKVCache(tokens: [1, 2, 3, 4, 5])
        let cache: [KVCache] = [CacheList(kv, transcript)]
        #expect(!canTrimPromptCache(cache, numTokens: 6))
        // beyond the transcript length the request is refused and nothing changes
        #expect(trimPromptCache(cache, numTokens: 6) == 0)
        #expect(kv.offset == 8)
        #expect(transcript.tokens.count == 5)
        // within it, every child follows
        #expect(canTrimPromptCache(cache, numTokens: 5))
        #expect(trimPromptCache(cache, numTokens: 5) == 5)
        #expect(kv.offset == 3)
        #expect(transcript.tokens.isEmpty)
    }

    @Test func `tokens follow append, trim, and copy`() throws {
        let transcript = TranscriptKVCache(tokens: [1, 2])
        transcript.append([3, 4, 5])
        #expect(transcript.tokens == [1, 2, 3, 4, 5])
        #expect(transcript.offset == 5)

        #expect(transcript.trim(2) == 2)
        #expect(transcript.tokens == [1, 2, 3])
        #expect(transcript.trim(10) == 3)
        #expect(transcript.offset == 0)

        transcript.append([9])
        let copy = try #require(transcript.copy() as? TranscriptKVCache)
        transcript.append([10])
        #expect(copy.tokens == [9])
    }

    @Test func `capacity validation skips the transcript`() throws {
        let parameters = GenerateParameters(maxKVSize: 16)
        let plan = try parameters.kvCachePlan()
        let attention = try makeAttentionKVCache(parameters: parameters)

        _ = try plan.validated([attention, TranscriptKVCache()])
        #expect(throws: (any Error).self) {
            _ = try plan.validated([attention, PlainCustomCache()])
        }
    }

    @Test func `compression leaves the transcript untouched`() {
        let transcript = TranscriptKVCache(tokens: Array(0 ..< 8))
        var cache: [KVCache] = [Self.attention(8), transcript]

        maybeQuantizeKVCache(cache: &cache, kvBits: 4, quantizedKVStart: 0)

        #expect(cache[0] is QuantizedKVCache)
        #expect((cache[1] as? TranscriptKVCache) === transcript)
        #expect(transcript.tokens == Array(0 ..< 8))
    }

    @Test func `status reports and the shared timeline skip the transcript`() {
        // Offsets disagree on purpose: the timeline must come from the attention entry.
        let cache: [KVCache] = [Self.attention(5), TranscriptKVCache(tokens: [1, 2, 3])]

        let status = KVCacheStatus(cache: cache, plan: .disabled, phase: .planned)
        #expect(status.layers.count == 1)

        let storage = KVCacheStorage(cache, plan: .disabled)
        #expect(storage.processedTokenCount == 5)
        #expect(storage.nativeAttentionOffsetsAreAligned)
    }

    @Test func `trimming and copying include the transcript`() throws {
        let transcript = TranscriptKVCache(tokens: [1, 2, 3, 4, 5])
        let storage = KVCacheStorage([Self.attention(5), transcript], plan: .disabled)

        #expect(storage.trim(2) == 2)
        #expect(storage.cache[0].offset == 3)
        #expect(transcript.tokens == [1, 2, 3])

        let copy = try #require(storage.copy().cache[1] as? TranscriptKVCache)
        #expect(copy !== transcript)
        #expect(copy.tokens == [1, 2, 3])
    }

    @Test func `a staged speculative round writes through and trims the transcript back`() throws {
        let attention = Self.attention(3)
        let transcript = TranscriptKVCache(tokens: [1, 2, 3])
        let storage = KVCacheStorage([attention, transcript], plan: .disabled)

        let round = try #require(storage.beginRound(maximumPositions: 4))
        let presented = try #require(round.caches[1] as? TranscriptKVCache)
        #expect(presented === transcript)

        // The verify pass writes 4 positions to every entry; 2 are accepted.
        let entry = MLXArray.zeros([1, 2, 4, 64])
        _ = round.caches[0].update(keys: entry, values: entry)
        presented.append([4, 5, 6, 7])
        let commit = storage.commit(round, retaining: 2)

        #expect(commit.committedPositions == 2)
        #expect(attention.offset == 5)
        #expect(transcript.tokens == [1, 2, 3, 4, 5])
        #expect(storage.processedTokenCount == 5)
    }

    @Test func `a transcript round-trips through a prompt cache file`() throws {
        let url = FileManager.default.temporaryDirectory
            .appending(component: "transcript-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(
            url: url, cache: [Self.attention(3), TranscriptKVCache(tokens: [4, 5, 6])])
        let restored = try loadPromptCacheSnapshot(url: url).cache

        #expect(restored[0].offset == 3)
        let transcript = try #require(restored[1] as? TranscriptKVCache)
        #expect(transcript.tokens == [4, 5, 6])
    }

    @Test func `an empty transcript round-trips`() throws {
        let url = FileManager.default.temporaryDirectory
            .appending(component: "transcript-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [KVCacheSimple(), TranscriptKVCache()])
        let transcript = try #require(
            try loadPromptCacheSnapshot(url: url).cache.last as? TranscriptKVCache)
        #expect(transcript.tokens.isEmpty)
    }
}
