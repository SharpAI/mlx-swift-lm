// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// Release builds compile out the assert()s in KVCacheStorage (KVCachePlan.swift). These tests pin
// the same invariants with #expect so they also run under `swift test -c release`.

// MARK: - Fixtures



private func kv(_ positions: Range<Int>, headDim: Int = 2) -> (MLXArray, MLXArray) {
    let n = positions.count
    let keys = MLXArray(
        positions.flatMap { Array(repeating: Float($0), count: headDim) }, [1, 1, n, headDim])
    let values = MLXArray(
        positions.flatMap { Array(repeating: Float(-$0), count: headDim) }, [1, 1, n, headDim])
    return (keys, values)
}

/// Gemma-shaped: one global layer and one sliding layer, prefilled to `prefill` tokens.
private func prefilledStorage(window: Int, prefill: Int) -> KVCacheStorage {
    let live = KVCacheStorage(
        [KVCacheSimple(), RotatingKVCache(maxSize: window, keep: 0)], plan: .disabled)
    if prefill > 0 {
        let (k, v) = kv(0 ..< prefill)
        for leaf in live.cache { _ = leaf.update(keys: k, values: v) }
        live.commitProcessedTokens(prefill)
    }
    return live
}

// Replaces assert at KVCachePlan.swift:295 (commit); Release compiles the assert out.
@Test(arguments: [0, 1, 3, 4])
func testCommitLeavesEveryAttentionLeafOnTheTimeline(accepted: Int) throws {
    // prefill 10 > window 8: the ring has wrapped before the round starts.
    let live = prefilledStorage(window: 8, prefill: 10)
    #expect(live.nativeAttentionOffsetsAreAligned)

    let round = try #require(live.beginRound(maximumPositions: 4))
    let (k, v) = kv(10 ..< 14)
    for cache in round.caches { _ = cache.update(keys: k, values: v) }
    let commit = live.commit(round, retaining: accepted)

    #expect(commit.committedPositions == accepted)
    #expect(commit.discardedPositions == 4 - accepted)
    #expect(live.processedTokenCount == 10 + accepted)
    #expect(live.nativeAttentionOffsetsAreAligned)
    for leaf in live.cache {
        #expect(leaf.offset == live.processedTokenCount)
    }
    #expect(live.roundIsOpen == false)
}

// Companion to the KVCachePlan.swift:295 commit check: rollback must also leave offsets aligned.
@Test func testRollbackLeavesEveryAttentionLeafOnTheTimeline() throws {
    let live = prefilledStorage(window: 8, prefill: 5)
    let round = try #require(live.beginRound(maximumPositions: 3))
    let (k, v) = kv(5 ..< 8)
    for cache in round.caches { _ = cache.update(keys: k, values: v) }
    live.rollback(round)

    #expect(live.processedTokenCount == 5)
    #expect(live.nativeAttentionOffsetsAreAligned)
    for leaf in live.cache { #expect(leaf.offset == 5) }
}

// Replaces assert at KVCachePlan.swift:360 (rewindLastRound); Release compiles the assert out.
@Test(arguments: [(window: 8, prefill: 10), (window: 16, prefill: 10)])
func testRewindLeavesEveryAttentionLeafOnTheTimeline(window: Int, prefill: Int) throws {
    // window 8 / prefill 10: the ring has wrapped, so rewind uses a restore point.
    // window 16 / prefill 10: not wrapped, rewind is a plain trim.
    let live = KVCacheStorage([KVCacheSimple(), RotatingKVCache(maxSize: window, keep: 0)], plan: .disabled)
    let (pk, pv) = kv(0 ..< prefill)
    for leaf in live.cache { _ = leaf.update(keys: pk, values: pv) }
    live.commitProcessedTokens(prefill)
    #expect(live.nativeAttentionOffsetsAreAligned)

    let round = try #require(live.beginRound(maximumPositions: 4))
    let (k, v) = kv(prefill ..< (prefill + 4))
    for cache in round.caches { _ = cache.update(keys: k, values: v) }
    _ = live.commit(round, retaining: 4)
    #expect(live.processedTokenCount == prefill + 4)
    #expect(live.nativeAttentionOffsetsAreAligned)

    let rewound = live.rewindLastRound(3)

    #expect(rewound == 3)
    #expect(live.processedTokenCount == prefill + 1)
    #expect(live.nativeAttentionOffsetsAreAligned)
    for leaf in live.cache {
        #expect(leaf.offset == live.processedTokenCount)
    }
}

private func inferenceKV(_ count: Int) -> (MLXArray, MLXArray) {
    let keys = MLXArray(Array(repeating: Float(1), count: count * 2), [1, 1, count, 2])
    return (keys, keys)
}

// Replaces assert at KVCachePlan.swift:401 (attention offsets agree at adoption); Release compiles it out.
@Test func testAdoptionInfersOneSharedTimelineFromAlignedAttentionLeaves() {
    let a = KVCacheSimple()
    let b = RotatingKVCache(maxSize: 16, keep: 0)
    let (k, v) = inferenceKV(7)
    _ = a.update(keys: k, values: v)
    _ = b.update(keys: k, values: v)
    // A recurrent entry with an offset BELOW the attention offset: if the attention filter were
    // dropped, min() would pick 2 and the expectations below would fail.
    let mamba = MambaCache()
    mamba.offset = 2

    let storage = KVCacheStorage([a, mamba, b], plan: .disabled)

    #expect(a.offset == 7 && b.offset == 7)
    #expect(storage.processedTokenCount == 7)
    #expect(storage.nativeAttentionOffsetsAreAligned)
}

#if !DEBUG
    // Release only: in Debug the assert at KVCachePlan.swift:401 traps on divergence.
    // With asserts stripped, divergence must still be observable through the audit.
    // Replaces assert at KVCachePlan.swift:401; Release compiles the assert out.
    @Test func testDivergentAttentionOffsetsAreReportedMisaligned() {
        let a = KVCacheSimple()
        let b = KVCacheSimple()
        let (k, v) = inferenceKV(7)
        let (k3, v3) = inferenceKV(3)
        _ = a.update(keys: k, values: v)
        _ = b.update(keys: k3, values: v3)

        let storage = KVCacheStorage([a, b], plan: .disabled)

        #expect(storage.processedTokenCount == 3)
        #expect(!storage.nativeAttentionOffsetsAreAligned)
    }
#endif

// Replaces assert at KVCachePlan.swift:411 (recurrent-only offsets agree at adoption); Release compiles it out.
@Test func testRecurrentOnlyAdoptionUsesTheSharedLegacyOffset() {
    let first = MambaCache()
    let second = MambaCache()
    first.offset = 4
    second.offset = 4

    let storage = KVCacheStorage([first, second], plan: .disabled)

    #expect(storage.processedTokenCount == 4)
    // Recurrent entries are outside the attention audit, so it is vacuously aligned.
    #expect(storage.nativeAttentionOffsetsAreAligned)
}

@Test func testRecurrentOnlyAdoptionWithFreshStateStartsAtZero() {
    let storage = KVCacheStorage([MambaCache(), MambaCache()], plan: .disabled)
    #expect(storage.processedTokenCount == 0)

    let empty = KVCacheStorage([], plan: .disabled)
    #expect(empty.processedTokenCount == 0)
}

// Distinguishes min from max/first/last: the shared offset is the same whichever is picked,
// so single-value cases alone cannot tell them apart.
@Test func testRecurrentOnlyAdoptionIsIndependentOfLeafOrder() {
    for offsets in [[4, 4, 4], [0, 0], [7]] {
        let caches = offsets.map { value -> MambaCache in
            let cache = MambaCache()
            cache.offset = value
            return cache
        }
        #expect(KVCacheStorage(caches, plan: .disabled).processedTokenCount == offsets[0])
        #expect(
            KVCacheStorage(caches.reversed(), plan: .disabled).processedTokenCount == offsets[0])
    }
}

#if !DEBUG
    // Replaces assert at KVCachePlan.swift:411. Release builds compile the assert out, so a violation is only observable as the fallback
    // value. In Debug the assert traps, so this case cannot run there.
    @Test func testRecurrentOnlyDivergedOffsetsFallBackToTheMinimumInRelease() {
        let low = MambaCache()
        let high = MambaCache()
        low.offset = 3
        high.offset = 9

        #expect(KVCacheStorage([high, low], plan: .disabled).processedTokenCount == 3)
    }
#endif
