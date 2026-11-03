// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// ArraysCache.metaState / CacheList.metaState setters are assertionFailure traps in Debug and
// silent no-ops in Release. These tests pin the real restore paths so that a regression which
// restores through the setter fails under `swift test -c release` as well.

// Replaces assertionFailure at KVCache.swift:1917 (ArraysCache.metaState setter); Release compiles it out.
@Test func testArraysCacheRestoreReproducesAllMetadataWithoutTheMetaStateSetter() throws {
    let source = ArraysCache(size: 3, leftPadding: [1, 2])
    source.prepare(lengths: [4, 5])
    let a = MLXArray.ones([2, 4], dtype: .float32) * 3.0
    let b = MLXArray.ones([2, 4], dtype: .float32) * 7.0
    source[0] = a
    // slot 1 stays nil: sparse
    source[2] = b

    let saved = source.metaState
    let restored = try #require(
        try ArraysCache.restore(state: source.state, metaState: saved) as? ArraysCache)

    // A setter-based restore would leave these at their defaults in Release.
    #expect(restored.slotCount == 3)
    #expect(restored[0] != nil)
    #expect(restored[1] == nil)
    #expect(restored[2] != nil)
    #expect(allClose(restored[0]!, a).item(Bool.self))
    #expect(allClose(restored[2]!, b).item(Bool.self))
    #expect(restored.leftPaddingValues == [1, 2])
    #expect(restored.lengthsValues == [4, 5])
    #expect(restored.metaState == saved, "restore did not reproduce the saved metaState")
}

// Replaces assertionFailure at KVCache.swift:1917 (ArraysCache.metaState setter, via MambaCache); Release compiles it out.
@Test func testMambaCacheRestoreReproducesMetaStateWithoutTheMetaStateSetter() throws {
    let source = MambaCache(leftPadding: [2, 0])
    source[0] = MLXArray.ones([2, 4], dtype: .float32) * 3.0
    source[1] = MLXArray.ones([2, 4], dtype: .float32) * 4.0

    let restored = try #require(
        try MambaCache.restore(state: source.state, metaState: source.metaState) as? MambaCache)

    #expect(restored.slotCount == 2)
    #expect(restored.leftPaddingValues == [2, 0])
    #expect(restored.metaState == source.metaState)
    #expect(restored.state.count == source.state.count)
}
// Replaces assertionFailure at KVCache.swift:2237 (CacheList.metaState setter); Release compiles it out.
@Test func testCacheListRestoreReproducesChildrenAndMetaStateWithoutTheMetaStateSetter() throws {
    let simple = KVCacheSimple()
    let rotating = RotatingKVCache(maxSize: 32)
    let mamba = MambaCache()
    let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
    _ = simple.update(keys: keys, values: values)
    _ = rotating.update(keys: keys * 2.0, values: values * 2.0)
    mamba[0] = MLXArray.ones([2, 4], dtype: .float32) * 3.0
    mamba[1] = MLXArray.ones([2, 4], dtype: .float32) * 4.0

    let source = CacheList(mamba, simple, rotating)
    let savedMeta = source.metaState
    let savedState = source.state

    let restored = try #require(
        try CacheList.restore(state: savedState, metaState: savedMeta) as? CacheList)

    // childCount is the first element and the three children follow; a setter-based restore
    // would leave the list empty in Release.
    #expect(restored.children.count == 3)
    let m = try #require(restored[0] as? MambaCache)
    let s = try #require(restored[1] as? KVCacheSimple)
    let r = try #require(restored[2] as? RotatingKVCache)
    #expect(m.slotCount == 2)
    #expect(allClose(m[0]!, mamba[0]!).item(Bool.self))
    #expect(allClose(m[1]!, mamba[1]!).item(Bool.self))
    #expect(s.offset == simple.offset)
    #expect(r.metaState == rotating.metaState)
    #expect(restored.metaState == savedMeta, "restore did not reproduce the flattened metaState")
    #expect(restored.state.count == savedState.count)
}

// Replaces assertionFailure at KVCache.swift:2237 (CacheList.metaState setter); Release compiles it out.
@Test func testCacheListFromStateRejectsATruncatedMetaState() {
    // Guards the same restore path: malformed metaState must throw, not fall back to the setter.
    #expect(throws: (any Error).self) {
        _ = try CacheList.fromState(state: [], metaState: ["2", "KVCacheSimple"])
    }
    #expect(throws: (any Error).self) {
        _ = try CacheList.fromState(state: [], metaState: [])
    }
}
