import MLX
import Testing

@testable import MLXLMCommon

/// The rollback checkpoint `MTPTokenIterator` and `SpeculativeTokenIterator` rely on.
struct MambaCacheCheckpointTests {
    @Test func trimRestoresStateAndOffset() {
        let cache = MambaCache()
        cache.state = [MLXArray([1, 2]), MLXArray([3, 4])]
        cache.offset = 5
        cache.checkpoint()
        cache.state = [MLXArray([9, 9]), MLXArray([9, 9])]
        cache.offset = 9
        #expect(cache.trim(2) == 2)
        #expect(cache.offset == 5)
        #expect(cache.state[0].asArray(Int32.self) == [1, 2])
        #expect(cache.state[1].asArray(Int32.self) == [3, 4])
    }

    @Test func anEmptyCacheRollsBackToEmpty() {
        let cache = MambaCache()
        cache.checkpoint()
        #expect(cache.hasRollbackCheckpoint)
        cache.state = [MLXArray([9, 9]), MLXArray([9, 9])]
        #expect(cache.trim(4) == 4)
        #expect(cache.state.isEmpty)
    }

    @Test func trimZeroDropsTheCheckpoint() {
        let cache = MambaCache()
        cache.state = [MLXArray([1, 2]), MLXArray([3, 4])]
        cache.checkpoint()
        #expect(cache.hasRollbackCheckpoint)
        cache.state = [MLXArray([9, 9]), MLXArray([9, 9])]
        #expect(cache.trim(0) == 0)
        #expect(!cache.hasRollbackCheckpoint)
        // A later rewind has nothing stale to restore.
        #expect(cache.trim(3) == 0)
        #expect(cache.state[0].asArray(Int32.self) == [9, 9])
    }
}

/// Hybrid detection and rewind checks must see caches nested in a `CacheList`.
struct CacheListRewindTests {
    @Test func recurrentCachesFindsNestedMambaCaches() {
        let nested = MambaCache()
        let caches: [KVCache] = [KVCacheSimple(), CacheList(nested, KVCacheSimple())]
        let found = recurrentCaches(in: caches)
        #expect(found.count == 1)
        #expect(found.first === nested)
    }

    @Test func aCacheListWithAWrappedRingCanRewind() {
        let ring = RotatingKVCache(maxSize: 4, keep: 0)
        let kv = MLXArray.zeros([1, 1, 8, 4])
        _ = ring.update(keys: kv, values: kv)
        #expect(!ring.isTrimmable)
        #expect(canRewindCacheLayer(CacheList(ring, KVCacheSimple())))
    }
}
