import Foundation
import MLX
import MLXLMCommon
import Testing

extension MLXTestingSuite {
    @Suite
    struct KVCacheTests {
    private static let cacheCreators: [@Sendable () -> any KVCache] = [
        { KVCacheSimple() },
        { RotatingKVCache(maxSize: 32) },
        { QuantizedKVCache() },
        { ChunkedKVCache(chunkSize: 16) },
        { ArraysCache(size: 2) },
        { MambaCache() },
    ]

@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheSerialization(creator: (() -> any KVCache)) async throws {
    let cache = (0 ..< 10).map { _ in creator() }
    let keys = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    for item in cache {
        switch item {
        case let arrays as ArraysCache:
            arrays[0] = keys
            arrays[1] = values
        case let quantized as QuantizedKVCache:
            _ = quantized.updateQuantized(keys: keys, values: values)
        default:
            _ = item.update(keys: keys, values: values)
        }
    }

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try savePromptCache(url: url, cache: cache, metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    #expect(cache.count == loadedCache.count)
    for (lhs, rhs) in zip(cache, loadedCache) {
        #expect(type(of: lhs) == type(of: rhs))
        #expect(lhs.metaState == rhs.metaState)
        #expect(lhs.state.count == rhs.state.count)
    }
}

/// Verify that copy() produces an independent cache: same type, same state,
/// but mutating the copy does not affect the original.
@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheCopyIsIndependent(creator: (() -> any KVCache)) async throws {
    let original = creator()

    let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)

    // populate the original
    switch original {
    case let arrays as ArraysCache:
        arrays[0] = keys
        arrays[1] = values
    case let quantized as QuantizedKVCache:
        _ = quantized.updateQuantized(keys: keys, values: values)
    default:
        _ = original.update(keys: keys, values: values)
    }

    let originalOffset = original.offset
    let originalState = original.state
    eval(originalState)
    let originalMeta = original.metaState

    // copy
    let copied = original.copy()

    // same type
    #expect(type(of: original) == type(of: copied))

    // same offset and metadata
    #expect(copied.offset == originalOffset)
    #expect(copied.metaState == originalMeta)

    // same state values
    let copiedState = copied.state
    eval(copiedState)
    #expect(copiedState.count == originalState.count)
    for (origArr, copyArr) in zip(originalState, copiedState) {
        #expect(origArr.shape == copyArr.shape)
        #expect(allClose(origArr, copyArr).item(Bool.self))
    }

    // mutate the copy — push more tokens through it
    let moreKeys = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
    let moreValues = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)

    switch copied {
    case let arrays as ArraysCache:
        // overwrite slot 0 with a different array
        arrays[0] = moreKeys
    case let quantized as QuantizedKVCache:
        _ = quantized.updateQuantized(keys: moreKeys, values: moreValues)
    default:
        _ = copied.update(keys: moreKeys, values: moreValues)
    }

    // original must be unchanged
    #expect(original.offset == originalOffset)
    #expect(original.metaState == originalMeta)
    let currentState = original.state
    eval(currentState)
    #expect(currentState.count == originalState.count)
    for (origArr, savedArr) in zip(currentState, originalState) {
        #expect(origArr.shape == savedArr.shape)
        #expect(allClose(origArr, savedArr).item(Bool.self))
    }
}

/// copy() on an empty (unpopulated) cache must not crash.
@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheCopyOnEmptyCache(creator: (() -> any KVCache)) async throws {
    let empty = creator()
    let copied = empty.copy()

    #expect(type(of: empty) == type(of: copied))
    #expect(copied.offset == 0)
    #expect(copied.state.count == empty.state.count)
}

/// CacheList.copy() produces independent sub-caches.
@Test
func testCacheListCopyIsIndependent() async throws {
    let sub1 = KVCacheSimple()
    let sub2 = RotatingKVCache(maxSize: 32)
    let composite = CacheList(sub1, sub2)

    let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    _ = sub1.update(keys: keys, values: values)
    _ = sub2.update(keys: keys, values: values)

    // snapshot original state — eval to materialize before copy
    let originalState = composite.state
    eval(originalState)
    let originalOffset0 = sub1.offset
    let originalOffset1 = sub2.offset

    let copied = composite.copy()

    #expect(copied is CacheList)
    let copiedState = copied.state
    eval(copiedState)
    #expect(copiedState.count == originalState.count)
    for (orig, copy) in zip(originalState, copiedState) {
        #expect(orig.shape == copy.shape)
        #expect(allClose(orig, copy).item(Bool.self))
    }

    // mutate inside the copy
    let copiedList = copied as! CacheList
    _ = copiedList[0].update(
        keys: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16),
        values: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
    )

    // originals unchanged
    #expect(sub1.offset == originalOffset0)
    #expect(sub2.offset == originalOffset1)
    let currentState = composite.state
    eval(currentState)
    #expect(currentState.count == originalState.count)
    for (orig, saved) in zip(currentState, originalState) {
        #expect(orig.shape == saved.shape)
        #expect(allClose(orig, saved).item(Bool.self))
    }
}

/// Regression test for a silent-data-loss bug: TurboKV's `compressedOffset` used to
/// keep accumulating across evictions even though `self.keys` gets rebuilt into a
/// fresh buffer on every eviction (holding only the hot window, indexed from 0
/// again). That stale, ever-growing offset was then used to slice the *new*
/// buffer on the next eviction, silently skipping whole token windows — they
/// were never compressed into `polarKeys` and were then overwritten, vanishing
/// from history with no error. Feeding four `step`-sized (256-token) chunks
/// through a real cache (hot window at its default of 256, so evictions trigger
/// every other chunk) must yield back exactly the tokens that went in, in order.
@Test
func testTurboKVMultiRoundEvictionPreservesAllTokens() async throws {
    let dim = 128
    let nKVHeads = 4
    let step = 256
    let totalTokens = 1024  // 4 chunks -> 2 eviction rounds with the default hot window

    let cache = KVCacheSimple()
    cache.turboQuantEnabled = true
    cache.turboMinActivationTokens = 0  // activate immediately instead of waiting for 2048 tokens
    cache.step = step
    // turboHotWindowSize left at its production default (256).

    var chunks = [MLXArray]()
    var chunkStart = 0
    var chunkIndex: UInt64 = 0
    while chunkStart < totalTokens {
        let chunkLen = min(step, totalTokens - chunkStart)
        let chunk = MLXRandom.normal(
            [1, nKVHeads, chunkLen, dim], key: MLXRandom.key(chunkIndex)
        ).asType(.float16)
        eval(chunk)
        chunks.append(chunk)
        _ = cache.update(keys: chunk, values: chunk)
        chunkStart += chunkLen
        chunkIndex += 1
    }

    let state = cache.state
    eval(state)

    // The bug manifested as `state[0]` coming back SHORTER than `totalTokens`
    // (an entire step-sized chunk silently dropped) rather than a crash on its
    // own — the crash only appeared downstream, comparing against the true
    // token count. Assert the count directly so this fails clearly either way.
    #expect(state[0].dim(2) == totalTokens)
    #expect(state[1].dim(2) == totalTokens)

    let originalKeys = concatenated(chunks, axis: 2).asType(.float32)
    let reconstructedKeys = state[0].asType(.float32)
    let diff = originalKeys - reconstructedKeys
    let overallMSE = mean(diff * diff).item(Float.self)
    // Per-step-sized-chunk MSE pinpoints WHICH chunk (if any) is still wrong,
    // rather than only an aggregate number.
    var perChunkMSE = [Float]()
    for i in 0 ..< (totalTokens / step) {
        let lo = i * step, hi = lo + step
        let o = originalKeys[.ellipsis, lo ..< hi, 0...]
        let r = reconstructedKeys[.ellipsis, lo ..< hi, 0...]
        let d = o - r
        perChunkMSE.append(mean(d * d).item(Float.self))
    }
    FileHandle.standardError.write(Data((
        "[REGRESSION TEST] overallMSE=\(overallMSE) perChunkMSE=\(perChunkMSE)\n"
    ).utf8))
    // 3-bit TurboQuant compression is lossy by design (see the module's own
    // ~0.03 MSE at this bit depth) — this checks the history is the RIGHT
    // tokens in the RIGHT order, not lossless, hence a loose tolerance.
    #expect(overallMSE < 0.5)
}
}
}
