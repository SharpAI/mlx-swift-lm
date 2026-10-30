// DeepSeek-V4 compressor / indexer state for cached decode (SL-35 lane CACHE).
// SharpAI fork overlay: new file, not part of upstream.
//
// A layer with a compressor keeps, next to its ordinary KV cache, one `V4CompressorStateCache` (an
// `AuxiliaryKVCache`), and `SharpAIDeepseekV4Model.newCache` returns the pair as `CacheList(local KV, state)`.
// Layers without a compress ratio keep the plain cache (bit-identical to before).
//
// What is cached, per compressor (the attention compressor, and for ratio-4 layers the indexer's own):
//   * `entries`  the finished compressed entries [B, blocks, headDim] (RMSNorm applied, RoPE applied at
//                position b * ratio), i.e. exactly what the uncached forward recomputes every call;
//   * `tailKV` / `tailScore`  the RAW projections wkv(x) / wgate(x) (before `ape`, before pooling) of the
//                most recent tokens: the pending partial block, plus the previous block that the ratio-4
//                overlap reads through the first channel half, plus one more block so a trim can reopen
//                a finished block. Keeping raw rows (not pooled results) is what makes trim exact.
// Appending chunk x of any length pools every block that x completes, with the same `V4Compressor.pool`
// as the full-sequence forward, so cached prefill / chunked prefill / token-by-token decode reproduce it.
//
// INFERRED (one place to change each): the tail keeps `pending + keepBlocks * ratio` rows with
// keepBlocks = 2 for the overlap (ratio 4) and 1 otherwise (`keepBlocks`); the cache is trimmable only as
// far as that tail allows (`maxTrim`). Batch size 1 only: batch > 1 is untested.
// upstream-watch: DeepseekV4.swift in mlx-swift-lm.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension SharpAIOverlay {

    /// Value-type state of one compressor.
    struct V4CompressorState {
        let ratio: Int
        let overlap: Bool
        var count = 0
        var tailKV: MLXArray?
        var tailScore: MLXArray?
        var entries: MLXArray?

        init(ratio: Int) {
            self.ratio = ratio
            self.overlap = ratio == 4
        }

        var tailRows: Int { tailKV?.dim(1) ?? 0 }
        var blocks: Int { count / ratio }
        var keepBlocks: Int { overlap ? 2 : 1 }

        /// Tail rows needed to describe a sequence of `c` tokens: its pending rows, plus the previous
        /// block when a later block pools through the overlap.
        func rowsNeeded(forCount c: Int) -> Int {
            c % ratio + ((overlap && c / ratio >= 1) ? ratio : 0)
        }

        /// Largest n such that every rewind of 1...n tokens is exact.
        var maxTrim: Int {
            var n = 0
            while n < count, tailRows - (n + 1) >= rowsNeeded(forCount: count - (n + 1)) { n += 1 }
            return n
        }

        mutating func trim(_ n: Int) {
            guard n > 0 else { return }
            let rows = tailRows - n
            tailKV = rows > 0 ? tailKV![0..., ..<rows] : nil
            tailScore = rows > 0 ? tailScore![0..., ..<rows] : nil
            count -= n
            entries = blocks > 0 ? entries![0..., ..<blocks] : nil
        }

        var arrays: [MLXArray] { [tailKV, tailScore, entries].compactMap { $0 } }
    }

    /// Cache entry holding the compressor state (and the indexer's, for ratio-4 layers) of one layer.
    public final class V4CompressorStateCache: BaseKVCache, AuxiliaryKVCache {
        public override static var serializedClassName: String? { "SharpAIOverlay.V4CompressorStateCache" }

        private static let registration: Void = {
            try! registerKVCacheType(V4CompressorStateCache.self)
        }()

        private static let registerCallLock = NSLock()
        private static let registerCallCounter = Counter()
        private final class Counter: @unchecked Sendable { var n = 0 }

        /// Number of `register()` calls so far (tests: proves the model registers eagerly).
        static var registerCalls: Int { registerCallLock.withLock { registerCallCounter.n } }

        /// Register the type for prompt cache restore. `SharpAIDeepseekV4Model.init` calls this, so a saved V4 prompt
        /// cache restores in a process that has built the model but not yet any cache instance; creating an
        /// instance also registers it (backstop for caches built without a model).
        public static func register() {
            registerCallLock.withLock { registerCallCounter.n += 1 }
            _ = registration
        }

        let ratio: Int
        let headDim: Int
        let indexHeadDim: Int?
        var comp: V4CompressorState
        var index: V4CompressorState?

        // Counts read by the `metaState` setter, consumed by the `state` setter.
        private var pendingMeta: [Int]?

        init(ratio: Int, headDim: Int, indexHeadDim: Int?) {
            _ = Self.registration
            self.ratio = ratio
            self.headDim = headDim
            self.indexHeadDim = indexHeadDim
            self.comp = V4CompressorState(ratio: ratio)
            self.index = indexHeadDim == nil ? nil : V4CompressorState(ratio: V4Indexer.ratio)
            super.init()
        }

        /// Call after appending to `comp` / `index`.
        func syncOffset() { offset = comp.count }

        public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
            preconditionFailure("V4CompressorStateCache holds compressor state; it does not store keys and values")
        }

        public override func innerState() -> [MLXArray] { comp.arrays + (index?.arrays ?? []) }

        public override var isTrimmable: Bool { true }

        public override var maxTrimCount: Int { min(comp.maxTrim, index?.maxTrim ?? Int.max) }

        @discardableResult
        public override func trim(_ n: Int) -> Int {
            let t = Swift.min(Swift.max(n, 0), maxTrimCount)
            comp.trim(t)
            index?.trim(t)
            syncOffset()
            return t
        }

        public override func copy() -> any KVCache {
            let c = V4CompressorStateCache(ratio: ratio, headDim: headDim, indexHeadDim: indexHeadDim)
            c.comp = comp
            c.index = index
            c.syncOffset()
            return c
        }

        // MARK: prompt cache files

        /// metaState: [version, ratio, headDim, indexHeadDim (0 = none), count, compEntryBlocks, compTailRows,
        /// indexEntryBlocks, indexTailRows]. state: per compressor tailKV, tailScore, then entries if any.
        public override var metaState: [String] {
            get {
                ["v1", ratio, headDim, indexHeadDim ?? 0, comp.count, comp.entries?.dim(1) ?? 0, comp.tailRows,
                    index?.entries?.dim(1) ?? 0, index?.tailRows ?? 0].map { "\($0)" }
            }
            set {
                pendingMeta = newValue.count == 9 && newValue[0] == "v1" ? newValue.dropFirst().compactMap { Int($0) } : nil
            }
        }

        public override var state: [MLXArray] {
            get { comp.count == 0 ? [] : innerState() }
            set {
                guard let m = pendingMeta, m.count == 8, m[0] == ratio, m[1] == headDim, m[2] == (indexHeadDim ?? 0)
                else { preconditionFailure("V4CompressorStateCache: metaState must be set before state, and match") }
                var i = 0
                func take(_ s: inout V4CompressorState, entries: Int, rows: Int) {
                    guard rows > 0 else { return }
                    s.tailKV = newValue[i]; s.tailScore = newValue[i + 1]; i += 2
                    if entries > 0 { s.entries = newValue[i]; i += 1 }
                }
                comp = V4CompressorState(ratio: ratio)
                comp.count = m[3]
                take(&comp, entries: m[4], rows: m[5])
                if index != nil {
                    index = V4CompressorState(ratio: V4Indexer.ratio)
                    index!.count = m[3]
                    take(&index!, entries: m[6], rows: m[7])
                }
                syncOffset()
            }
        }

        public override static func restore(state: [MLXArray], metaState: [String]) throws -> any KVCache {
            guard metaState.count == 9, metaState[0] == "v1", let m = Optional(metaState.dropFirst().compactMap { Int($0) }),
                m.count == 8, m[0] > 1, m[1] > 0
            else { throw V4CompressorStateCacheError.badMeta(metaState) }
            let cache = V4CompressorStateCache(ratio: m[0], headDim: m[1], indexHeadDim: m[2] > 0 ? m[2] : nil)
            func arrays(entries: Int, rows: Int) -> Int { rows > 0 ? 2 + (entries > 0 ? 1 : 0) : 0 }
            let expected = m[3] == 0 ? 0 : arrays(entries: m[4], rows: m[5]) + (cache.index == nil ? 0 : arrays(entries: m[6], rows: m[7]))
            guard state.count == expected else { throw V4CompressorStateCacheError.badState(expected: expected, got: state.count) }
            cache.metaState = metaState
            if expected > 0 { cache.state = state } else { cache.syncOffset() }
            return cache
        }
    }

    public enum V4CompressorStateCacheError: LocalizedError, Equatable {
        case badMeta([String])
        case badState(expected: Int, got: Int)

        public var errorDescription: String? {
            switch self {
            case .badMeta(let m): "Corrupt V4CompressorStateCache metadata: \(m)"
            case .badState(let e, let g): "V4CompressorStateCache expects \(e) arrays, file holds \(g)"
            }
        }
    }

    // MARK: layer cache plumbing

    private final class FallbackState: @unchecked Sendable {
        var warned = Set<String>()
        var log: [String] = []
    }
    private static let fallbackLock = NSLock()
    private static let fallbackState = FallbackState()
    private static let fallbackLogger = MLXLogger(label: "DeepseekV4")

    /// A compressed layer was handed a cache it cannot keep compressor state in, so it runs local-attention
    /// only. Logs one warning per cause (the cache type) per process and returns the message when it logged,
    /// nil when that cause was already reported.
    ///
    /// Causes: a plain cache (`KVCacheSimple`, `BatchKVCache`, an MTP cache, a prompt-cache file written
    /// before the compressor state existed) has no `V4CompressorStateCache` next to it; or dynamic cache
    /// quantization (`kvBits` / `kvScheme`) turned the local child of the layer's `CacheList` into a
    /// `QuantizedKVCache` (the state child is an `AuxiliaryKVCache` and is skipped by the rewrite). Quantizing
    /// a V4 compressed layer's local keys is not supported: run with unquantized KV, or accept that the
    /// compressed layers lose their compressed entries.
    @discardableResult
    static func v4NoteCacheFallback(layerCache: KVCache?, local: KVCache, hasState: Bool) -> String? {
        let cause: String
        if hasState {
            cause = "quantized local KV cache \(type(of: local))"
        } else {
            cause = "cache \(type(of: layerCache ?? local)) without a V4CompressorStateCache"
        }
        let message =
            "DeepSeek-V4: a layer with compress_ratio > 1 received a \(cause); its compressed attention is skipped "
            + "and it attends over local keys only, so output differs from the uncached forward. "
            + (hasState
                ? "Dynamic KV quantization (kvBits / kvScheme) is not supported for V4 compressed layers; disable it. "
                : "Build caches with SharpAIDeepseekV4Model.newCache(parameters:) and do not reuse prompt-cache files saved without compressor state. ")
        let first = fallbackLock.withLock { fallbackState.warned.insert(cause).inserted }
        guard first else { return nil }
        fallbackLock.withLock { fallbackState.log.append(message) }
        fallbackLogger.warning(message)
        return message
    }

    /// Messages logged by `v4NoteCacheFallback` since the last reset (tests).
    static var v4FallbackWarnings: [String] { fallbackLock.withLock { fallbackState.log } }

    static func v4ResetFallbackWarnings() {
        fallbackLock.withLock {
            fallbackState.warned = []
            fallbackState.log = []
        }
    }

    /// Splits a layer cache into the ordinary KV cache and the compressor state (nil for a plain cache).
    static func v4Split(_ cache: KVCache?) -> (local: KVCache?, state: V4CompressorStateCache?) {
        guard let list = cache as? CacheList else { return (cache, nil) }
        return (list[0], list[1] as? V4CompressorStateCache)
    }

    /// Layer cache for a layer with compress ratio > 1: `CacheList(local KV, compressor state)`.
    static func v4MakeLayerCache(local: KVCache, ratio: Int, headDim: Int, indexHeadDim: Int?) -> KVCache {
        CacheList(local, V4CompressorStateCache(ratio: ratio, headDim: headDim, indexHeadDim: indexHeadDim))
    }
}

extension SharpAIOverlay.V4Compressor {
    /// Appends x [B, L, hidden] (the next L tokens) to `state` and returns all finished entries
    /// [B, blocks, headDim] (nil while no block is complete). The pooling is `pool`, the same code as the
    /// full-sequence forward; only the rows it reads come from the cached tail.
    func append(
        _ x: MLXArray, state s: inout SharpAIOverlay.V4CompressorState, rope: RoPELayer, ropeDim: Int
    ) -> MLXArray? {
        let l = x.dim(1)
        let (kv, sc) = project(x)
        let m = s.tailRows
        let fullKV = s.tailKV.map { concatenated([$0, kv], axis: 1) } ?? kv
        let fullSc = s.tailScore.map { concatenated([$0, sc], axis: 1) } ?? sc
        let newCount = s.count + l
        let j0 = s.count / ratio, j1 = newCount / ratio
        if j1 > j0 {
            let start = m - s.count % ratio  // row of block j0's first token
            let usePrev = overlap && j0 >= 1
            let from = usePrev ? start - ratio : start
            let end = start + (j1 - j0) * ratio
            // same rounding as the full-sequence forward: fp32 pooling and norm, one cast to the model dtype
            var fresh = norm(
                Self.pool(
                    kv: fullKV[0..., from ..< end], score: fullSc[0..., from ..< end], ape: ape, ratio: ratio,
                    overlap: overlap, headDim: headDim)
            ).asType(x.dtype)
            if usePrev { fresh = fresh[0..., 1...] }  // block j0 - 1 was already finished
            fresh = SharpAIOverlay.v4RopeCompressed(
                fresh, rope: rope, ropeDim: ropeDim, ratio: ratio, firstBlock: j0)
            s.entries = s.entries.map { concatenated([$0, fresh], axis: 1) } ?? fresh
        }
        let drop = max(0, fullKV.dim(1) - (newCount % ratio + s.keepBlocks * ratio))
        s.tailKV = fullKV[0..., drop...]
        s.tailScore = fullSc[0..., drop...]
        s.count = newCount
        return s.entries
    }
}
