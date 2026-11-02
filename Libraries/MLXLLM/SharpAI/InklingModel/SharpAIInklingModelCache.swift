// SharpAI fork overlay: inkling_model token-history cache. Not part of upstream mlx-swift-lm.
//
// The text backbone carries four short-convolution streams per layer plus a hidden-state
// conditioned relative bias. Rather than port per-layer KV and short-conv state (unverified
// against the reference), generation recomputes the full sequence from this record of the tokens
// processed so far. The result is exactly the full-sequence forward, at O(n^2) cost per request.

import Foundation
import MLX
import MLXLMCommon

/// Records the token ids seen so far, shape `[batch, offset]`. Holds no attention state.
final class SharpAIInklingTokenHistoryCache: BaseKVCache, AuxiliaryKVCache {
    override class var serializedClassName: String? { "SharpAIInklingTokenHistoryCache" }

    override class func restore(
        state: [MLXArray], metaState: [String]
    ) throws -> any KVCache {
        guard state.count <= 1, metaState == [""] else {
            throw SharpAIInklingModelError.corruptTokenHistoryCache
        }
        let cache = SharpAIInklingTokenHistoryCache()
        if let tokens = state.first {
            cache.state = [tokens]
        }
        return cache
    }

    /// Registers the class once so a saved prompt cache restores in a fresh process.
    static let registration: Void = {
        try! registerKVCacheType(SharpAIInklingTokenHistoryCache.self)
    }()

    private(set) var tokens: MLXArray?

    override init() { super.init() }

    /// Token ids before `inputs`, or `nil` when nothing has been recorded.
    var history: MLXArray? { tokens }

    func append(_ inputs: MLXArray) {
        let ids = inputs.asType(.int32)
        tokens = tokens.map { MLX.concatenated([$0, ids], axis: 1) } ?? ids
        offset = tokens?.dim(1) ?? 0
    }

    override func innerState() -> [MLXArray] {
        [tokens].compactMap { $0 }
    }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        (keys, values)
    }

    override var state: [MLXArray] {
        get { [tokens].compactMap { $0 } }
        set {
            tokens = newValue.first
            offset = tokens?.dim(1) ?? 0
        }
    }

    override var isTrimmable: Bool { true }

    @discardableResult
    override func trim(_ n: Int) -> Int {
        let removed = min(max(n, 0), offset)
        guard removed > 0, let current = tokens else { return 0 }
        let keep = offset - removed
        tokens = keep > 0 ? current[0..., ..<keep] : nil
        offset = keep
        return removed
    }

    override func copy() -> any KVCache {
        let other = SharpAIInklingTokenHistoryCache()
        other.tokens = tokens
        other.offset = offset
        return other
    }
}
