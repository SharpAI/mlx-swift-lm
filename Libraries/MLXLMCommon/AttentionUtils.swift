import Foundation
import MLX

// ── TurboKV Telemetry ────────────────────────────────────────────────────────
/// Logs once per process when the compressed-history decode path fires in SDPA.
enum TurboKVTelemetry {
    nonisolated(unsafe) static var didLogDecode = false

    static func logDecodeOnce(compressedTokens: Int, hotTokens: Int, totalSeqLen: Int) {
        guard !didLogDecode else { return }
        didLogDecode = true
        print(String(format: "[TurboKV] 🔓 Decode path ACTIVE — compressed=%d tokens + hot=%d tokens → SDPA seq_len=%d",
                     compressedTokens, hotTokens, totalSeqLen))
    }
}
// ─────────────────────────────────────────────────────────────────────────────

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
/// Fallback for scaledDotProductAttention when running on a CPU device
private func fallbackScaledDotProductAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    // Handle Grouped Query Attention (GQA) / Multi-Query Attention (MQA)
    var k = keys
    var v = values
    let qHeads = queries.dim(1)
    let kHeads = keys.dim(1)
    if qHeads > kHeads {
        let repeats = qHeads / kHeads
        k = MLX.repeated(k, count: repeats, axis: 1)
        v = MLX.repeated(v, count: repeats, axis: 1)
    }

    var scores = (queries * scale).matmul(k.transposed(0, 1, 3, 2))
    if let maskArray = mask.mask {
        scores = scores + maskArray
    }
    let softMaxScores = MLX.softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
    return matmul(softMaxScores, v)
}

public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    let isCPU = Device.defaultDevice().deviceType == .cpu


    guard let cache else {
        if isCPU {
            return fallbackScaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        }
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        if isCPU {
            fatalError("[metal_kernel] Quantized KV Cache partitioning is only supported on GPU.")
        }
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)

        // TurboKV: if this cache has compressed history, decode and prepend it.
        // This makes the full context (compressed history + hot window) visible to SDPA.
        var fullKeys = cachedKeys
        var fullValues = cachedValues
        if let kvCache = cache as? KVCacheSimple,
           let pk = kvCache.polarKeys,
           let pv = kvCache.polarValues,
           kvCache.compressedOffset > 0 {
            // Decode packed uint8 history → fp32 → model dtype
            let historyK = MLXFast.turboDecodeK(packed: pk).asType(cachedKeys.dtype)
            let historyV = MLXFast.turboDecodeV(packed: pv).asType(cachedValues.dtype)
            // Concatenate: [compressed_history | hot_window] along seq axis (dim 2)
            fullKeys   = concatenated([historyK, cachedKeys],   axis: 2)
            fullValues = concatenated([historyV, cachedValues], axis: 2)

            // Log first decode event so operator can confirm the pipeline is live
            TurboKVTelemetry.logDecodeOnce(
                compressedTokens: kvCache.compressedOffset,
                hotTokens: cachedKeys.dim(2),
                totalSeqLen: fullKeys.dim(2)
            )
        }

        if isCPU {
            return fallbackScaledDotProductAttention(
                queries: queries, keys: fullKeys, values: fullValues, scale: scale, mask: mask)
        }
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: fullKeys,
            values: fullValues,
            scale: scale,
            mask: mask
        )
    }
}
