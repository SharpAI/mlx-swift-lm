import Foundation
import MLX

// ── TurboKV Telemetry ────────────────────────────────────────────────────────
// Feeds C atomics in moe_stream_op.cpp; stats appear in the 10s SSD Stream log:
//   [⚡️ SSD Stream] 7330 MB/s | ... | 🗜 TurboKV 4244t 4.3x
// No per-event prints — zero log spam.

// Direct link to the C atomic recorder in moe_stream_op.cpp
@_silgen_name("mlx_turbo_kv_record")
func _mlxTurboKVRecord(_ tokens: UInt64, _ origBytes: UInt64, _ packedBytes: UInt64)

enum TurboKVTelemetry {
    /// Feed the 10s log aggregator from the cache compression path.
    ///
    /// `keys` and `values` are the packed uint8 arrays just produced by turboQuantEncode.
    /// Shape: [B, nKVH, newTokens, packedDim]
    /// origBytes = B × nKVH × newTokens × headDim × 2B (fp16) × 2 (K+V)
    static func logOnce(compressedOffset: Int, keys: MLXArray, values: MLXArray, headDim: Int) {
        let B         = keys.dim(0)
        let nKVH      = keys.dim(1)
        let newTokens = keys.dim(2)    // only the tokens just encoded, not cumulative
        let packedBytes = UInt64(keys.nbytes + values.nbytes)
        let origBytes   = UInt64(B * nKVH * newTokens * headDim * 2 * 2)  // K+V fp16
        _mlxTurboKVRecord(UInt64(newTokens), origBytes, packedBytes)
    }

    /// Feed the 10s log aggregator from the AttentionUtils decode path.
    static func record(tokens: Int, origBytes: Int, packedBytes: Int) {
        _mlxTurboKVRecord(UInt64(tokens), UInt64(origBytes), UInt64(packedBytes))
    }
}

/// Alias used by KVCache.swift on the compression path.
typealias TurboKVCacheTelemetry = TurboKVTelemetry
// ─────────────────────────────────────────────────────────────────────────────

/// Cache-native attention that can reuse an owner's KV without appending it again.
package protocol SharedAttentionKVCache: KVCache {
    func updateForAttention(keys: MLXArray, values: MLXArray)
    func attend(
        queries: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray
}

/// An owner's attention presentation, valid until its next cache update.
/// Keep this within one forward pass; serialization state is not an attention view.
package enum AttentionKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int, bits: Int, mode: QuantizationMode
    )
    case native(cache: any SharedAttentionKVCache, sequenceLength: Int)

    package var sequenceLength: Int {
        switch self {
        case .regular(let keys, _): keys.dim(2)
        case .quantized(let keys, _, _, _, _): keys.0.dim(2)
        case .native(_, let sequenceLength): sequenceLength
        }
    }

    package static func update(keys: MLXArray, values: MLXArray, cache: KVCache?) -> Self {
        if let cache = cache as? SharedAttentionKVCache {
            cache.updateForAttention(keys: keys, values: values)
            return .native(cache: cache, sequenceLength: cache.offset)
        } else if let cache = cache as? QuantizedKVCacheProtocol {
            let (keys, values) = cache.updateQuantized(keys: keys, values: values)
            return .quantized(
                keys: keys, values: values,
                groupSize: cache.groupSize, bits: cache.bits, mode: cache.mode)
        } else if let cache {
            let (keys, values) = cache.update(keys: keys, values: values)
            return .regular(keys: keys, values: values)
        } else {
            return .regular(keys: keys, values: values)
        }
    }

    package func attend(
        queries: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        switch self {
        case .regular(let keys, let values):
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        case .quantized(let keys, let values, let groupSize, let bits, let mode):
            return quantizedCacheAttention(
                queries: queries, quantizedKeys: keys, quantizedValues: values,
                scale: scale, mask: mask, groupSize: groupSize, bits: bits, mode: mode)
        case .native(let cache, let sequenceLength):
            // Native storage must still describe the owner's presentation.
            precondition(cache.offset == sequenceLength)
            return cache.attend(queries: queries, scale: scale, mask: mask)
        }
    }
}

/// Whether attention can be split around a plain `KVCache.update` call.
///
/// Quantized and TurboQuant caches own their complete attention operation, so
/// compiled model segments must leave those cache routes on the general path.
package func usesPlainAttentionCacheRoute(_ cache: KVCache) -> Bool {
    !(cache is QuantizedKVCacheProtocol) && !(cache is TurboQuantKVCache)
}

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Detects cache-native attention implementations
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
        var safeMask = mask
        let targetS = keys.dim(2)
        if case .array(let customMask) = mask {
            if customMask.dim(-1) != targetS && customMask.dim(-1) > targetS {
                let sliced: MLXArray
                if customMask.ndim == 2 {
                    sliced = customMask[0..., ..<targetS]
                } else if customMask.ndim == 4 {
                    sliced = customMask[0..., 0..., 0..., ..<targetS]
                } else {
                    fatalError("Unsupported mask dimensionality: \(customMask.ndim)")
                }
                safeMask = .array(sliced)
            }
        }

        if isCPU {
            return fallbackScaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: safeMask)
        }
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: safeMask
        )
    }

    if let turboCache = cache as? TurboQuantKVCache {
        let L = queries.dim(2)
        if L > 1 && !turboCache.isCompressed {
            // Prefill (L>1) on a raw cache: plain update + standard SDPA, // zero overhead; compression is deferred to the first decode step.
            let (cachedKeys, cachedValues) = turboCache.update(keys: keys, values: values)
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: cachedKeys, values: cachedValues,
                scale: scale, mask: mask
            )
        }
        // Decode (L=1) or any call once the cache is compressed (speculative
        // verify chunks, multi-turn re-prefill): the compressed path. The raw
        // update() path is invalid after compression, its raw buffers are
        // gone. First decode call triggers compressRawCache().
        return turboCache.compressedAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        )
    } else if let attentionCache = cache as? KVCacheAttentionProtocol {
        return attentionCache.updateAndAttend(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask)
    } else if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        if isCPU {
            fatalError("[metal_kernel] Quantized KV Cache partitioning is only supported on GPU.")
        }
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
            
        let targetS = quantizedKeys.0.dim(2)
        var safeMask = mask
        if case .array(let customMask) = mask {
            if customMask.dim(-1) != targetS && customMask.dim(-1) > targetS {
                let sliced: MLXArray
                if customMask.ndim == 2 {
                    sliced = customMask[0..., ..<targetS]
                } else if customMask.ndim == 4 {
                    sliced = customMask[0..., 0..., 0..., ..<targetS]
                } else {
                    fatalError("Unsupported mask dimensionality: \(customMask.ndim)")
                }
                safeMask = .array(sliced)
            }
        }
        
        return quantizedCacheAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: safeMask,
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
        if let kvCache = cache as? KVCacheSimple, let pk = kvCache.polarKeys,
            let pv = kvCache.polarValues
        {
            // Hot-window design: cachedKeys = fp16 hot window only (self.keys after eviction).
            // polarKeys = compressed older history. They are disjoint — no duplication possible.
            // SDPA sees: [decoded_prior_history | fp16_hot_window]
            let historyK = MLXFast.turboDecodeK(packed: pk).asType(cachedKeys.dtype)
            let historyV = MLXFast.turboDecodeV(packed: pv).asType(cachedValues.dtype)
            // Merge 2×256 virtual heads back if 512-dim split was used
            var mergedK = historyK
            var mergedV = historyV
            if kvCache.turboSplitHeads {
                let B = historyK.dim(0), H2 = historyK.dim(1), T = historyK.dim(2)
                mergedK = historyK.reshaped(B, H2 / 2, T, 512)
                mergedV = historyV.reshaped(B, H2 / 2, T, 512)
            }
            fullKeys   = concatenated([mergedK, cachedKeys],   axis: 2)
            fullValues = concatenated([mergedV, cachedValues], axis: 2)
            // Telemetry fed from KVCache.update() on the compression path.
        }

        let targetS = fullKeys.dim(2)
        var safeMask = mask
        if case .array(let customMask) = mask {
            if customMask.dim(-1) != targetS && customMask.dim(-1) > targetS {
                let sliced: MLXArray
                if customMask.ndim == 2 {
                    sliced = customMask[0..., ..<targetS]
                } else if customMask.ndim == 4 {
                    sliced = customMask[0..., 0..., 0..., ..<targetS]
                } else {
                    fatalError("Unsupported mask dimensionality: \(customMask.ndim)")
                }
                safeMask = .array(sliced)
            }
        }
        
        if isCPU {
            return fallbackScaledDotProductAttention(
                queries: queries, keys: fullKeys, values: fullValues, scale: scale, mask: safeMask)
        }
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: fullKeys,
            values: fullValues,
            scale: scale,
            mask: safeMask
        )
    }
}

/// Attention over a quantized cache's keys and values. Every quantized-cache route
/// goes through here so the fused-versus-quantized decision lives in one place.
package func quantizedCacheAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    groupSize: Int,
    bits: Int,
    mode: QuantizationMode
) -> MLXArray {
    // Scales carry one entry per group, so they recover the value head dim for any mode.
    let valueHeadDim = quantizedValues.1.dim(-1) * groupSize
    if usesDequantizedPrefillAttention(queries: queries, valueHeadDim: valueHeadDim) {
        // Prefill over a quantized cache: dequantize once and run the fused kernel
        // instead of materializing [B, H, L, N] scores through quantized matmuls.
        let keys = dequantized(
            quantizedKeys.0, scales: quantizedKeys.1, biases: quantizedKeys.2,
            groupSize: groupSize, bits: bits, mode: mode, dtype: queries.dtype)
        let values = dequantized(
            quantizedValues.0, scales: quantizedValues.1, biases: quantizedValues.2,
            groupSize: groupSize, bits: bits, mode: mode, dtype: queries.dtype)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }
    return quantizedScaledDotProductAttention(
        queries: queries, quantizedKeys: quantizedKeys, quantizedValues: quantizedValues,
        scale: scale, mask: mask, groupSize: groupSize, bits: bits, mode: mode)
}

/// Query length from which a quantized cache attends through the fused SDPA kernel over
/// dequantized keys and values rather than through quantized matmuls.
///
/// This is a measured policy, not a kernel boundary. On M4 Pro (Qwen3-14B shapes, 4-bit
/// KV, 8k context) the quantized path is 1.45-1.7x slower from 64 queries up and
/// allocates [B, H, L, N] scores; below 32 the two are even, and at one query the
/// quantized path reads a quarter of the bytes. The crossover shifts with the GQA ratio,
/// head dim, cache length, bits, and GPU generation; retune it against new measurements.
package let dequantizedPrefillAttentionMinimumLength = 64

/// Head dims whose fused SDPA kernel MLX compiles for every GPU generation
/// (`mlx/backend/metal/scaled_dot_product_attention.cpp`). MLX routes 192 and 256 to
/// its unfused path, where the quantized matmuls are competitive, so they stay out.
package let dequantizedPrefillAttentionHeadDims: Set<Int> = [64, 72, 80, 96, 128]

/// Whether prefill attention over a quantized cache should dequantize and use the fused
/// kernel. MLA-style caches with narrower values than queries stay on the quantized route.
package func usesDequantizedPrefillAttention(queries: MLXArray, valueHeadDim: Int) -> Bool {
    let headDim = queries.dim(-1)
    return queries.dim(2) >= dequantizedPrefillAttentionMinimumLength
        && headDim == valueHeadDim
        && dequantizedPrefillAttentionHeadDims.contains(headDim)
}
