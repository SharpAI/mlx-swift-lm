//
//  SharpAIMimoV2Sanitize.swift
//  MLXLLM
//
//  Checkpoint key and value rewrites for mimo_v2:
//  1. drop rotary caches, MTP keys (unless retained) and layers past num_hidden_layers;
//  2. dequantize FP8 block weights (weight_scale_inv, 128x128 blocks);
//  3. split the fused self_attn.qkv_proj into q/k/v;
//  4. stack routed experts into switch_mlp.
//  The fused qkv rows are stored as num_key_value_heads chunks of [Q_c | K_c | V_c].
//  Scales are tiled per chunk when their row count is a multiple of the chunk count.
//  UNVERIFIED: no checkpoint index was read. The chunk layout comes from vLLM.
//

import Foundation
import MLX
import MLXLMCommon

enum SharpAIMimoV2Sanitizer {

    static func sanitize(
        _ weights: [String: MLXArray],
        configuration c: SharpAIMimoV2Configuration,
        retainMTP: Bool
    ) throws -> [String: MLXArray] {
        var out = filterLMHeadWeights(from: weights, tiedWordEmbeddings: c.tieWordEmbeddings)
        out = out.filter { key, _ in
            if key.contains("rotary_emb.inv_freq") { return false }
            if key.hasPrefix("model.mtp.") {
                let keepMTP = retainMTP && c.numNextnPredictLayers > 0
                return keepMTP && key.hasPrefix("model.mtp.layers.0.")
            }
            if let index = layerIndex(of: key) { return index < c.hiddenLayers }
            return true
        }

        try dequantizeBlockScales(&out, configuration: c)
        try splitFusedQKV(&out, configuration: c)
        stackExperts(&out, configuration: c)
        return out
    }

    /// Index N for keys starting with `model.layers.N.`.
    static func layerIndex(of key: String) -> Int? {
        let prefix = "model.layers."
        guard key.hasPrefix(prefix) else { return nil }
        return Int(key.dropFirst(prefix.count).prefix { $0.isNumber })
    }

    // MARK: - FP8 block dequantization

    private static func dequantizeBlockScales(
        _ out: inout [String: MLXArray], configuration c: SharpAIMimoV2Configuration
    ) throws {
        var dequantized: [String: MLXArray] = [:]
        for (key, scaleInv) in out where key.hasSuffix(".weight_scale_inv") {
            let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
            guard let weight = out[weightKey] else { continue }
            // A fused qkv weight carries one scale grid per checkpoint chunk.
            let groups = weightKey.hasSuffix("self_attn.qkv_proj.weight") ? c.kvHeads : 1
            dequantized[weightKey] = try dequantizeBlocks(
                weight, scaleInv: scaleInv, groups: groups)
        }
        out = out.filter { !$0.key.hasSuffix(".weight_scale_inv") }
        for (key, value) in dequantized {
            out[key] = value
        }
    }

    /// `groups` row chunks, each with its own grid of 128x128 scale blocks.
    /// Falls back to one grid when the scale rows match a single continuous grid.
    static func dequantizeBlocks(
        _ weight: MLXArray, scaleInv: MLXArray, groups: Int
    ) throws -> MLXArray {
        let block = 128
        let rows = weight.dim(0)
        let cols = weight.dim(1)
        guard groups > 0, rows % groups == 0 else {
            throw SharpAIMimoV2Error.invalidConfiguration(
                "weight rows \(rows) do not split into \(groups) chunks")
        }

        var chunks = groups
        if scaleInv.dim(0) != chunks * ceilDiv(rows / chunks, block) {
            chunks = 1
        }
        let rowsPerChunk = rows / chunks
        let chunkScaleRows = ceilDiv(rowsPerChunk, block)
        let colBlocks = ceilDiv(cols, block)
        guard scaleInv.dim(0) == chunks * chunkScaleRows, scaleInv.dim(1) == colBlocks else {
            throw SharpAIMimoV2Error.invalidConfiguration(
                "scale shape \(scaleInv.shape) does not match weight shape \(weight.shape)")
        }

        // Swift MLX loads F8_E4M3 as uint8. fromFp8 gives the signed values.
        let values: MLXArray =
            weight.dtype == .uint8
            ? MLXFast.fromFp8(weight, dtype: .float32) : weight.asType(.float32)
        var x = values.reshaped([chunks, rowsPerChunk, cols])
        x = padded(
            x,
            widths: [
                .init((0, 0)),
                .init((0, chunkScaleRows * block - rowsPerChunk)),
                .init((0, colBlocks * block - cols)),
            ])
        x = x.reshaped([chunks, chunkScaleRows, block, colBlocks, block])
        let scale = scaleInv.asType(.float32).reshaped([chunks, chunkScaleRows, 1, colBlocks, 1])
        let scaled = (x * scale).reshaped([chunks, chunkScaleRows * block, colBlocks * block])
        return scaled[0..., 0 ..< rowsPerChunk, 0 ..< cols]
            .reshaped([rows, cols])
            .asType(.bfloat16)
    }

    private static func ceilDiv(_ a: Int, _ b: Int) -> Int {
        (a + b - 1) / b
    }

    // MARK: - Fused qkv

    private static func splitFusedQKV(
        _ out: inout [String: MLXArray], configuration c: SharpAIMimoV2Configuration
    ) throws {
        let fusedKeys = out.keys.filter {
            $0.hasSuffix(".self_attn.qkv_proj.weight") || $0.hasSuffix(".self_attn.qkv_proj.bias")
        }
        for key in fusedKeys {
            guard let fused = out.removeValue(forKey: key) else { continue }
            let kind = key.hasSuffix(".weight") ? "weight" : "bias"
            let attnPrefix = String(key.dropLast("qkv_proj.\(kind)".count))

            let swa = key.hasPrefix("model.mtp.") || isSlidingWindowKey(key, c)
            let heads = swa ? c.swaAttentionHeads : c.attentionHeads
            let kvHeads = swa ? c.swaKvHeads : c.kvHeads
            let headDim = swa ? c.swaHeadDim : c.headDim
            let vHeadDim = swa ? c.swaVHeadDim : c.vHeadDim

            // The checkpoint is pre-sharded into num_key_value_heads chunks.
            let chunks = c.kvHeads
            guard heads % chunks == 0, kvHeads % chunks == 0 else {
                throw SharpAIMimoV2Error.invalidConfiguration(
                    "fused qkv_proj at \(attnPrefix) needs head counts divisible by \(chunks)")
            }
            let qRows = heads / chunks * headDim
            let kRows = kvHeads / chunks * headDim
            let vRows = kvHeads / chunks * vHeadDim
            guard fused.dim(0) == chunks * (qRows + kRows + vRows) else {
                throw SharpAIMimoV2Error.invalidConfiguration(
                    "fused qkv_proj at \(attnPrefix) has \(fused.dim(0)) rows, "
                        + "expected \(chunks * (qRows + kRows + vRows))")
            }

            let parts = splitChunks(fused, chunks: chunks, sizes: [qRows, kRows, vRows])
            out["\(attnPrefix)q_proj.\(kind)"] = parts[0]
            out["\(attnPrefix)k_proj.\(kind)"] = parts[1]
            out["\(attnPrefix)v_proj.\(kind)"] = parts[2]
        }
    }

    private static func isSlidingWindowKey(
        _ key: String, _ c: SharpAIMimoV2Configuration
    ) -> Bool {
        guard let index = layerIndex(of: key) else { return false }
        return c.isSlidingWindowLayer(index)
    }

    /// Splits [chunks * sum(sizes), ...] into per-section outputs, each in chunk-major order.
    private static func splitChunks(
        _ tensor: MLXArray, chunks: Int, sizes: [Int]
    ) -> [MLXArray] {
        let total = sizes.reduce(0, +)
        let rest = Array(tensor.shape.dropFirst())
        let grouped = tensor.reshaped([chunks, total] + rest)
        var parts: [MLXArray] = []
        var start = 0
        for size in sizes {
            let section = grouped[0..., start ..< (start + size), .ellipsis]
            parts.append(section.reshaped([chunks * size] + rest))
            start += size
        }
        return parts
    }

    // MARK: - Experts

    private static func stackExperts(
        _ out: inout [String: MLXArray], configuration c: SharpAIMimoV2Configuration
    ) {
        guard let experts = c.nRoutedExperts else { return }
        for layer in 0 ..< c.hiddenLayers where c.isMoELayer(layer) {
            let prefix = "model.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for kind in ["weight", "scales", "biases"] {
                    guard out["\(prefix).experts.0.\(projection).\(kind)"] != nil else { continue }
                    let toJoin = (0 ..< experts).compactMap {
                        out.removeValue(forKey: "\(prefix).experts.\($0).\(projection).\(kind)")
                    }
                    out["\(prefix).switch_mlp.\(projection).\(kind)"] = MLX.stacked(toJoin)
                }
            }
        }
    }
}
