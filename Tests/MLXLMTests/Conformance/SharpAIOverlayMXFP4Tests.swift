// SL-35 step 5: checkpoint-side mapping of DeepSeek-V4 `expert_dtype: "fp4"` experts (MXFP4: E2M1 nibbles,
// E8M0 group scales, group 32) onto MLX's existing mxfp4 quantized layers, and the use of
// `quantization_config` (scale_fmt ue8m0, weight_block_size) by the FP8 dequant in sanitize.
//
// Synthetic packed tensors only (no weights, no network). The oracle is SharpAIOverlayGapReference.mxfp4Dequant,
// itself pinned against MLX dequantized(mode: .mxfp4) in ConformanceGapReferenceTests.
// upstream-watch: DeepseekV4.sanitize in mlx-swift-lm.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXLLM

struct SharpAIOverlayMXFP4Tests {
    typealias Ref = SharpAIOverlayGapReference

    /// packed uint8 [rows, cols/2] and E8M0 uint8 scales [rows, cols/32] (exponents 120...134: finite, comparable).
    static func packed(rows: Int, cols: Int, seed: UInt64) -> (bytes: MLXArray, scales: MLXArray) {
        let bytes = MLXRandom.randInt(low: 0, high: 256, [rows, cols / 2], key: MLXRandom.key(seed)).asType(.uint8)
        let scales = MLXRandom.randInt(low: 120, high: 135, [rows, cols / 32], key: MLXRandom.key(seed + 1000))
            .asType(.uint8)
        eval(bytes, scales)
        return (bytes, scales)
    }

    static func mlxDequant(_ w: MLXArray, _ s: MLXArray) -> MLXArray {
        let out = dequantized(w, scales: s, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4, dtype: .float32)
        eval(out)
        return out
    }

    static func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float { abs(a - b).max().item(Float.self) }

    // MARK: mapping

    @Test("fp4 expert weight/.scale map to uint32 weight + .scales; dequantized() equals the reference exactly")
    func mappingMatchesReference() {
        let (bytes, scales) = Self.packed(rows: 16, cols: 64, seed: 1)
        let norm = MLXArray([Float(1), 2, 3])
        let out = SharpAIOverlay.mapMXFP4Experts(
            [
                "model.layers.1.ffn.experts.0.gate_proj.weight": bytes,
                "model.layers.1.ffn.experts.0.gate_proj.scale": scales,
                "model.layers.1.attn_norm.weight": norm,
            ], expertDtype: "fp4")
        let w = out["model.layers.1.ffn.experts.0.gate_proj.weight"]
        let s = out["model.layers.1.ffn.experts.0.gate_proj.scales"]
        #expect(out["model.layers.1.ffn.experts.0.gate_proj.scale"] == nil)
        #expect(out["model.layers.1.attn_norm.weight"] != nil)
        #expect(w?.dtype == .uint32 && w?.shape == [16, 8])
        #expect(s?.dtype == .uint8 && s?.shape == [16, 2])
        guard let w, let s else { return }
        let got = Self.mlxDequant(w, s)
        let want = Ref.mxfp4Dequant(packed: bytes, scales: scales)
        eval(want)
        #expect(got.shape == [16, 64])
        #expect(Self.maxDiff(got, want) == 0)
    }

    @Test("int8-stored nibbles and scale bytes (safetensors may surface them signed) map to the same result")
    func int8StorageMaps() {
        let (bytes, scales) = Self.packed(rows: 8, cols: 64, seed: 2)
        let out = SharpAIOverlay.mapMXFP4Experts(
            [
                "l.ffn.experts.3.w1.weight": bytes.asType(.int8),
                "l.ffn.experts.3.w1.scale": scales.asType(.int8),
            ], expertDtype: "fp4")
        guard let w = out["l.ffn.experts.3.w1.weight"], let s = out["l.ffn.experts.3.w1.scales"] else {
            Issue.record("mapped keys missing")
            return
        }
        let want = Ref.mxfp4Dequant(packed: bytes, scales: scales)
        eval(want)
        #expect(Self.maxDiff(Self.mlxDequant(w, s), want) == 0)
    }

    @Test("no-op unless expert_dtype is fp4, and only `.ffn.experts.` pairs are touched")
    func gatingAndScope() {
        let (bytes, scales) = Self.packed(rows: 4, cols: 32, seed: 3)
        let input = [
            "l.ffn.experts.0.w1.weight": bytes, "l.ffn.experts.0.w1.scale": scales,
            "l.attn.wq_a.weight": bytes, "l.attn.wq_a.scale": scales,
        ]
        for dtype in [nil, "fp8", "bf16"] as [String?] {
            let out = SharpAIOverlay.mapMXFP4Experts(input, expertDtype: dtype)
            #expect(Set(out.keys) == Set(input.keys), "expert_dtype \(dtype ?? "nil") must not rewrite")
        }
        let out = SharpAIOverlay.mapMXFP4Experts(input, expertDtype: "fp4")
        #expect(out["l.attn.wq_a.scale"] != nil && out["l.attn.wq_a.weight"]?.dtype == .uint8)
        #expect(out["l.ffn.experts.0.w1.scales"] != nil)
    }

    @Test("expert mxfp4 quantization is group 32, 4 bit, mode mxfp4")
    func quantizationParameters() {
        let q = SharpAIOverlay.mxfp4Quantization
        #expect(q.groupSize == 32 && q.bits == 4 && q.mode == .mxfp4)
    }

    @Test("DeepseekV4.sanitize maps fp4 experts and stacks them into switch_mlp")
    func sanitizeStacksFp4Experts() async throws {
        let m = try await ConformanceDeepseekV4.tinyModel()
        #expect(m.args.expertDtype == "fp4")
        var weights = [String: MLXArray]()
        var refs = [MLXArray]()
        for e in 0 ..< 4 {
            let (b, s) = Self.packed(rows: 16, cols: 32, seed: UInt64(10 + e))
            weights["model.layers.1.ffn.experts.\(e).gate_proj.weight"] = b
            weights["model.layers.1.ffn.experts.\(e).gate_proj.scale"] = s
            let r = Ref.mxfp4Dequant(packed: b, scales: s)
            eval(r)
            refs.append(r)
        }
        let out = m.sanitize(weights: weights)
        let w = try #require(out["model.layers.1.ffn.switch_mlp.gate_proj.weight"])
        let s = try #require(out["model.layers.1.ffn.switch_mlp.gate_proj.scales"])
        #expect(w.dtype == .uint32 && w.shape == [4, 16, 4])
        #expect(s.dtype == .uint8 && s.shape == [4, 16, 1])
        #expect(out.keys.allSatisfy { !$0.contains(".experts.") })
        for e in 0 ..< 4 {
            #expect(Self.maxDiff(Self.mlxDequant(w[e], s[e]), refs[e]) == 0, "expert \(e)")
        }
    }

    // MARK: quantization_config

    @Test("quantization_config is read: scale_fmt ue8m0 and weight_block_size")
    func fp8ConfigFromQuantizationConfig() async throws {
        let m = try await ConformanceDeepseekV4.tinyModel()
        let c = SharpAIOverlay.FP8BlockConfig(quantizationConfig: m.args.quantizationConfig)
        #expect(c.blockSize == 128 && c.ue8m0)
        let other = SharpAIOverlay.FP8BlockConfig(quantizationConfig: [
            "scale_fmt": .string("ue8m0"), "weight_block_size": .array([.int(64), .int(64)]),
        ])
        #expect(other.blockSize == 64 && other.ue8m0)
        let none = SharpAIOverlay.FP8BlockConfig(quantizationConfig: nil)
        #expect(none.blockSize == 128 && !none.ue8m0)
    }

    @Test("dequantFP8Blocks honours the block size and decodes ue8m0 byte scales as 2^(b-127)")
    func fp8DequantBlocks() {
        // 4 x 8 weight, block 4: two column blocks, scale bytes 127 (1.0) and 129 (4.0).
        let weight = MLXArray.ones([4, 8])
        let scale = MLXArray([UInt8(127), 129], [1, 2])
        let out = SharpAIOverlay.dequantFP8Blocks(weight: weight, scaleInv: scale, blockSize: 4, ue8m0: true)
        eval(out)
        let v = out.asType(.float32).asArray(Float.self)
        #expect(out.shape == [4, 8])
        for r in 0 ..< 4 {
            for c in 0 ..< 8 { #expect(v[r * 8 + c] == (c < 4 ? 1 : 4)) }
        }
        // float scales are used as they are, with or without ue8m0 set.
        let f = SharpAIOverlay.dequantFP8Blocks(
            weight: weight, scaleInv: MLXArray([Float(0.5), 2], [1, 2]), blockSize: 4, ue8m0: true)
        eval(f)
        #expect(f.asType(.float32).asArray(Float.self)[0] == 0.5)
    }

    @Test("DeepseekV4.sanitize decodes ue8m0 weight_scale_inv bytes (not as the integer 128)")
    func sanitizeDecodesUe8m0() async throws {
        let m = try await ConformanceDeepseekV4.tinyModel()
        let out = m.sanitize(weights: [
            "model.layers.0.attn.wq_a.weight": MLXArray.ones([8, 8]) * 3,
            "model.layers.0.attn.wq_a.weight_scale_inv": MLXArray([UInt8(128)], [1, 1]),
        ])
        let w = try #require(out["model.layers.0.attn.wq_a.weight"])
        eval(w)
        #expect(w.asType(.float32).asArray(Float.self).allSatisfy { $0 == 6 })
    }
}
