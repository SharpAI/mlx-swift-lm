// SL-35 (lane MXFP4): `expert_dtype == "fp4"` checkpoints must be BUILT as mxfp4 quantized switch_mlp layers by the real
// load path (loadWeights = prepareCheckpoint/sanitize + quantize(model:) + update), not only mapped tensor-wise.
//
// Acceptance: a synthetic fp4 checkpoint (packed E2M1 nibbles + E8M0 scales for the routed experts, written as
// safetensors) is loaded into the tiny V4 model; its logits are compared with a second tiny V4 model loaded from a
// checkpoint whose experts are the SAME values dequantized by SharpAIOverlayGapReference.mxfp4Dequant and stored as
// dense float32 switch_mlp weights.
//
// Tolerance (stated): max |A - B| <= 1e-3 * max(1, max |B|) on the float32 logits. Both sides see identical float32
// expert values (every E2M1 value times a power of two is exact in float32); only the accumulation order of the
// quantized gather matmul differs from the dense one.
//
// Inferred, not verified against a real checkpoint: the tensor names / dtype of the fp4 experts (see
// SharpAIOverlay.mapMXFP4Experts) -- the checkpoint is built from the names in `SharpAIOverlayMXFP4LoadTests.expertName`.
// No weights, no network.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

struct SharpAIOverlayMXFP4LoadTests {
    typealias Ref = SharpAIOverlayGapReference

    static let projs = ["gate_proj", "up_proj", "down_proj"]
    static let layers = 3
    static let experts = 4

    /// The checkpoint-side name of one expert tensor (single place to change if the real names differ).
    static func expertName(layer: Int, expert: Int, proj: String, leaf: String) -> String {
        "model.layers.\(layer).ffn.experts.\(expert).\(proj).\(leaf)"
    }

    /// The tiny V4 config with moe_intermediate_size 32 so every expert matrix is [32, 32] (mxfp4 group size 32 divides
    /// both the input dim of gate/up (hidden 32) and of down (intermediate 32)).
    static func tinyModel() async throws -> DeepseekV4Model {
        var o = try #require(try JSONSerialization.jsonObject(with: ConformanceDeepseekV4.tinyJSON()) as? [String: Any])
        o["moe_intermediate_size"] = 32
        let data = try JSONSerialization.data(withJSONObject: o)
        let m = try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: "deepseek_v4")
        return try #require(m as? DeepseekV4Model)
    }

    static func write(_ arrays: [String: MLXArray]) throws -> URL {
        let dir = URL(filePath: NSTemporaryDirectory()).appending(component: "mxfp4-load-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try save(arrays: arrays, url: dir.appending(component: "model.safetensors"))
        return dir
    }

    /// (fp4 checkpoint, dense checkpoint) sharing every non-expert tensor of `template`.
    static func checkpoints(template: DeepseekV4Model) -> (fp4: [String: MLXArray], dense: [String: MLXArray]) {
        var shared = [String: MLXArray]()
        for (k, v) in template.parameters().flattened() where !k.contains(".ffn.switch_mlp.") { shared[k] = v }
        var fp4 = shared
        var dense = shared
        for l in 0 ..< layers {
            for (pi, proj) in projs.enumerated() {
                var stackedDense = [MLXArray]()
                for e in 0 ..< experts {
                    let (bytes, scales) = SharpAIOverlayMXFP4Tests.packed(
                        rows: 32, cols: 32, seed: UInt64(1000 * l + 100 * pi + e))
                    fp4[expertName(layer: l, expert: e, proj: proj, leaf: "weight")] = bytes
                    fp4[expertName(layer: l, expert: e, proj: proj, leaf: "scale")] = scales
                    let d = Ref.mxfp4Dequant(packed: bytes, scales: scales)
                    eval(d)
                    stackedDense.append(d)
                }
                dense["model.layers.\(l).ffn.switch_mlp.\(proj).weight"] = MLX.stacked(stackedDense)
            }
        }
        return (fp4, dense)
    }

    static func logits(_ m: DeepseekV4Model) -> MLXArray {
        let out = m(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil).asType(.float32)
        eval(out)
        return out
    }

    @Test("a fp4 checkpoint loads through loadWeights as mxfp4 switch_mlp layers whose logits match dense dequantized experts")
    func fp4LoadMatchesDenseDequant() async throws {
        let template = try await Self.tinyModel()
        #expect(template.args.expertDtype == "fp4")
        let (fp4, dense) = Self.checkpoints(template: template)
        let fp4Dir = try Self.write(fp4)
        let denseDir = try Self.write(dense)
        defer {
            try? FileManager.default.removeItem(at: fp4Dir)
            try? FileManager.default.removeItem(at: denseDir)
        }

        let quantizedModel = try await Self.tinyModel()
        try await loadWeights(modelDirectory: fp4Dir, model: quantizedModel)
        let denseModel = try await Self.tinyModel()
        try await loadWeights(modelDirectory: denseDir, model: denseModel)

        // the experts really were built as mxfp4 layers; the dense side stays dense
        let q = Dictionary(uniqueKeysWithValues: quantizedModel.leafModules().flattened())
        let d = Dictionary(uniqueKeysWithValues: denseModel.leafModules().flattened())
        for l in 0 ..< Self.layers {
            for proj in Self.projs {
                let path = "model.layers.\(l).ffn.switch_mlp.\(proj)"
                let layer = try #require(q[path] as? QuantizedSwitchLinear, "\(path) must be quantized")
                #expect(layer.mode == .mxfp4 && layer.bits == 4 && layer.groupSize == 32)
                #expect(!(d[path] is QuantizedSwitchLinear))
            }
        }

        let got = Self.logits(quantizedModel)
        let want = Self.logits(denseModel)
        #expect(got.shape == want.shape)
        let tolerance = 1e-3 * max(1, abs(want).max().item(Float.self))
        let diff = abs(got - want).max().item(Float.self)
        #expect(got.asArray(Float.self).allSatisfy { $0.isFinite })
        #expect(diff <= tolerance, "max |A-B| = \(diff), tolerance \(tolerance)")
    }

    // MARK: the hook alone

    private static func checkpoint(_ weights: [String: MLXArray], quantization: BaseConfiguration.PerLayerQuantization?)
        -> ModelCheckpoint
    {
        var c = ModelCheckpoint(weights: weights)
        c.perLayerQuantization = quantization
        return c
    }

    @Test("applyMXFP4ExpertQuantization adds an mxfp4 entry per switch_mlp path that has uint32 weight + uint8 scales")
    func hookAddsEntries() {
        let w = MLXArray.zeros([4, 8, 4]).asType(.uint32)
        let s = MLXArray.zeros([4, 8, 1]).asType(.uint8)
        let weights = [
            "model.layers.1.ffn.switch_mlp.gate_proj.weight": w,
            "model.layers.1.ffn.switch_mlp.gate_proj.scales": s,
            // an affine-quantized dense layer must not be touched
            "model.layers.1.attn.wq_a.weight": w, "model.layers.1.attn.wq_a.scales": s,
        ]
        let out = SharpAIOverlay.applyMXFP4ExpertQuantization(
            Self.checkpoint(weights, quantization: nil), expertDtype: "fp4")
        let per = out.perLayerQuantization?.perLayerQuantization ?? [:]
        #expect(Set(per.keys) == ["model.layers.1.ffn.switch_mlp.gate_proj"])
        if case .quantize(let q)? = per["model.layers.1.ffn.switch_mlp.gate_proj"] {
            #expect(q == SharpAIOverlay.mxfp4Quantization)
        } else {
            Issue.record("expected a .quantize entry")
        }
        #expect(out.perLayerQuantization?.quantization == nil)
    }

    @Test("applyMXFP4ExpertQuantization keeps the global default and existing entries, and is a no-op unless fp4")
    func hookPreservesAndGates() {
        let w = MLXArray.zeros([4, 8, 4]).asType(.uint32)
        let s = MLXArray.zeros([4, 8, 1]).asType(.uint8)
        let weights = [
            "l.ffn.switch_mlp.up_proj.weight": w, "l.ffn.switch_mlp.up_proj.scales": s,
        ]
        let global = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
        let existing = BaseConfiguration.PerLayerQuantization(
            quantization: global, perLayerQuantization: ["lm_head": .skip])
        let out = SharpAIOverlay.applyMXFP4ExpertQuantization(
            Self.checkpoint(weights, quantization: existing), expertDtype: "fp4")
        #expect(out.perLayerQuantization?.quantization == global)
        #expect(out.perLayerQuantization?.perLayerQuantization.count == 2)
        for dtype in [nil, "fp8"] as [String?] {
            let same = SharpAIOverlay.applyMXFP4ExpertQuantization(
                Self.checkpoint(weights, quantization: nil), expertDtype: dtype)
            #expect(same.perLayerQuantization == nil, "expert_dtype \(dtype ?? "nil") must not add entries")
        }
    }
}
