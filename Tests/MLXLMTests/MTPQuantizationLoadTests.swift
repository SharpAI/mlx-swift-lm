// MTPQuantizationLoadTests.swift
// Regression coverage for mlx-swift-lm#56 / SwiftLM#153.
//
// loadWeights's per-layer quantization closure (Libraries/MLXLMCommon/Load.swift)
// keys its override lookup by the Swift module tree's own path, e.g.
// "language_model.mtp.0.proj" for the first (and usually only) MTP head. Real
// checkpoints declare their per-layer quantization overrides the way mlx_lm's
// Python-side quantize tool does — without the array-depth index, e.g.
// "language_model.mtp.proj" — because on disk there is exactly one MTP head and
// the config never carries the Swift side's per-instance array indexing.
//
// Before #56, that mismatch meant an MTP head quantized at a different bit-width
// than the main model (a standard mlx_lm.quantize output shape) silently fell
// through to the main model's default bits instead of its own override, and
// crashed quantized_matmul with a shape mismatch on the first real forward pass —
// reported against a qwen3_5_moe checkpoint under --stream-experts --mtp.
//
// This test reproduces the exact path shape with a minimal synthetic module tree
// (not the full Qwen35 model, which needs a great deal of unrelated weight
// machinery) so it runs in milliseconds and needs no downloaded checkpoint.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

extension MLXTestingSuite {
    @Suite
    struct MTPQuantizationLoadTests {

        /// Mirrors Qwen35MTPLayer's own nesting: the MTP head's quantizable
        /// weight lives one level under the array element itself ("mtp.0.proj",
        /// not just "mtp.0"), which is what makes the checkpoint's depth-free
        /// override key ("mtp.proj") a genuine normalization, not a no-op.
        private final class MTPHead: Module {
            @ModuleInfo(key: "proj") var proj: Linear
            init(dims: Int) {
                _proj.wrappedValue = Linear(dims, dims, bias: false)
            }
        }

        /// Mirrors Qwen35TextModel: a top-level Linear alongside an "mtp" array.
        private final class InnerLanguageModel: Module {
            @ModuleInfo(key: "main") var main: Linear
            @ModuleInfo(key: "mtp") var mtp: [MTPHead]
            init(dims: Int) {
                _main.wrappedValue = Linear(dims, dims, bias: false)
                _mtp.wrappedValue = [MTPHead(dims: dims)]
            }
        }

        /// Mirrors Qwen35Model: wraps everything under "language_model", which is
        /// what puts a leading dot before "mtp" in the flattened path and makes
        /// the ".mtp.<N>." normalization regex in Load.swift actually fire.
        private final class TinyMTPQuantModel: Module, BaseLanguageModel {
            @ModuleInfo(key: "language_model") var languageModel: InnerLanguageModel
            init(dims: Int) {
                _languageModel.wrappedValue = InnerLanguageModel(dims: dims)
            }
        }

        @Test("loadWeights applies the checkpoint's MTP-head quantization override, not the main model's default")
        func testMTPHeadQuantizationOverrideApplies() throws {
            let dims = 64
            let groupSize = 64
            let defaultBits = 4
            let mtpBits = 8

            let model = TinyMTPQuantModel(dims: dims)

            // Real quantized_matmul-shaped arrays via MLX.quantized, so the failure
            // mode under test (bits/scales shape mismatch) is the genuine one, not
            // a hand-rolled approximation of it.
            let (mainWq, mainScales, mainBiases) = MLX.quantized(
                MLXRandom.normal([dims, dims]), groupSize: groupSize, bits: defaultBits)
            let (mtpWq, mtpScales, mtpBiases) = MLX.quantized(
                MLXRandom.normal([dims, dims]), groupSize: groupSize, bits: mtpBits)

            var weights: [String: MLXArray] = [
                "language_model.main.weight": mainWq,
                "language_model.main.scales": mainScales,
                "language_model.mtp.0.proj.weight": mtpWq,
                "language_model.mtp.0.proj.scales": mtpScales,
            ]
            if let mainBiases { weights["language_model.main.biases"] = mainBiases }
            if let mtpBiases { weights["language_model.mtp.0.proj.biases"] = mtpBiases }

            let tempDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("mtp-quant-load-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tempDir) }
            try MLX.save(arrays: weights, url: tempDir.appendingPathComponent("model.safetensors"))

            let perLayerQuantization = BaseConfiguration.PerLayerQuantization(
                quantization: .init(groupSize: groupSize, bits: defaultBits),
                perLayerQuantization: [
                    "language_model.mtp.proj": .quantize(.init(groupSize: groupSize, bits: mtpBits))
                ]
            )

            try loadWeights(
                modelDirectory: tempDir,
                model: model,
                perLayerQuantization: perLayerQuantization
            )

            guard let quantizedMain = model.languageModel.main as? QuantizedLinear else {
                Issue.record("main Linear was not quantized")
                return
            }
            guard let quantizedMTP = model.languageModel.mtp[0].proj as? QuantizedLinear else {
                Issue.record("mtp.0.proj Linear was not quantized")
                return
            }

            #expect(quantizedMain.bits == defaultBits)
            // Before #56 this read `defaultBits` (4) instead of the checkpoint's
            // declared 8-bit override, because "language_model.mtp.0.proj" never
            // matched the override dict's "language_model.mtp.proj" key — the
            // module ended up structured for 4-bit scales while the weights on
            // disk (mtpScales above) were genuinely 8-bit, which is exactly the
            // quantized_matmul shape mismatch reported in SwiftLM#153.
            #expect(quantizedMTP.bits == mtpBits)
        }
    }
}
