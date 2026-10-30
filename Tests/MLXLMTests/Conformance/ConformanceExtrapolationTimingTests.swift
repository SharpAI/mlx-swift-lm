// Conformance harness, SL-29 phase 3: prove the slope-extrapolation METHOD air-gapped.
//
// Build a model from config-only JSON with random weights at N1 < N2 < N3 layers, time prefill and
// decode steps (warm-up, then the median of several iterations), fit  time = a + b * layers  and
// extrapolate to the real layer count read from an offline fixture.
//
// WHAT THIS SUITE CAN AND CANNOT FAIL ON: it exercises the extrapolation METHOD end to end (build models, time
// real forwards, fit, extrapolate). It does NOT assert any performance property: R^2, monotonicity and the
// extrapolated number are print-only, so a slow or noisy machine can never fail it and a perf regression can
// never be caught by it. It fails only on the deterministic facts listed below (layer counts, cache advance,
// positive finite times, a fit existing, extrapolation being exactly the fitted line).
//
// Determinism policy: wall-clock timing is machine-load dependent, so every timing outcome (R^2,
// monotonicity, the extrapolated number) is REPORT-ONLY (a printed `[extrapolation]` line). The hard
// assertions are the deterministic ones: the models really have the requested layer counts, the
// cache advanced as expected (so the timed step did real work), times are finite and positive, the
// fit exists, and the fixture layer count was found. See scripts/conformance/extrapolate/README.md.
//
// upstream-watch: uses only the public registry (`LLMTypeRegistry.shared.createModel`) and
// `LanguageModel.newCache`; if upstream changes either signature, adjust `timeSteps` below.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

private typealias X = SharpAIOverlayExtrapolation

enum ConformanceExtrapolation {
    static let layerPoints = [2, 4, 8]
    static let promptLength = 64
    static let warmup = 3
    static let iterations = 9

    struct Sample {
        var layers: Int
        var prefillSeconds: Double
        var decodeSeconds: Double
        var cacheOffset: Int
    }

    /// Offline fixture config for `model_type`, searched by file name suffix under fixtures/.
    static func fixtureLayers(file: String, modelType: String) throws -> Int {
        let url = ConformanceBundle.root.appendingPathComponent("fixtures/\(file)")
        let data = try Data(contentsOf: url)
        return try #require(X.layerCount(configJSON: data, modelType: modelType))
    }

    /// Builds the model at `layers` layers and returns median prefill / decode step times.
    static func timeSteps(json: String, modelType: String, layers: Int) async throws -> Sample {
        let model = try await ConformanceBuild.llm(json, modelType: modelType)
        let llm = try #require(model as? any LLMModel)
        let tokens = (0..<promptLength).map { $0 % 50 + 1 }
        let prompt = MLXArray(tokens).reshaped(1, tokens.count)

        func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

        // Prefill: fresh cache every iteration.
        var prefill: [Double] = []
        var lastOffset = 0
        for i in 0..<(warmup + iterations) {
            let cache = try llm.newCache(parameters: nil)
            let t0 = now()
            let out = llm(prompt, cache: cache)
            eval(out)
            let dt = now() - t0
            if i >= warmup { prefill.append(dt) }
            lastOffset = cache.first?.offset ?? -1
        }

        // Decode: one token at a time on a prefilled cache.
        let cache = try llm.newCache(parameters: nil)
        eval(llm(prompt, cache: cache))
        var decode: [Double] = []
        let one = MLXArray([7]).reshaped(1, 1)
        for i in 0..<(warmup + iterations) {
            let t0 = now()
            let out = llm(one, cache: cache)
            eval(out)
            let dt = now() - t0
            if i >= warmup { decode.append(dt) }
        }
        let expectedOffset = promptLength + warmup + iterations
        return Sample(
            layers: layers, prefillSeconds: X.median(prefill), decodeSeconds: X.median(decode),
            cacheOffset: lastOffset == promptLength && (cache.first?.offset ?? -1) == expectedOffset
                ? expectedOffset : -1)
    }

    /// Runs the method for one family and prints the report line. Returns the deterministic facts.
    static func run(
        family: String, modelType: String, jsonForLayers: (Int) -> String, targetLayers: Int
    ) async throws {
        var samples: [Sample] = []
        for n in layerPoints {
            let json = jsonForLayers(n)
            // Deterministic: the config really yields n layers.
            #expect(X.layerCount(configJSON: Data(json.utf8), modelType: modelType) == n)
            samples.append(try await timeSteps(json: json, modelType: modelType, layers: n))
        }
        let xs = samples.map { Double($0.layers) }
        for s in samples {
            // Deterministic: the timed steps advanced the KV cache (real work happened).
            #expect(s.cacheOffset == promptLength + warmup + iterations, "cache did not advance at \(s.layers) layers")
            #expect(s.prefillSeconds.isFinite && s.prefillSeconds > 0)
            #expect(s.decodeSeconds.isFinite && s.decodeSeconds > 0)
        }
        for (name, ys) in [
            ("prefill", samples.map { $0.prefillSeconds }), ("decode", samples.map { $0.decodeSeconds }),
        ] {
            let fit = try #require(X.fit(layers: xs, times: ys))
            let mono = X.isMonotonicIncreasing(layers: xs, times: ys)
            let predicted = X.extrapolate(fit, toLayers: targetLayers)
            let ms = ys.map { String(format: "%.2f", $0 * 1000) }.joined(separator: ", ")
            // REPORT-ONLY: timing outcomes depend on machine load.
            print(
                "[extrapolation] \(family) \(name): layers=\(layerPoints) median_ms=[\(ms)] "
                    + String(format: "slope_ms/layer=%.3f intercept_ms=%.3f R2=%.3f ", fit.slope * 1000, fit.intercept * 1000, fit.r2)
                    + "monotonic=\(mono) target_layers=\(targetLayers) predicted_ms=\(String(format: "%.2f", predicted * 1000)) "
                    + "(R2>=0.9: \(fit.r2 >= 0.9))")
            // Deterministic: predicting at the target is just the line, and is finite.
            #expect(predicted.isFinite)
            #expect(predicted == fit.intercept + fit.slope * Double(targetLayers), "extrapolate() is not the fitted line")
        }
    }
}

struct ConformanceExtrapolationTimingTests {

    static func denseJSON(layers: Int) -> String {
        """
        {"model_type": "qwen3", "hidden_size": 256, "num_hidden_layers": \(layers),
         "intermediate_size": 512, "num_attention_heads": 8, "num_key_value_heads": 4,
         "head_dim": 32, "rms_norm_eps": 1e-6, "vocab_size": 256, "rope_theta": 10000,
         "tie_word_embeddings": false, "max_position_embeddings": 512}
        """
    }

    static func moeJSON(layers: Int) -> String {
        """
        {"model_type": "qwen3_moe", "hidden_size": 128, "num_hidden_layers": \(layers),
         "intermediate_size": 256, "num_attention_heads": 4, "num_key_value_heads": 2,
         "head_dim": 32, "num_experts": 8, "num_experts_per_tok": 2, "decoder_sparse_step": 1,
         "mlp_only_layers": [], "moe_intermediate_size": 64, "rms_norm_eps": 1e-6,
         "vocab_size": 256, "tie_word_embeddings": false, "norm_topk_prob": true}
        """
    }

    @Test("dense (qwen3, llama-like) slope extrapolation to the Qwen3-8B fixture layer count")
    func dense() async throws {
        let target = try ConformanceExtrapolation.fixtureLayers(
            file: "qwen3/Qwen__Qwen3-8B.json", modelType: "qwen3")
        #expect(target == 36)
        try await ConformanceExtrapolation.run(
            family: "dense/qwen3", modelType: "qwen3", jsonForLayers: Self.denseJSON, targetLayers: target)
    }

    @Test("MoE (qwen3_moe) slope extrapolation to the Ovis2.6-30B-A3B llm_config layer count")
    func moe() async throws {
        let target = try ConformanceExtrapolation.fixtureLayers(
            file: "ovis2_6_moe/AIDC-AI__Ovis2.6-30B-A3B.json", modelType: "qwen3_moe")
        #expect(target == 48)
        try await ConformanceExtrapolation.run(
            family: "moe/qwen3_moe", modelType: "qwen3_moe", jsonForLayers: Self.moeJSON, targetLayers: target)
    }
}
