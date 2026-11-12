// SharpAI fork overlay tests for the jina_embeddings_v5 task adapter merge. Not part of upstream.
// Layout and merge rule follow vLLM jina.py (adapters/<task>/, W += (B @ A) * alpha / r).

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIJinaEmbeddingsV5AdapterTests: XCTestCase {

    func testMergeAddsScaledLowRankProduct() throws {
        // r = 1, alpha = 2, so scaling = 2. B @ A = [[1, 0, 0], [0, 0, 0]].
        let a = MLXArray([1.0, 0.0, 0.0] as [Float]).reshaped(1, 3)
        let b = MLXArray([1.0, 0.0] as [Float]).reshaped(2, 1)
        let config = SharpAIJinaEmbeddingsV5Adapter.AdapterConfig(r: 1, loraAlpha: 2)

        let merged = try SharpAIJinaEmbeddingsV5Adapter.merge(
            [
                "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([2, 3])
            ],
            config: config,
            adapterTensors: [
                "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
                "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
            ])

        let result = merged["model.layers.0.self_attn.q_proj.weight"]!
        XCTAssertEqual(result.shape, [2, 3])
        XCTAssertEqual(result.asArray(Float.self), [2, 0, 0, 0, 0, 0])
    }

    func testMergeLeavesUnpairedWeightsUnchanged() throws {
        let config = SharpAIJinaEmbeddingsV5Adapter.AdapterConfig(r: 1, loraAlpha: 2)
        let untouched = MLXArray([3.0, 4.0] as [Float])

        let merged = try SharpAIJinaEmbeddingsV5Adapter.merge(
            [
                "model.norm.weight": untouched
            ],
            config: config,
            adapterTensors: [
                "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": MLXArray.zeros([1, 3])
            ])

        XCTAssertEqual(merged["model.norm.weight"]!.asArray(Float.self), [3, 4])
    }

    func testUnsupportedTaskThrowsBeforeReadingFiles() {
        XCTAssertThrowsError(
            try SharpAIJinaEmbeddingsV5Adapter.applyTaskAdapter(
                root: URL(fileURLWithPath: "/nonexistent-jina-root"),
                task: "not-a-task",
                to: [:])
        ) { error in
            XCTAssertEqual(
                error as? SharpAIJinaEmbeddingsV5Error, .unsupportedTask("not-a-task"))
        }
    }

    // Was a skip ("nano not implemented"). The nano variant is now the bidirectional EuroBERT
    // encoder (is_decoder: false), per modeling_eurobert.py and the nano config.json.
    func testNanoVariantIsBidirectionalEncoder() throws {
        let nanoJSON = """
            {
              "model_type": "jina_embeddings_v5",
              "is_decoder": false,
              "hidden_size": 8,
              "num_hidden_layers": 1,
              "intermediate_size": 16,
              "num_attention_heads": 2,
              "num_key_value_heads": 2,
              "vocab_size": 32,
              "rms_norm_eps": 1e-05,
              "rope_theta": 1000000,
              "rope_scaling": null,
              "tie_word_embeddings": false
            }
            """

        let config = try JSONDecoder.json5().decode(
            SharpAIJinaEmbeddingsV5NanoConfiguration.self, from: Data(nanoJSON.utf8))
        XCTAssertEqual(config.headDim, 4)
        XCTAssertEqual(config.ropeTheta, 1_000_000)
        XCTAssertFalse(config.isDecoder)

        // The creator refuses nano at load time, so a nano checkpoint fails before first use.
        let creator = try XCTUnwrap(SharpAIJinaEmbeddingsV5Overlay.creators["jina_embeddings_v5"])
        XCTAssertThrowsError(try creator(Data(nanoJSON.utf8))) { error in
            XCTAssertEqual(error as? SharpAIJinaEmbeddingsV5Error, .nanoVariantNotWired)
        }

        // The nano class itself is still bidirectional; it is built directly here.
        let model = SharpAIJinaEmbeddingsV5NanoModel(config)

        // Bidirectional: changing the last token must change the output at position 0.
        // A causal mask would leave position 0 unchanged.
        let base = model.callAsFunction(MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3), cache: nil)
        let changed = model.callAsFunction(MLXArray([1, 2, 7] as [Int32]).reshaped(1, 3), cache: nil)
        XCTAssertEqual(base.shape, [1, 3, 8])
        XCTAssertNotEqual(base[0, 0].asArray(Float.self), changed[0, 0].asArray(Float.self))
    }

    func testNanoSanitizeAddsModelPrefixAndDropsHeadAndAdapterKeys() {
        let weights: [String: MLXArray] = [
            "embed_tokens.weight": MLXArray.zeros([4, 2]),
            "layers.0.self_attn.q_proj.weight": MLXArray.zeros([2, 2]),
            "norm.weight": MLXArray.zeros([2]),
            "lm_head.weight": MLXArray.zeros([4, 2]),
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": MLXArray.zeros([1, 2]),
        ]

        let sanitized = SharpAIJinaEmbeddingsV5NanoSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.norm.weight",
            ])
    }

    private func pairTensors() -> [String: MLXArray] {
        [
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": MLXArray.zeros([1, 3]),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": MLXArray.zeros([2, 1]),
        ]
    }

    func testMergeRejectsShapeMismatch() {
        let config = SharpAIJinaEmbeddingsV5Adapter.AdapterConfig(r: 1, loraAlpha: 2)
        XCTAssertThrowsError(
            try SharpAIJinaEmbeddingsV5Adapter.merge(
                ["model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([4, 3])],
                config: config, adapterTensors: pairTensors())
        ) { error in
            XCTAssertEqual(
                error as? SharpAIJinaEmbeddingsV5Error,
                .incompatibleBaseWeight(key: "model.layers.0.self_attn.q_proj.weight"))
        }
    }

    func testMergeRejectsQuantizedBaseWeight() {
        let config = SharpAIJinaEmbeddingsV5Adapter.AdapterConfig(r: 1, loraAlpha: 2)
        let packed = MLXArray.zeros([2, 3], dtype: .uint32)
        XCTAssertThrowsError(
            try SharpAIJinaEmbeddingsV5Adapter.merge(
                ["model.layers.0.self_attn.q_proj.weight": packed],
                config: config, adapterTensors: pairTensors()))
    }

    func testMergeRejectsNonPositiveRank() {
        let config = SharpAIJinaEmbeddingsV5Adapter.AdapterConfig(r: 0, loraAlpha: 2)
        XCTAssertThrowsError(
            try SharpAIJinaEmbeddingsV5Adapter.merge(
                ["model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([2, 3])],
                config: config, adapterTensors: pairTensors())
        ) { error in
            XCTAssertEqual(error as? SharpAIJinaEmbeddingsV5Error, .invalidAdapterRank(0))
        }
    }

    func testErrorsHaveReadableDescriptions() {
        XCTAssertNotNil(SharpAIJinaEmbeddingsV5Error.nanoVariantNotWired.errorDescription)
        XCTAssertTrue(
            (SharpAIJinaEmbeddingsV5Error.noMatchingAdapterPairs(task: "x").errorDescription ?? "")
                .contains("x"))
    }
}
