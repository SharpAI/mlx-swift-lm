// SharpAI fork overlay tests for the minimax_m3_text creator. Not part of upstream.
// Random-weight tests only: they check that the model builds, runs, and that the KV and indexer
// caches agree with a full forward pass. They do not check checkpoint parity.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMinimaxM3TextRegistrationTests: XCTestCase {

    private let tinyJSON = """
        {
          "architectures": ["MiniMaxM3SparseForCausalLM"],
          "model_type": "minimax_m3_text",
          "hidden_size": 64,
          "intermediate_size": 128,
          "num_hidden_layers": 2,
          "num_attention_heads": 8,
          "num_key_value_heads": 2,
          "vocab_size": 1000
        }
        """

    /// Four layers. Layers 0-2 use dense attention and a dense MLP. Layer 3 uses sparse attention
    /// and the MoE block. The sparse layer keeps one block per query, so a 200-token sequence
    /// (two blocks) really exercises block selection.
    private let sparseTinyJSON = """
        {
          "architectures": ["MiniMaxM3SparseForCausalLM"],
          "model_type": "minimax_m3_text",
          "hidden_size": 64,
          "intermediate_size": 32,
          "dense_intermediate_size": 96,
          "shared_intermediate_size": 32,
          "num_hidden_layers": 4,
          "num_attention_heads": 8,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rotary_dim": 8,
          "vocab_size": 128,
          "num_local_experts": 4,
          "num_experts_per_tok": 2,
          "moe_layer_freq": [0, 0, 0, 1],
          "routed_scaling_factor": 2.0,
          "sparse_attention_config": {
            "sparse_attention_freq": [0, 0, 0, 1],
            "sparse_num_index_heads": 2,
            "sparse_index_dim": 8,
            "sparse_topk_blocks": 1,
            "sparse_block_size": 128,
            "sparse_local_block": 1
          }
        }
        """

    private func creator() throws -> (Data) throws -> any LanguageModel {
        try XCTUnwrap(SharpAIMinimaxM3TextOverlay.creators["minimax_m3_text"])
    }

    private func buildModel(_ json: String) throws -> SharpAIMinimaxM3TextModel {
        let model = try creator()(Data(json.utf8))
        return try XCTUnwrap(model as? SharpAIMinimaxM3TextModel)
    }

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMinimaxM3TextOverlay.creators["minimax_m3_text"])
    }

    func testWrongArchitectureFailsAtValidation() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "MiniMaxM3SparseForCausalLM", with: "LlamaForCausalLM")

        XCTAssertThrowsError(try creator()(Data(json.utf8))) { error in
            guard let typed = error as? SharpAIMinimaxM3TextError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testValidConfigBuildsModel() throws {
        let model = try buildModel(tinyJSON)

        XCTAssertEqual(model.kvHeads, [2, 2])
        XCTAssertEqual(model.vocabularySize, 1000)
    }

    func testMoELayersWithoutExpertCountFailAtConstruction() throws {
        let json = sparseTinyJSON.replacingOccurrences(
            of: "\"num_local_experts\": 4,", with: "")

        XCTAssertThrowsError(try creator()(Data(json.utf8))) { error in
            guard let typed = error as? SharpAIMinimaxM3TextError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testTinyForwardPassWithRandomWeights() throws {
        MLXRandom.seed(0)
        let model = try buildModel(sparseTinyJSON)
        let tokens = MLXArray([Int32(1), 5, 9, 2, 7, 3]).reshaped(1, 6)

        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 6, 128])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testCachedTailMatchesFullForwardAcrossKeyBlocks() throws {
        MLXRandom.seed(1)
        let model = try buildModel(sparseTinyJSON)
        let count = 200
        let split = 150
        let ids = MLXArray((0 ..< count).map { Int32(($0 * 7 + 3) % 128) })
            .reshaped(1, count)

        let full = model(ids, cache: nil)
        let cache = model.newCache(parameters: nil)
        _ = model(ids[0..., ..<split], cache: cache)
        let tail = model(ids[0..., split...], cache: cache)

        let expected = full[0..., split..., 0...].asArray(Float.self)
        let actual = tail.asArray(Float.self)
        XCTAssertEqual(actual.count, expected.count)
        let maxDiff = zip(actual, expected).map { abs($0 - $1) }.max() ?? .infinity
        XCTAssertLessThan(maxDiff, 1e-3)
    }
}
