// SharpAI fork overlay tests for minimax_m3_text key remapping. Not part of upstream.
// UNVERIFIED: the M3 key layout is unknown, so the default table is empty and these tests pin the
// pass-through behaviour and the prefix-rewrite helper with a test-only table.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMinimaxM3TextSanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDefaultTableKeepsAllKeys() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.q_proj.weight": vector(0),
            "model.layers.1.block_sparse_moe.gate.weight": vector(1),
            "lm_head.weight": vector(2),
        ]

        let result = SharpAIMinimaxM3TextSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(weights.keys))
    }

    func testDefaultTableKeepsValues() {
        let weights: [String: MLXArray] = [
            "model.norm.weight": vector(5)
        ]

        let result = SharpAIMinimaxM3TextSanitizer.sanitize(weights)

        XCTAssertEqual(result["model.norm.weight"]?.asArray(Float.self), [5, 6])
    }

    func testEmptyInputGivesEmptyOutput() {
        XCTAssertTrue(SharpAIMinimaxM3TextSanitizer.sanitize([:]).isEmpty)
    }

    func testPrefixMappingRewritesMatchingKeysOnly() {
        let weights: [String: MLXArray] = [
            "old.layers.0.weight": vector(1),
            "other.bias": vector(2),
        ]

        let result = SharpAIMinimaxM3TextSanitizer.sanitize(
            weights, mapping: [(from: "old.", to: "model.")])

        XCTAssertEqual(Set(result.keys), ["model.layers.0.weight", "other.bias"])
        XCTAssertEqual(result["model.layers.0.weight"]?.asArray(Float.self), [1, 2])
    }

    func testFirstMatchingPrefixWins() {
        let weights: [String: MLXArray] = [
            "a.b.weight": vector(3)
        ]

        let result = SharpAIMinimaxM3TextSanitizer.sanitize(
            weights, mapping: [(from: "a.", to: "x."), (from: "a.b.", to: "y.")])

        XCTAssertEqual(Set(result.keys), ["x.b.weight"])
    }

    // MARK: - remapCheckpoint

    private func remapConfiguration(tied: Bool = false) throws -> SharpAIMinimaxM3TextConfiguration {
        let json = """
            {
              "architectures": ["MiniMaxM3SparseForCausalLM"],
              "model_type": "minimax_m3_text",
              "hidden_size": 4,
              "intermediate_size": 6,
              "num_hidden_layers": 4,
              "num_attention_heads": 2,
              "num_key_value_heads": 2,
              "vocab_size": 10,
              "num_local_experts": 2,
              "num_experts_per_tok": 1,
              "tie_word_embeddings": \(tied ? "true" : "false")
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIMinimaxM3TextConfiguration.self, from: Data(json.utf8))
    }

    func testFusedRoutedExpertsAreSplitIntoGateAndUp() throws {
        // [E=2, 2I=4, H=3]: the first two rows of each expert are gate, the rest up.
        let fused = MLXArray((0 ..< 24).map { Float($0) }).reshaped(2, 4, 3)
        let weights: [String: MLXArray] = [
            "model.layers.3.mlp.experts.gate_up_proj": fused,
            "model.layers.3.mlp.experts.down_proj": MLXArray.zeros([2, 4, 2]),
        ]

        let result = SharpAIMinimaxM3TextSanitizer.remapCheckpoint(
            weights, configuration: try remapConfiguration())

        XCTAssertEqual(result["model.layers.3.mlp.experts.gate_proj.weight"]?.shape, [2, 2, 3])
        XCTAssertEqual(result["model.layers.3.mlp.experts.up_proj.weight"]?.shape, [2, 2, 3])
        XCTAssertEqual(
            result["model.layers.3.mlp.experts.up_proj.weight"]?.asArray(Float.self).first, 6)
        XCTAssertEqual(result["model.layers.3.mlp.experts.down_proj.weight"]?.shape, [2, 4, 2])
    }

    func testPerExpertTensorsAreStackedUnderTheirFusedName() throws {
        let weights: [String: MLXArray] = [
            "model.layers.3.mlp.experts.0.w1.weight": vector(0),
            "model.layers.3.mlp.experts.1.w1.weight": vector(1),
            "model.layers.3.mlp.experts.0.w2.weight": vector(2),
            "model.layers.3.mlp.experts.1.w2.weight": vector(3),
        ]

        let result = SharpAIMinimaxM3TextSanitizer.remapCheckpoint(
            weights, configuration: try remapConfiguration())

        XCTAssertEqual(result["model.layers.3.mlp.experts.gate_proj.weight"]?.shape, [2, 2])
        XCTAssertEqual(
            result["model.layers.3.mlp.experts.gate_proj.weight"]?.asArray(Float.self),
            [0, 1, 1, 2])
        XCTAssertEqual(result["model.layers.3.mlp.experts.down_proj.weight"]?.shape, [2, 2])
    }

    func testRouterBiasMovesOntoGateAndMTPIsDropped() throws {
        let weights: [String: MLXArray] = [
            "model.layers.3.mlp.experts.e_score_correction_bias": vector(1),
            "model.layers.3.mlp.gate.weight": vector(2),
            "model.layers.4.self_attn.q_proj.weight": vector(3),
            "mtp.0.eh_proj.weight": vector(4),
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(5),
        ]

        let result = SharpAIMinimaxM3TextSanitizer.remapCheckpoint(
            weights, configuration: try remapConfiguration())

        XCTAssertEqual(
            Set(result.keys),
            ["model.layers.3.mlp.gate.e_score_correction_bias", "model.layers.3.mlp.gate.weight"])
    }

    func testTiedLMHeadIsDropped() throws {
        let weights: [String: MLXArray] = [
            "lm_head.weight": vector(0),
            "model.embed_tokens.weight": vector(1),
        ]

        let tiedResult = SharpAIMinimaxM3TextSanitizer.remapCheckpoint(
            weights, configuration: try remapConfiguration(tied: true))
        XCTAssertEqual(Set(tiedResult.keys), ["model.embed_tokens.weight"])

        let untiedResult = SharpAIMinimaxM3TextSanitizer.remapCheckpoint(
            weights, configuration: try remapConfiguration(tied: false))
        XCTAssertEqual(Set(untiedResult.keys), ["lm_head.weight", "model.embed_tokens.weight"])
    }

    func testFusedDenseGateUpIsSplitOnOutputRows() throws {
        // [2I=4, H=2] for a dense MLP: rows 0-1 are gate, rows 2-3 are up.
        let fused = MLXArray((0 ..< 8).map { Float($0) }).reshaped(4, 2)
        let weights: [String: MLXArray] = [
            "model.layers.0.mlp.gate_up_proj.weight": fused
        ]

        let result = SharpAIMinimaxM3TextSanitizer.remapCheckpoint(
            weights, configuration: try remapConfiguration())

        XCTAssertEqual(result["model.layers.0.mlp.gate_proj.weight"]?.shape, [2, 2])
        XCTAssertEqual(
            result["model.layers.0.mlp.up_proj.weight"]?.asArray(Float.self), [4, 5, 6, 7])
    }
}
