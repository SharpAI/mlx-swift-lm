// SharpAI fork overlay tests for granitemoe_swa checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIGranitemoeSwaSanitizeTests: XCTestCase {

    /// A `[experts, 2 * intermediate, hidden]` tensor whose values count up from 0.
    private func fusedInputWeight() -> MLXArray {
        // experts 4, 2 * intermediate 16, hidden 16
        MLXArray((0 ..< 1024).map { Float($0) }).reshaped(4, 16, 16)
    }

    func testSplitsFusedExpertInputIntoGateAndUp() throws {
        let config = try SharpAIGranitemoeSwaTestFixtures.configuration()
        let weights: [String: MLXArray] = [
            "model.layers.0.block_sparse_moe.input_linear.weight": fusedInputWeight(),
            "model.layers.0.block_sparse_moe.output_linear.weight": MLXArray.zeros([4, 16, 8]),
        ]

        let result = SharpAIGranitemoeSwaSanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(result["model.layers.0.block_sparse_moe.input_linear.weight"])
        XCTAssertNil(result["model.layers.0.block_sparse_moe.output_linear.weight"])

        let gate = try XCTUnwrap(result["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"])
        let up = try XCTUnwrap(result["model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight"])
        let down = try XCTUnwrap(result["model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"])

        XCTAssertEqual(gate.shape, [4, 8, 16])
        XCTAssertEqual(up.shape, [4, 8, 16])
        XCTAssertEqual(down.shape, [4, 16, 8])

        // Gate takes the first half of rows, up takes the second half.
        // Expert 0, row 0 starts at value 0. Row 8 starts at value 128.
        XCTAssertEqual(gate.reshaped(-1).asArray(Float.self)[0], 0)
        XCTAssertEqual(up.reshaped(-1).asArray(Float.self)[0], 128)
    }

    func testDropsRotaryBuffersAndTiedLMHead() throws {
        let config = try SharpAIGranitemoeSwaTestFixtures.configuration()
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1.0 as Float]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray([1.0 as Float]),
            "lm_head.weight": MLXArray([2.0 as Float]),
        ]

        let result = SharpAIGranitemoeSwaSanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(result["lm_head.weight"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
    }

    func testKeepsLMHeadWhenUntied() throws {
        let config = try SharpAIGranitemoeSwaTestFixtures.configuration(
            overriding: ["tie_word_embeddings": false])
        let weights: [String: MLXArray] = ["lm_head.weight": MLXArray([2.0 as Float])]

        let result = SharpAIGranitemoeSwaSanitizer.sanitize(weights, configuration: config)

        XCTAssertNotNil(result["lm_head.weight"])
    }

    func testKeepsRouterSharedMLPSinksAndNormKeysUnchanged() throws {
        let config = try SharpAIGranitemoeSwaTestFixtures.configuration()
        let names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.sinks",
            "model.layers.0.block_sparse_moe.router.layer.weight",
            "model.layers.0.shared_mlp.input_linear.weight",
            "model.layers.0.shared_mlp.output_linear.weight",
            "model.layers.0.input_layernorm.weight",
            "model.norm.weight",
        ]
        let weights = Dictionary(
            uniqueKeysWithValues: names.map { ($0, MLXArray([1.0 as Float])) })

        let result = SharpAIGranitemoeSwaSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(Set(result.keys), Set(names))
    }
}
