// SharpAI fork overlay tests for step3p5 checkpoint key sanitization. Not part of upstream.
// Offline checks only: synthetic tensors, no checkpoint, no network.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIStep3p5SanitizeTests: XCTestCase {

    private func configuration(_ overrides: [String: Any] = [:]) throws
        -> SharpAIStep3p5Configuration
    {
        try JSONDecoder.json5().decode(
            SharpAIStep3p5Configuration.self,
            from: Data(SharpAIStep3p5TestFixture.json(overrides).utf8))
    }

    func testDropsRotaryBuffersAndMTPLayers() throws {
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": zeros([100, 32]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": zeros([4]),
            "model.layers.3.eh_proj.weight": zeros([32, 64]),
            "model.layers.3.mtp_block.self_attn.q_proj.weight": zeros([32, 32]),
        ]

        let out = SharpAIStep3p5Sanitizer.sanitize(weights, configuration: try configuration())

        XCTAssertNotNil(out["model.embed_tokens.weight"])
        XCTAssertNil(out["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(out["model.layers.3.eh_proj.weight"])
        XCTAssertNil(out["model.layers.3.mtp_block.self_attn.q_proj.weight"])
    }

    func testRenamesShareExpertUnderMoE() throws {
        let weights: [String: MLXArray] = [
            "model.layers.1.share_expert.gate_proj.weight": zeros([16, 32])
        ]

        let out = SharpAIStep3p5Sanitizer.sanitize(weights, configuration: try configuration())

        XCTAssertNil(out["model.layers.1.share_expert.gate_proj.weight"])
        XCTAssertEqual(out["model.layers.1.moe.share_expert.gate_proj.weight"]?.shape, [16, 32])
    }

    func testShareExpertRenameIsIdempotent() throws {
        let weights: [String: MLXArray] = [
            "model.layers.1.moe.share_expert.gate_proj.weight": zeros([16, 32])
        ]

        let out = SharpAIStep3p5Sanitizer.sanitize(weights, configuration: try configuration())

        XCTAssertEqual(out["model.layers.1.moe.share_expert.gate_proj.weight"]?.shape, [16, 32])
        XCTAssertNil(out["model.layers.1.moe.moe.share_expert.gate_proj.weight"])
    }

    func testStacksPerExpertWeightsIntoSwitchMLP() throws {
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 4 {
            weights["model.layers.1.moe.experts.\(expert).gate_proj.weight"] = zeros([16, 32])
        }

        let out = SharpAIStep3p5Sanitizer.sanitize(weights, configuration: try configuration())

        XCTAssertEqual(
            out["model.layers.1.moe.switch_mlp.gate_proj.weight"]?.shape, [4, 16, 32])
        XCTAssertNil(out["model.layers.1.moe.experts.0.gate_proj.weight"])
    }

    func testPackedExpertWeightsAreRenamedWithoutStacking() throws {
        let weights: [String: MLXArray] = [
            "model.layers.2.moe.down_proj.weight": zeros([4, 32, 16])
        ]

        let out = SharpAIStep3p5Sanitizer.sanitize(weights, configuration: try configuration())

        XCTAssertEqual(out["model.layers.2.moe.switch_mlp.down_proj.weight"]?.shape, [4, 32, 16])
        XCTAssertNil(out["model.layers.2.moe.down_proj.weight"])
    }

    func testKeepsRouterBiasAndGateInPlace() throws {
        let weights: [String: MLXArray] = [
            "model.layers.1.moe.router_bias": zeros([4]),
            "model.layers.1.moe.gate.weight": zeros([4, 32]),
        ]

        let out = SharpAIStep3p5Sanitizer.sanitize(weights, configuration: try configuration())

        XCTAssertEqual(out["model.layers.1.moe.router_bias"]?.shape, [4])
        XCTAssertEqual(out["model.layers.1.moe.gate.weight"]?.shape, [4, 32])
    }

    func testDropsVLLMIgnoredExpertBias() throws {
        let weights: [String: MLXArray] = [
            "model.layers.1.moe.expert_bias": zeros([4])
        ]

        let out = SharpAIStep3p5Sanitizer.sanitize(weights, configuration: try configuration())

        XCTAssertTrue(out.isEmpty)
    }

    func testTiedEmbeddingsDropLMHead() throws {
        let weights: [String: MLXArray] = [
            "lm_head.weight": zeros([100, 32]),
            "model.embed_tokens.weight": zeros([100, 32]),
        ]

        let out = SharpAIStep3p5Sanitizer.sanitize(
            weights, configuration: try configuration(["tie_word_embeddings": true]))

        XCTAssertNil(out["lm_head.weight"])
        XCTAssertNotNil(out["model.embed_tokens.weight"])
    }
}
