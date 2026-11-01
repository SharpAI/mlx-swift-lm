// SharpAI fork overlay tests for step3p7 key sanitization. Not part of upstream.
// Key names follow the step3p5 rules. UNVERIFIED against a real step3p7 checkpoint index.
// Written without compiling.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIStep3p7SanitizeTests: XCTestCase {

    /// Fixture config (4 hidden layers) with a small expert count.
    private func configuration(experts: Int = 2) throws -> SharpAIStep3p7Configuration {
        let json = SharpAIStep3p7Fixtures.fixtureJSON.replacingOccurrences(
            of: "\"moe_num_experts\": 288", with: "\"moe_num_experts\": \(experts)")
        return try JSONDecoder.json5().decode(
            SharpAIStep3p7Configuration.self, from: Data(json.utf8))
    }

    func testDropsRotaryExpertBiasAndMTPLayers() throws {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "model.layers.1.moe.expert_bias": MLXArray.zeros([2]),
            "model.layers.4.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
        ]

        let sanitized = SharpAIStep3p7Sanitizer.sanitize(
            weights, configuration: try configuration())

        XCTAssertTrue(sanitized.isEmpty)
    }

    func testRenamesShareExpertIntoMoE() throws {
        let weights: [String: MLXArray] = [
            "model.layers.1.share_expert.gate_proj.weight": MLXArray.zeros([16, 8])
        ]

        let sanitized = SharpAIStep3p7Sanitizer.sanitize(
            weights, configuration: try configuration())

        XCTAssertEqual(Set(sanitized.keys), Set(["model.layers.1.moe.share_expert.gate_proj.weight"]))
    }

    func testStacksPerExpertWeightsIntoSwitchMLP() throws {
        let weights: [String: MLXArray] = [
            "model.layers.1.moe.experts.0.gate_proj.weight": MLXArray.zeros([16, 8]),
            "model.layers.1.moe.experts.1.gate_proj.weight": MLXArray.zeros([16, 8]),
        ]

        let sanitized = SharpAIStep3p7Sanitizer.sanitize(
            weights, configuration: try configuration())

        XCTAssertEqual(Set(sanitized.keys), Set(["model.layers.1.moe.switch_mlp.gate_proj.weight"]))
        XCTAssertEqual(
            sanitized["model.layers.1.moe.switch_mlp.gate_proj.weight"]?.shape, [2, 16, 8])
    }

    func testVisionKeysPassThroughUnchanged() throws {
        let weights: [String: MLXArray] = [
            "vision_model.encoder.layers.0.weight": MLXArray.zeros([16, 16]),
            "vit_large_projector.weight": MLXArray.zeros([32, 16]),
        ]

        let sanitized = SharpAIStep3p7Sanitizer.sanitize(
            weights, configuration: try configuration())

        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
        XCTAssertEqual(sanitized["vision_model.encoder.layers.0.weight"]?.shape, [16, 16])
    }

    /// Vanilla raw names (`moe.gate.`) mark a checkpoint whose norm weights need the +1 shift.
    func testVanillaCheckpointAddsOneToNormWeights() throws {
        let weights: [String: MLXArray] = [
            "model.layers.1.moe.gate.weight": MLXArray.zeros([2, 8]),
            "model.layers.1.input_layernorm.weight": MLXArray.zeros([8]),
        ]

        let sanitized = SharpAIStep3p7Sanitizer.sanitize(
            weights, configuration: try configuration())

        XCTAssertEqual(
            sanitized["model.layers.1.input_layernorm.weight"]?.asArray(Float.self),
            Array(repeating: Float(1), count: 8))
        XCTAssertEqual(
            sanitized["model.layers.1.moe.gate.weight"]?.asArray(Float.self),
            Array(repeating: Float(0), count: 16))
    }

    /// Keys already in MLX names (`mlp.gate.gate.`) are not vanilla, so norm weights are untouched.
    func testMLXNamedCheckpointLeavesNormWeightsUnchanged() throws {
        let weights: [String: MLXArray] = [
            "model.layers.1.mlp.gate.gate.weight": MLXArray.zeros([2, 8]),
            "model.layers.1.input_layernorm.weight": MLXArray.zeros([8]),
        ]

        let sanitized = SharpAIStep3p7Sanitizer.sanitize(
            weights, configuration: try configuration())

        XCTAssertEqual(
            sanitized["model.layers.1.input_layernorm.weight"]?.asArray(Float.self),
            Array(repeating: Float(0), count: 8))
    }

    func testEmptyInputGivesEmptyOutput() throws {
        XCTAssertTrue(
            SharpAIStep3p7Sanitizer.sanitize([:], configuration: try configuration()).isEmpty)
    }
}
