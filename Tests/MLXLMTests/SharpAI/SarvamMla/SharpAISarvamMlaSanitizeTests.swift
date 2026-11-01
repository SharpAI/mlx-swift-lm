// SharpAI fork sanitize tests for sarvam_mla. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAISarvamMlaSanitizeTests: XCTestCase {

    private func tinyConfiguration(numExperts: Int = 4) throws -> SharpAISarvamMlaConfiguration {
        let json = SharpAISarvamMlaTestFixture.tinyJSON(numExperts: numExperts)
        return try JSONDecoder.json5().decode(
            SharpAISarvamMlaConfiguration.self, from: Data(json.utf8))
    }

    func testDropsRotaryInvFreq() throws {
        let config = try tinyConfiguration()
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": zeros([4]),
            "model.embed_tokens.weight": zeros([64, 32]),
        ]
        let result = SharpAISarvamMlaSanitizer.sanitize(weights: weights, configuration: config)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["model.embed_tokens.weight"])
    }

    func testStacksPerExpertWeightsIntoSwitchMlp() throws {
        let config = try tinyConfiguration(numExperts: 4)
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 4 {
            weights["model.layers.1.mlp.experts.\(expert).gate_proj.weight"] = zeros([16, 32])
        }
        let result = SharpAISarvamMlaSanitizer.sanitize(weights: weights, configuration: config)

        XCTAssertEqual(result["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 16, 32])
        XCTAssertNil(result["model.layers.1.mlp.experts.0.gate_proj.weight"])
        XCTAssertNil(result["model.layers.1.mlp.experts.3.gate_proj.weight"])
    }

    func testLeavesNonExpertKeysUnchanged() throws {
        let config = try tinyConfiguration()
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": zeros([40, 32]),
            "model.layers.0.mlp.shared_experts.up_proj.weight": zeros([16, 32]),
        ]
        let result = SharpAISarvamMlaSanitizer.sanitize(weights: weights, configuration: config)

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(
            result["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"]?.shape, [40, 32])
        XCTAssertEqual(
            result["model.layers.0.mlp.shared_experts.up_proj.weight"]?.shape, [16, 32])
    }

    func testNoStackingWhenNoExperts() throws {
        let config = try tinyConfiguration(numExperts: 0)
        let weights: [String: MLXArray] = [
            "model.layers.0.mlp.experts.0.gate_proj.weight": zeros([16, 32])
        ]
        let result = SharpAISarvamMlaSanitizer.sanitize(weights: weights, configuration: config)

        XCTAssertNotNil(result["model.layers.0.mlp.experts.0.gate_proj.weight"])
        XCTAssertNil(result["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
    }
}
