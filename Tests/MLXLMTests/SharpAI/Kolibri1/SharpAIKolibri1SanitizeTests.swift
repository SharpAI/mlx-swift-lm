// SharpAI fork overlay tests for kolibri1 sanitize (key mapping). Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIKolibri1SanitizeTests: XCTestCase {

    func testStacksAllExpertsIntoSwitchMLP() throws {
        let config = try SharpAIKolibri1TestFixtures.configuration(numHiddenLayers: 2, numExperts: 4)
        var weights: [String: MLXArray] = [:]
        for e in 0 ..< 4 {
            for proj in ["gate_proj", "down_proj", "up_proj"] {
                weights["model.layers.1.mlp.experts.\(e).\(proj).weight"] = MLXArray.zeros([8, 8])
            }
        }

        let sanitized = SharpAIKolibri1Sanitizer.sanitize(weights: weights, config: config)

        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 8, 8])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 8, 8])
        XCTAssertNil(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"])
    }

    func testSkipsStackingWhenAnExpertIsMissing() throws {
        let config = try SharpAIKolibri1TestFixtures.configuration(numHiddenLayers: 1, numExperts: 4)
        var weights: [String: MLXArray] = [:]
        for e in 0 ..< 3 {
            weights["model.layers.0.mlp.experts.\(e).up_proj.weight"] = MLXArray.zeros([8, 8])
        }

        let sanitized = SharpAIKolibri1Sanitizer.sanitize(weights: weights, config: config)

        XCTAssertNil(sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"])
        XCTAssertNotNil(sanitized["model.layers.0.mlp.experts.0.up_proj.weight"])
    }

    func testPassesThroughNonExpertKeysUnchanged() throws {
        let config = try SharpAIKolibri1TestFixtures.configuration(numHiddenLayers: 2, numExperts: 4)
        let sanitized = SharpAIKolibri1Sanitizer.sanitize(
            weights: [
                "model.embed_tokens.weight": MLXArray.ones([8, 8]),
                "model.layers.0.self_attn.q_proj.weight": MLXArray.ones([8, 8]),
                "model.layers.0.mlp.gate.weight": MLXArray.ones([4, 8]),
                "lm_head.weight": MLXArray.ones([8, 8]),
            ], config: config)

        XCTAssertEqual(sanitized["model.embed_tokens.weight"]?.shape, [8, 8])
        XCTAssertEqual(sanitized["model.layers.0.self_attn.q_proj.weight"]?.shape, [8, 8])
        XCTAssertEqual(sanitized["model.layers.0.mlp.gate.weight"]?.shape, [4, 8])
        XCTAssertEqual(sanitized["lm_head.weight"]?.shape, [8, 8])
    }

    func testSplitsFusedGateUpAndDownExperts() throws {
        let config = try SharpAIKolibri1TestFixtures.configuration(numHiddenLayers: 1, numExperts: 4)
        // gate_up_proj is [E, 2I, H] with I = 3, H = 8. The gate half comes first on axis 1.
        var gateUpValues = [Float]()
        for e in 0 ..< 4 {
            for r in 0 ..< 6 {
                for c in 0 ..< 8 {
                    gateUpValues.append(Float(e * 1000 + r * 10 + c))
                }
            }
        }
        let weights: [String: MLXArray] = [
            "model.layers.0.mlp.experts.gate_up_proj":
                MLXArray(gateUpValues).reshaped([4, 6, 8]),
            "model.layers.0.mlp.experts.down_proj": MLXArray.zeros([4, 8, 3]),
        ]

        let sanitized = SharpAIKolibri1Sanitizer.sanitize(weights: weights, config: config)

        XCTAssertEqual(
            sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 3, 8])
        XCTAssertEqual(
            sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape, [4, 3, 8])
        XCTAssertEqual(
            sanitized["model.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 8, 3])
        // Expert 1, row 4 of gate_up_proj (first up row) is the up projection, row 0 the gate.
        let up = sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]!
        XCTAssertEqual(up[1, 0, 0].item(Float.self), 1030)
        let gate = sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"]!
        XCTAssertEqual(gate[1, 0, 0].item(Float.self), 1000)
        XCTAssertNil(sanitized["model.layers.0.mlp.experts.gate_up_proj.weight"])
        XCTAssertNil(sanitized["model.layers.0.mlp.experts.down_proj.weight"])
    }

    func testFP8ScaleInvDequantizationAppliesPerBlockScale() throws {
        let config = try SharpAIKolibri1TestFixtures.configuration(numHiddenLayers: 1, numExperts: 4)
        // A 256 x 256 weight has four 128 x 128 blocks. The scale grid is [2, 2].
        let scale = MLXArray([1, 2, 3, 4] as [Float]).reshaped([2, 2])
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.q_proj.weight": MLXArray.ones([256, 256]),
            "model.layers.0.self_attn.q_proj.weight_scale_inv": scale,
        ]

        let sanitized = SharpAIKolibri1Sanitizer.sanitize(weights: weights, config: config)

        let q = sanitized["model.layers.0.self_attn.q_proj.weight"]
        XCTAssertEqual(q?.shape, [256, 256])
        XCTAssertNil(sanitized["model.layers.0.self_attn.q_proj.weight_scale_inv"])
        let values = q?.asArray(Float.self) ?? []
        XCTAssertEqual(values.count, 256 * 256)
        XCTAssertEqual(values[0], 1)  // block (0, 0)
        XCTAssertEqual(values[200], 2)  // row 0, col 200: block (0, 1)
        XCTAssertEqual(values[200 * 256], 3)  // row 200, col 0: block (1, 0)
        XCTAssertEqual(values[200 * 256 + 200], 4)  // block (1, 1)
    }
}
