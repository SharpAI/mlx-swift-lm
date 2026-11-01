// SharpAI fork overlay tests for alice_ai checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIAliceAiSanitizeTests: XCTestCase {

    private func makeModel() throws -> SharpAIAliceAiModel {
        let config = try JSONDecoder.json5().decode(
            SharpAIAliceAiConfiguration.self, from: try SharpAIAliceAiTestSupport.tinyData())
        try config.validateModelConfiguration()
        return SharpAIAliceAiModel(config)
    }

    func testMapsCheckpointKeysToModelKeys() throws {
        let model = try makeModel()

        // Tiny config: 2 KDA heads of dim 4, so projection P = 8. Four experts, hidden 16, expert width 8.
        var weights: [String: MLXArray] = [
            "mtp.layers.0.weight": MLXArray.zeros([2]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "model.layers.0.linear_attn.a_log_bias": MLXArray.zeros([1, 1, 2, 1]),
            "model.layers.0.linear_attn.dt_bias": MLXArray.zeros([2, 4]),
            "model.norm.weight": MLXArray.zeros([16]),
            "lm_head.weight": MLXArray.zeros([32, 16]),
        ]
        for name in ["q", "k", "v"] {
            weights["model.layers.0.linear_attn.\(name)_conv1d.weight"] = MLXArray.zeros([8, 1, 4])
        }
        for expert in 0 ..< 4 {
            weights["model.layers.0.mlp.experts.\(expert).gate_proj.weight"] = MLXArray.zeros([8, 16])
        }

        let out = model.sanitize(weights: weights)

        XCTAssertNil(out["mtp.layers.0.weight"])
        XCTAssertNil(out["model.layers.0.self_attn.rotary_emb.inv_freq"])

        XCTAssertNil(out["model.layers.0.linear_attn.a_log_bias"])
        XCTAssertEqual(out["model.layers.0.linear_attn.A_log"]?.shape, [2])
        XCTAssertEqual(out["model.layers.0.linear_attn.dt_bias"]?.shape, [8])

        XCTAssertNil(out["model.layers.0.linear_attn.q_conv1d.weight"])
        XCTAssertEqual(out["model.layers.0.linear_attn.conv1d.weight"]?.shape, [24, 4, 1])

        XCTAssertNil(out["model.layers.0.mlp.experts.0.gate_proj.weight"])
        XCTAssertEqual(out["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 8, 16])

        XCTAssertEqual(
            out["model.norm.weight"]?.asArray(Float.self),
            Array(repeating: Float(1), count: 16))
        XCTAssertNotNil(out["lm_head.weight"])
    }

    func testAcceptsTwoDimensionalConvWeights() throws {
        let model = try makeModel()
        var weights: [String: MLXArray] = [:]
        for name in ["q", "k", "v"] {
            weights["model.layers.0.linear_attn.\(name)_conv1d.weight"] = MLXArray.zeros([8, 4])
        }
        let out = model.sanitize(weights: weights)
        XCTAssertEqual(out["model.layers.0.linear_attn.conv1d.weight"]?.shape, [24, 4, 1])
    }
}
