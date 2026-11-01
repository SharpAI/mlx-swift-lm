// SharpAI fork overlay tests for domynedge checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIDomynedgeSanitizeTests: XCTestCase {

    private func sampleWeights() -> [String: MLXArray] {
        let scalar = MLXArray(Float(0))
        return [
            "model.embed_tokens.weight": scalar,
            "model.layers.0.self_attn.q_proj.weight": scalar,
            "model.layers.0.self_attn.rotary_emb.inv_freq": scalar,
            "model.layers.0.pre_feedforward_layernorm.weight": scalar,
            "model.norm.weight": scalar,
            "lm_head.weight": scalar,
        ]
    }

    func testKeepsHFNamesAndDropsRotaryBuffers() throws {
        let config = try SharpAIDomynedgeTestFixtures.tinyConfiguration()
        let out = SharpAIDomynedgeSanitizer.sanitize(sampleWeights(), configuration: config)

        XCTAssertFalse(out.keys.contains { $0.contains("rotary_emb.inv_freq") })
        XCTAssertTrue(out.keys.contains("model.layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(out.keys.contains("model.layers.0.pre_feedforward_layernorm.weight"))
        XCTAssertTrue(out.keys.contains("model.norm.weight"))
    }

    func testDropsLMHeadWhenEmbeddingsAreTied() throws {
        let config = try SharpAIDomynedgeTestFixtures.tinyConfiguration(tied: true)
        var weights = sampleWeights()
        weights["lm_head.scales"] = MLXArray(Float(0))
        let out = SharpAIDomynedgeSanitizer.sanitize(weights, configuration: config)

        XCTAssertFalse(out.keys.contains { $0.hasPrefix("lm_head.") })
    }

    func testKeepsLMHeadWhenEmbeddingsAreUntied() throws {
        let config = try SharpAIDomynedgeTestFixtures.tinyConfiguration(tied: false)
        let out = SharpAIDomynedgeSanitizer.sanitize(sampleWeights(), configuration: config)

        XCTAssertTrue(out.keys.contains("lm_head.weight"))
    }

    func testModelSanitizeMatchesStandaloneSanitizer() throws {
        let config = try SharpAIDomynedgeTestFixtures.tinyConfiguration()
        let model = SharpAIDomynedgeModel(config)
        let fromModel = model.sanitize(weights: sampleWeights())
        let direct = SharpAIDomynedgeSanitizer.sanitize(sampleWeights(), configuration: config)

        XCTAssertEqual(Set(fromModel.keys), Set(direct.keys))
    }
}
