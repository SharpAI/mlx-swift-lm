import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIHyV4SanitizeTests: XCTestCase {

    private func makeConfig() throws -> SharpAIHyV4Configuration {
        try JSONDecoder.json5().decode(
            SharpAIHyV4Configuration.self,
            from: Data(SharpAIHyV4ConfigurationTests.makeJSON().utf8))
    }

    func testSanitizeDropsRotaryBuffersAndMTPLayers() throws {
        let config = try makeConfig()
        let one = MLXArray(Float(1))
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": one,
            "model.layers.0.self_attn.q_a_proj.weight": one,
            "model.layers.2.mlp.gate.weight": one,
            "model.layers.0.rotary_emb.inv_freq": one,
            "model.layers.3.eh_proj.weight": one,
            "model.layers.3.self_attn.q_a_proj.weight": one,
            "lm_head.weight": one,
        ]

        let kept = SharpAIHyV4Sanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(
            Set(kept.keys),
            [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_a_proj.weight",
                "model.layers.2.mlp.gate.weight",
                "lm_head.weight",
            ])
    }

    func testLayerIndexParsesOnlyModelLayerKeys() {
        XCTAssertEqual(SharpAIHyV4Sanitizer.layerIndex("model.layers.12.mlp.weight"), 12)
        XCTAssertNil(SharpAIHyV4Sanitizer.layerIndex("model.embed_tokens.weight"))
        XCTAssertNil(SharpAIHyV4Sanitizer.layerIndex("model.layers.x.weight"))
    }
}
