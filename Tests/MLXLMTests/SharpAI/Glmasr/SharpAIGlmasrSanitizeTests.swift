// SharpAI fork overlay tests for glmasr checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
@testable import MLXVLM
import XCTest

final class SharpAIGlmasrSanitizeTests: XCTestCase {

    private func config() throws -> SharpAIGlmasrConfiguration {
        try SharpAIGlmasrTestConfig.decode(SharpAIGlmasrTestConfig.tiny())
    }

    func testConvWeightsMoveToMLXLayout() throws {
        let weights: [String: MLXArray] = [
            // PyTorch [out, in, kernel]: conv1 in = 4 mel bins, conv2 in = 8 hidden.
            "audio_tower.conv1.weight": MLXArray.zeros([8, 4, 3]),
            "audio_tower.conv2.weight": MLXArray.zeros([8, 8, 3]),
        ]

        let result = SharpAIGlmasrSanitize.sanitize(weights: weights, config: try config())

        XCTAssertEqual(result["audio_tower.conv1.weight"]?.shape, [8, 3, 4])
        XCTAssertEqual(result["audio_tower.conv2.weight"]?.shape, [8, 3, 8])
    }

    func testRemapsTransformersPrefixesAndDropsRotaryBuffers() throws {
        let weights: [String: MLXArray] = [
            "model.language_model.embed_tokens.weight": MLXArray.zeros([32, 8]),
            "model.language_model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
            "model.language_model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
            "model.audio_tower.norm.weight": MLXArray.zeros([8]),
            "model.multi_modal_projector.linear_1.weight": MLXArray.zeros([16, 32]),
            "lm_head.weight": MLXArray.zeros([32, 8]),
        ]

        let result = SharpAIGlmasrSanitize.sanitize(weights: weights, config: try config())

        XCTAssertNotNil(result["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(result["language_model.model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(result["audio_tower.norm.weight"])
        XCTAssertNotNil(result["multi_modal_projector.linear_1.weight"])
        XCTAssertNotNil(result["language_model.lm_head.weight"])
        XCTAssertNil(result["model.language_model.embed_tokens.weight"])
        XCTAssertNil(result["lm_head.weight"])
        XCTAssertFalse(result.keys.contains { $0.contains("rotary_emb.inv_freq") })
        XCTAssertEqual(result.count, 5)
    }

    func testLlamaStylePrefixMapsIntoLanguageModel() {
        XCTAssertEqual(
            SharpAIGlmasrSanitize.checkpointName("model.layers.3.mlp.up_proj.weight"),
            "language_model.model.layers.3.mlp.up_proj.weight")
    }

    func testVLMLayoutKeysPassThroughUnchanged() throws {
        let weights: [String: MLXArray] = [
            "audio_tower.layers.0.self_attn.k_proj.weight": MLXArray.zeros([8, 8]),
            "language_model.model.layers.0.mlp.up_proj.weight": MLXArray.zeros([16, 8]),
            "language_model.lm_head.weight": MLXArray.zeros([32, 8]),
        ]

        let result = SharpAIGlmasrSanitize.sanitize(weights: weights, config: try config())

        XCTAssertEqual(Set(result.keys), Set(weights.keys))
        XCTAssertEqual(result["audio_tower.layers.0.self_attn.k_proj.weight"]?.shape, [8, 8])
    }
}
