// SharpAI fork overlay tests for minicpm_ocr checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
@testable import MLXVLM
import XCTest

final class SharpAIMinicpmOcrSanitizeTests: XCTestCase {

    func testDropsRotaryBuffersAndKeepsOtherKeysUnchanged() {
        let weights: [String: MLXArray] = [
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
            "language_model.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
            "vision_model.encoder.layers.0.mlp.fc1.weight": MLXArray.zeros([16, 8]),
            "lm_head.weight": MLXArray.zeros([32, 8]),
        ]

        let result = SharpAIMinicpmOcrSanitize.sanitize(weights: weights)

        XCTAssertEqual(result.count, 3)
        XCTAssertNil(result["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertEqual(
            result["language_model.model.layers.0.self_attn.q_proj.weight"]?.shape, [8, 8])
        XCTAssertEqual(result["vision_model.encoder.layers.0.mlp.fc1.weight"]?.shape, [16, 8])
        XCTAssertEqual(result["lm_head.weight"]?.shape, [32, 8])
    }

    func testEmptyInputStaysEmpty() {
        XCTAssertTrue(SharpAIMinicpmOcrSanitize.sanitize(weights: [:]).isEmpty)
    }
}
