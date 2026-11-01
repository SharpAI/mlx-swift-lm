// SharpAI fork overlay tests for NemotronH_Nano_VL_V2 key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAINemotronhNanoVlV2SanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    func testStripsLanguagePrefixAndKeepsTextKeys() throws {
        let weights: [String: MLXArray] = [
            "language_model.backbone.embeddings.weight": scalar(),
            "language_model.backbone.layers.0.norm.weight": scalar(),
        ]
        let sanitized = try SharpAINemotronhNanoVlV2Sanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            ["backbone.embeddings.weight", "backbone.layers.0.norm.weight"])
    }

    func testDropsVisionWeights() throws {
        let weights: [String: MLXArray] = [
            "vision_model.encoder.layers.0.weight": scalar(),
            "vision_tower.patch_embed.weight": scalar(),
            "mlp1.0.weight": scalar(),
            "language_model.lm_head.weight": scalar(),
        ]
        let sanitized = try SharpAINemotronhNanoVlV2Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["lm_head.weight"])
    }

    func testDuplicateAfterStripThrows() {
        let weights: [String: MLXArray] = [
            "language_model.lm_head.weight": scalar(),
            "lm_head.weight": scalar(),
        ]

        XCTAssertThrowsError(try SharpAINemotronhNanoVlV2Sanitizer.sanitize(weights))
    }
}
