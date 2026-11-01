// SharpAI fork overlay tests for nemotron_h_omni key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAINemotronHOmniSanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    func testStripsLanguagePrefixFromTextKeys() throws {
        let weights: [String: MLXArray] = [
            "language_model.backbone.embeddings.weight": scalar(),
            "language_model.lm_head.weight": scalar(),
        ]
        let text = try SharpAINemotronHOmniSanitizer.textWeights(weights)

        XCTAssertEqual(
            Set(text.keys),
            ["backbone.embeddings.weight", "lm_head.weight"])
    }

    func testDropsVisionAndSoundWeights() throws {
        let weights: [String: MLXArray] = [
            "vision_model.radio.blocks.0.weight": scalar(),
            "vision_tower.patch_embed.weight": scalar(),
            "mlp1.0.weight": scalar(),
            "sound_encoder.layers.0.weight": scalar(),
            "language_model.backbone.norm_f.weight": scalar(),
        ]
        let text = try SharpAINemotronHOmniSanitizer.textWeights(weights)

        XCTAssertEqual(Set(text.keys), ["backbone.norm_f.weight"])
    }

    func testKeepsUnprefixedTextKeys() throws {
        let weights: [String: MLXArray] = [
            "backbone.norm_f.weight": scalar()
        ]
        let text = try SharpAINemotronHOmniSanitizer.textWeights(weights)

        XCTAssertEqual(Set(text.keys), ["backbone.norm_f.weight"])
    }

    func testDuplicateAfterStripThrows() {
        let weights: [String: MLXArray] = [
            "language_model.lm_head.weight": scalar(),
            "lm_head.weight": scalar(),
        ]

        XCTAssertThrowsError(try SharpAINemotronHOmniSanitizer.textWeights(weights)) { error in
            XCTAssertEqual(
                error as? SharpAINemotronHOmniError,
                .duplicateWeightKey("lm_head.weight"))
        }
    }
}
