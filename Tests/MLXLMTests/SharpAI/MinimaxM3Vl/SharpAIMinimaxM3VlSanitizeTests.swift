// SharpAI fork overlay tests for minimax_m3_vl key sanitization. Not part of upstream.
// UNVERIFIED: key names are placeholders until an M3 checkpoint is checked. Written without compiling.

import MLX
import MLXVLM
import XCTest

final class SharpAIMinimaxM3VlSanitizeTests: XCTestCase {

    func testKeepsAllKeysAndShapesUnchanged() {
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([64, 32]),
            "vision_tower.encoder.blocks.0.weight": MLXArray.zeros([16, 16]),
            "multi_modal_projector.linear_1.weight": MLXArray.zeros([32, 16]),
        ]

        let sanitized = SharpAIMinimaxM3VlSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
        XCTAssertEqual(sanitized["language_model.model.embed_tokens.weight"]?.shape, [64, 32])
        XCTAssertEqual(sanitized["vision_tower.encoder.blocks.0.weight"]?.shape, [16, 16])
    }

    func testEmptyInputGivesEmptyOutput() {
        XCTAssertTrue(SharpAIMinimaxM3VlSanitizer.sanitize([:]).isEmpty)
    }
}
