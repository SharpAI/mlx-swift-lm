// SharpAI fork overlay tests for kimi_k3 key sanitization. Not part of upstream.
// Key names are checked against moonshotai/Kimi-K3 model.safetensors.index.json (fetched 2026-10-30):
// language_model.model.embed_tokens.weight, vision_tower.encoder.blocks.N.wqkv.weight, mm_projector.proj.N.weight.
// The shapes are tiny placeholders; only the key set and the shapes that sanitize must keep are asserted.
// Written without compiling.

import MLX
import MLXVLM
import XCTest

final class SharpAIKimiK3SanitizeTests: XCTestCase {

    func testDropsVisionKeysAndKeepsLanguageKeys() {
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([64, 32]),
            "vision_tower.encoder.blocks.0.wqkv.weight": MLXArray.zeros([16, 16]),
            "mm_projector.proj.0.weight": MLXArray.zeros([32, 16]),
        ]

        let sanitized = SharpAIKimiK3Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["language_model.model.embed_tokens.weight"])
        XCTAssertEqual(sanitized["language_model.model.embed_tokens.weight"]?.shape, [64, 32])
    }

    func testEmptyInputGivesEmptyOutput() {
        XCTAssertTrue(SharpAIKimiK3Sanitizer.sanitize([:]).isEmpty)
    }
}
