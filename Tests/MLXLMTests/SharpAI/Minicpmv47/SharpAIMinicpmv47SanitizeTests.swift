// SharpAI fork overlay tests for minicpmv4_7 key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIMinicpmv47SanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    func testKeepsAllKeysUnchanged() {
        let weights: [String: MLXArray] = [
            "llm.model.embed_tokens.weight": scalar(),
            "vpm.encoder.layers.0.weight": scalar(),
            "resampler.attn.in_proj_weight": scalar(),
        ]
        let sanitized = SharpAIMinicpmv47Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
    }

    func testKeepsValues() {
        let value = MLXArray(Float(2.0))
        let sanitized = SharpAIMinicpmv47Sanitizer.sanitize(["lm_head.weight": value])

        XCTAssertEqual(sanitized.count, 1)
        XCTAssertNotNil(sanitized["lm_head.weight"])
    }
}
