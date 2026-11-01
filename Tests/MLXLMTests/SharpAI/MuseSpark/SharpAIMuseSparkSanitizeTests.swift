// SharpAI fork overlay tests for muse_spark checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIMuseSparkSanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    // Placeholder: the sanitizer is a pass-through until the key layout is verified.
    func testPassThroughKeepsAllKeysAndValues() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.q_proj.weight": vector(1),
            "lm_head.weight": vector(2),
        ]

        let result = SharpAIMuseSparkSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(weights.keys))
        XCTAssertEqual(result["lm_head.weight"]?.shape, [2])
    }
}
