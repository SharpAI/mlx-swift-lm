// SharpAI fork overlay tests for ming_v3_asr key sanitization. Not part of upstream.
// UNVERIFIED: no checkpoint key names are known, so this only checks the identity mapping.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIMingV3AsrSanitizeTests: XCTestCase {

    func testKeysAndShapesPassThroughUnchanged() {
        let weights: [String: MLXArray] = [
            "encoder.layers.0.weight": MLXArray.zeros([4, 2]),
            "language_model.model.embed_tokens.weight": MLXArray.zeros([32, 8]),
        ]

        let result = SharpAIMingV3AsrSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(weights.keys))
        XCTAssertEqual(result["encoder.layers.0.weight"]?.shape, [4, 2])
        XCTAssertEqual(result["language_model.model.embed_tokens.weight"]?.shape, [32, 8])
    }
}
