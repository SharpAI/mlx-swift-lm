// SharpAI fork overlay tests for the UltraVBERT registration and sanitize. Not part of upstream.
//
// Offline checks only: inline JSON, random tiny tensors, no weights, no network.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIUltravbertRegistrationTests: XCTestCase {

    private let validConfig = Data(
        """
        {
          "architectures": ["ColUltraVBertForRetrieval"],
          "model_type": "ultravbert",
          "embedding_dim": 128
        }
        """.utf8)

    func testCreatorIsRegisteredForUltravbertKey() {
        XCTAssertNotNil(SharpAIUltravbertOverlay.creators["ultravbert"])
        XCTAssertEqual(SharpAIUltravbertOverlay.creators.count, 1)
    }

    func testCreatorThrowsNotImplementedForValidConfig() {
        let creator = SharpAIUltravbertOverlay.creators["ultravbert"]!

        XCTAssertThrowsError(try creator(validConfig)) { error in
            XCTAssertEqual(
                error as? SharpAIUltravbertError,
                .notImplemented(modelType: "ultravbert"))
        }
    }

    func testCreatorRejectsMalformedConfig() {
        let creator = SharpAIUltravbertOverlay.creators["ultravbert"]!

        XCTAssertThrowsError(try creator(Data("not json".utf8)))
    }

    func testSanitizePreservesKeysUntilLayoutIsKnown() {
        let weights: [String: MLXArray] = [
            "model.layers.0.weight": MLXArray.zeros([2, 2]),
            "linear.weight": MLXArray.zeros([2, 2]),
        ]

        let sanitized = SharpAIUltravbertSanitize.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
    }

    /// Text tower only (UltraBERT stack). The vision tower and retrieval head are not implemented.
    func testTinyTextForwardPass() {
        let config = SharpAIUltravbertTextConfiguration(
            vocabularySize: 64, hiddenSize: 16, intermediateSize: 32, hiddenLayers: 3,
            attentionHeads: 2, slidingWindow: 2)
        let model = SharpAIUltravbertTextModel(config)
        let ids = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4), Int32(5)]).reshaped(1, 5)

        let hidden = model(ids, paddingMask: nil)

        XCTAssertEqual(hidden.shape, [1, 5, 16])
        XCTAssertTrue(hidden.asArray(Float.self).allSatisfy { $0.isFinite })
    }
}
