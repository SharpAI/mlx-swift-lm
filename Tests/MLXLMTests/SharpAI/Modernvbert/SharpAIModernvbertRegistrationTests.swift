// SharpAI fork overlay tests for the ModernVBERT creator. Not part of upstream.
//
// Offline checks only: inline JSON, no weights, no network.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIModernvbertRegistrationTests: XCTestCase {

    private let json = """
        {
          "architectures": ["ColModernVBertForRetrieval"],
          "model_type": "modernvbert",
          "embedding_dim": 16,
          "image_token_id": 60,
          "pixel_shuffle_factor": 2,
          "text_config": {
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2
          },
          "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 8,
            "image_size": 32
          }
        }
        """

    func testCreatorRegistersModernvbertKey() {
        XCTAssertNotNil(SharpAIModernvbertOverlay.creators["modernvbert"])
    }

    func testCreatorBuildsModelFromConfigData() throws {
        let creator = try XCTUnwrap(SharpAIModernvbertOverlay.creators["modernvbert"])

        let model = try creator(Data(json.utf8))

        XCTAssertTrue(model is SharpAIModernvbert)
    }

    func testCreatorRejectsInvalidConfig() throws {
        let creator = try XCTUnwrap(SharpAIModernvbertOverlay.creators["modernvbert"])
        let badJSON = json.replacingOccurrences(of: "\"image_size\": 32", with: "\"image_size\": 30")

        XCTAssertThrowsError(try creator(Data(badJSON.utf8)))
    }
}
