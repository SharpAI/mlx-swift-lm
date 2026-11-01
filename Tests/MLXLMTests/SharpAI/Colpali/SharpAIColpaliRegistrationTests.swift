// SharpAI fork overlay tests for the colpali creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIColpaliRegistrationTests: XCTestCase {

    private let tinyJSON = """
        {
          "model_type": "colpali",
          "embedding_dim": 16,
          "text_config": {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 64
          },
          "vlm_config": {
            "image_token_index": 60,
            "vision_config": {
              "hidden_size": 16,
              "num_hidden_layers": 1,
              "intermediate_size": 32,
              "num_attention_heads": 2,
              "patch_size": 14,
              "image_size": 28,
              "projection_dim": 32
            }
          }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIColpaliOverlay.creators["colpali"])
    }

    func testCreatorBuildsColpaliModelFromValidConfig() throws {
        guard let creator = SharpAIColpaliOverlay.creators["colpali"] else {
            XCTFail("colpali creator missing")
            return
        }

        let model = try creator(Data(tinyJSON.utf8))

        XCTAssertTrue(model is SharpAIColpali)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIColpaliOverlay.creators["colpali"] else {
            XCTFail("colpali creator missing")
            return
        }
        let json = tinyJSON.replacingOccurrences(
            of: "\"embedding_dim\": 16,", with: "\"embedding_dim\": 0,")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case SharpAIColpaliError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
