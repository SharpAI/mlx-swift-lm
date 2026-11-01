// SharpAI fork overlay tests for the llama_nemotron_vl creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAILlamaNemotronVlRegistrationTests: XCTestCase {

    private let tinyJSON = """
        {
          "model_type": "llama_nemotron_vl",
          "llm_config": {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 64
          },
          "vision_config": {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "patch_size": 14,
            "image_size": 28
          },
          "downsample_ratio": 0.5,
          "img_context_token_id": 60
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAILlamaNemotronVlOverlay.creators["llama_nemotron_vl"])
    }

    func testCreatorBuildsEmbedderFromValidConfig() throws {
        guard let creator = SharpAILlamaNemotronVlOverlay.creators["llama_nemotron_vl"] else {
            XCTFail("llama_nemotron_vl creator missing")
            return
        }

        let model = try creator(Data(tinyJSON.utf8))

        XCTAssertTrue(model is SharpAILlamaNemotronVl)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAILlamaNemotronVlOverlay.creators["llama_nemotron_vl"] else {
            XCTFail("llama_nemotron_vl creator missing")
            return
        }
        let json = tinyJSON.replacingOccurrences(
            of: "\"downsample_ratio\": 0.5,", with: "\"downsample_ratio\": 0.3,")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case SharpAILlamaNemotronVlError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
