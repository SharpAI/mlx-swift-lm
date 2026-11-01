// SharpAI fork overlay tests for the hyperclovax_vision_v2 creator. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIHyperclovaxVisionV2RegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "hyperclovax_vision_v2",
            "text_config": {
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 8
            },
            "vision_config": {
                "model_type": "vision"
            }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIHyperclovaxVisionV2Overlay.creators["hyperclovax_vision_v2"])
    }

    // Pins the stub. Only checks that a valid config reaches the stub error, not model behaviour.
    // Delete this when the model class lands.
    func testValidConfigFailsUntilForwardIsImplemented() throws {
        guard let creator = SharpAIHyperclovaxVisionV2Overlay.creators["hyperclovax_vision_v2"]
        else {
            XCTFail("hyperclovax_vision_v2 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIHyperclovaxVisionV2Error.forwardNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "hyperclovax_vision_v2")
        }
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIHyperclovaxVisionV2Overlay.creators["hyperclovax_vision_v2"]
        else {
            XCTFail("hyperclovax_vision_v2 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 64 }".utf8))) { error in
            if case SharpAIHyperclovaxVisionV2Error.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }

    // UNVERIFIED: no tiny-shape forward pass test. The model class is not written, and the text
    // decoder and the fileprivate ViT have no reference source in the repo. Add it with the model.
}
