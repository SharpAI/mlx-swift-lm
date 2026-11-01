// SharpAI fork overlay tests for the granite_swa creator. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIGraniteSwaRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "granite_swa",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 512
        }
        """

    private let tinyJSON = """
        {
            "model_type": "granite_swa",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 32,
            "sliding_window": 4
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIGraniteSwaOverlay.creators["granite_swa"])
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAIGraniteSwaOverlay.creators["granite_swa"] else {
            XCTFail("granite_swa creator missing")
            return
        }

        let model = try creator(Data(validJSON.utf8))
        XCTAssertTrue(model is SharpAIGraniteSwaModel)
    }

    func testInvalidConfigFailsBeforeConstruction() {
        guard let creator = SharpAIGraniteSwaOverlay.creators["granite_swa"] else {
            XCTFail("granite_swa creator missing")
            return
        }
        let json = """
            { "hidden_size": 64 }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testTinyForwardPassShape() throws {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIGraniteSwaConfiguration.self, from: Data(tinyJSON.utf8))
        let model = SharpAIGraniteSwaModel(configuration)

        let tokens = MLXArray([1, 2, 3]).reshaped(1, 3)
        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }
}
