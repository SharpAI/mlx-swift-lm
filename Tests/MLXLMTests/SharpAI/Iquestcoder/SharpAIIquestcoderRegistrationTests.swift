// SharpAI fork overlay tests for the iquestcoder creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIIquestcoderRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "iquestcoder",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-05,
            "vocab_size": 100
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIIquestcoderOverlay.creators["iquestcoder"])
    }

    func testCreatorBuildsLlamaModel() throws {
        let creator = try XCTUnwrap(SharpAIIquestcoderOverlay.creators["iquestcoder"])
        let model = try creator(Data(validJSON.utf8))

        XCTAssertTrue(model is LlamaModel)
    }

    func testInvalidConfigThrows() throws {
        let creator = try XCTUnwrap(SharpAIIquestcoderOverlay.creators["iquestcoder"])
        let json = """
            { "model_type": "iquestcoder", "hidden_size": 32 }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testTinyForwardPassWithRandomWeights() throws {
        let creator = try XCTUnwrap(SharpAIIquestcoderOverlay.creators["iquestcoder"])
        let model = try XCTUnwrap(creator(Data(validJSON.utf8)) as? LlamaModel)

        let tokens = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let logits = model(tokens, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 100])
    }
}
