// SharpAI fork overlay tests for the PanguEmbedded creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIPanguembeddedRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "PanguEmbedded",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-05,
            "vocab_size": 100,
            "bias": true
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIPanguembeddedOverlay.creators["PanguEmbedded"])
    }

    func testCreatorBuildsPanguembeddedModel() throws {
        let creator = try XCTUnwrap(SharpAIPanguembeddedOverlay.creators["PanguEmbedded"])
        let model = try creator(Data(validJSON.utf8))
        XCTAssertTrue(model is SharpAIPanguembeddedModel)
    }

    func testCreatorRejectsInvalidHeadCounts() throws {
        let creator = try XCTUnwrap(SharpAIPanguembeddedOverlay.creators["PanguEmbedded"])
        let json = validJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 2,", with: "\"num_key_value_heads\": 3,")
        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }
}
