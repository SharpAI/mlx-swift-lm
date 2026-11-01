// SharpAI fork overlay tests for the muse_spark_dspark creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMuseSparkDsparkRegistrationTests: XCTestCase {

    private let validJSON = """
        {
          "model_type": "muse_spark_dspark",
          "architectures": ["MuseSparkDSparkModel"],
          "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
          "num_key_value_heads": 1, "head_dim": 8, "vocab_size": 50, "draft_vocab_size": 40,
          "moe": {"is_enabled": false},
          "target_layer_ids": [1, 3], "num_target_layers": 2, "target_hidden_size": 6,
          "markov_rank": 4
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMuseSparkDsparkOverlay.creators["muse_spark_dspark"])
    }

    func testValidConfigFailsBecauseTargetIsMissing() throws {
        let creator = try XCTUnwrap(SharpAIMuseSparkDsparkOverlay.creators["muse_spark_dspark"])

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIMuseSparkDsparkError.targetNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "muse_spark_dspark")
        }
    }

    func testInvalidConfigFailsBeforeTargetCheck() throws {
        let creator = try XCTUnwrap(SharpAIMuseSparkDsparkOverlay.creators["muse_spark_dspark"])
        let json = validJSON.replacingOccurrences(
            of: "\"target_layer_ids\": [1, 3]", with: "\"target_layer_ids\": []")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("config validation should fail first: \(error)")
                return
            }
        }
    }

    func testWrongModelTypeFailsBeforeTargetCheck() throws {
        let creator = try XCTUnwrap(SharpAIMuseSparkDsparkOverlay.creators["muse_spark_dspark"])
        let json = validJSON.replacingOccurrences(
            of: "\"model_type\": \"muse_spark_dspark\"", with: "\"model_type\": \"llama\"")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("config validation should fail first: \(error)")
                return
            }
        }
    }
}
