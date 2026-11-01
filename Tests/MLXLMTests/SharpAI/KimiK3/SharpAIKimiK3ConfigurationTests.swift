// SharpAI fork overlay tests for the kimi_k3 configuration. Not part of upstream.
// Inline JSON only. No checkpoint, no network.
// UNVERIFIED: field values are placeholders, not K3 values. Written without compiling.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

enum SharpAIKimiK3Fixtures {
    static let tinyJSON = """
        {
          "model_type": "kimi_k3",
          "architectures": ["KimiK3ForConditionalGeneration"],
          "text_config": {"hidden_size": 32, "num_hidden_layers": 2},
          "vision_config": {"hidden_size": 16}
        }
        """
}

final class SharpAIKimiK3ConfigurationTests: XCTestCase {

    func testDecodesTopLevelKeys() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIKimiK3Configuration.self, from: Data(SharpAIKimiK3Fixtures.tinyJSON.utf8))

        XCTAssertEqual(config.modelType, "kimi_k3")
        XCTAssertEqual(config.architectures, ["KimiK3ForConditionalGeneration"])
        XCTAssertNotNil(config.visionConfig)
    }

    func testTextConfigIsKeptAsRawObject() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIKimiK3Configuration.self, from: Data(SharpAIKimiK3Fixtures.tinyJSON.utf8))

        guard case .object(let text) = config.textConfig else {
            XCTFail("text_config should decode as an object")
            return
        }
        XCTAssertEqual(text["hidden_size"], JSONValue.int(32))
        XCTAssertEqual(text["num_hidden_layers"], JSONValue.int(2))
    }

    func testModelTypeDefaultsWhenAbsent() throws {
        let json = """
            {
              "architectures": ["KimiK3ForConditionalGeneration"],
              "text_config": {"hidden_size": 32}
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIKimiK3Configuration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "kimi_k3")
        XCTAssertNil(config.visionConfig)
    }

    func testArchitecturesDefaultToEmptyWhenAbsent() throws {
        let json = #"{"model_type": "kimi_k3", "text_config": {"hidden_size": 32}}"#
        let config = try JSONDecoder.json5().decode(
            SharpAIKimiK3Configuration.self, from: Data(json.utf8))

        XCTAssertEqual(config.architectures, [])
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidConfigPassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIKimiK3Configuration.self, from: Data(SharpAIKimiK3Fixtures.tinyJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsOtherArchitecture() throws {
        let json = SharpAIKimiK3Fixtures.tinyJSON.replacingOccurrences(
            of: "KimiK3ForConditionalGeneration", with: "LlamaForCausalLM")
        let config = try JSONDecoder.json5().decode(
            SharpAIKimiK3Configuration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsNonObjectTextConfig() throws {
        let json = #"{"model_type": "kimi_k3", "text_config": 3}"#
        let config = try JSONDecoder.json5().decode(
            SharpAIKimiK3Configuration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testMissingTextConfigFailsDecode() {
        let json = #"{"model_type": "kimi_k3"}"#
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIKimiK3Configuration.self, from: Data(json.utf8)))
    }
}
