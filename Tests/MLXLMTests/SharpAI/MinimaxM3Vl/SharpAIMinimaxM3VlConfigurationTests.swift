// SharpAI fork overlay tests for the minimax_m3_vl configuration. Not part of upstream.
// Inline JSON only. No checkpoint, no network.
// UNVERIFIED: field values are placeholders, not M3 values. Written without compiling.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

enum SharpAIMinimaxM3VlFixtures {
    static let tinyJSON = """
        {
          "model_type": "minimax_m3_vl",
          "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
          "text_config": {"hidden_size": 32, "num_hidden_layers": 2},
          "vision_config": {"hidden_size": 16}
        }
        """
}

final class SharpAIMinimaxM3VlConfigurationTests: XCTestCase {

    func testDecodesTopLevelKeys() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinimaxM3VlConfiguration.self,
            from: Data(SharpAIMinimaxM3VlFixtures.tinyJSON.utf8))

        XCTAssertEqual(config.modelType, "minimax_m3_vl")
        XCTAssertEqual(config.architectures, ["MiniMaxM3SparseForConditionalGeneration"])
        XCTAssertNotNil(config.visionConfig)
    }

    func testTextConfigIsKeptAsRawObject() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinimaxM3VlConfiguration.self,
            from: Data(SharpAIMinimaxM3VlFixtures.tinyJSON.utf8))

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
              "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
              "text_config": {"hidden_size": 32}
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIMinimaxM3VlConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "minimax_m3_vl")
        XCTAssertNil(config.visionConfig)
    }

    func testArchitecturesDefaultToEmptyWhenAbsent() throws {
        let json = #"{"model_type": "minimax_m3_vl", "text_config": {"hidden_size": 32}}"#
        let config = try JSONDecoder.json5().decode(
            SharpAIMinimaxM3VlConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.architectures, [])
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidConfigPassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinimaxM3VlConfiguration.self,
            from: Data(SharpAIMinimaxM3VlFixtures.tinyJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsOtherArchitecture() throws {
        let json = SharpAIMinimaxM3VlFixtures.tinyJSON.replacingOccurrences(
            of: "MiniMaxM3SparseForConditionalGeneration", with: "LlamaForCausalLM")
        let config = try JSONDecoder.json5().decode(
            SharpAIMinimaxM3VlConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsNonObjectTextConfig() throws {
        let json = #"{"model_type": "minimax_m3_vl", "text_config": 3}"#
        let config = try JSONDecoder.json5().decode(
            SharpAIMinimaxM3VlConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testMissingTextConfigFailsDecode() {
        let json = #"{"model_type": "minimax_m3_vl"}"#
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIMinimaxM3VlConfiguration.self, from: Data(json.utf8)))
    }
}
