// SharpAI fork overlay tests for the mistral_large4 configuration. Not part of upstream.
// Inline JSON only. No checkpoint, no network.
// UNVERIFIED: field values are placeholders, not Mistral Large 4 values. Written without compiling.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

enum SharpAIMistralLarge4Fixtures {
    static let tinyJSON = """
        {
          "model_type": "mistral_large4",
          "architectures": ["MistralLarge4ForConditionalGeneration"],
          "text_config": {"hidden_size": 32, "num_hidden_layers": 2, "tie_word_embeddings": true},
          "vision_config": {"hidden_size": 16}
        }
        """
}

final class SharpAIMistralLarge4ConfigurationTests: XCTestCase {

    func testDecodesTopLevelKeys() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMistralLarge4Configuration.self,
            from: Data(SharpAIMistralLarge4Fixtures.tinyJSON.utf8))

        XCTAssertEqual(config.modelType, "mistral_large4")
        XCTAssertEqual(config.architectures, ["MistralLarge4ForConditionalGeneration"])
        XCTAssertNotNil(config.visionConfig)
    }

    func testTextConfigIsKeptAsRawObject() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMistralLarge4Configuration.self,
            from: Data(SharpAIMistralLarge4Fixtures.tinyJSON.utf8))

        guard case .object(let text) = config.textConfig else {
            XCTFail("text_config should decode as an object")
            return
        }
        XCTAssertEqual(text["hidden_size"], .int(32))
        XCTAssertTrue(config.tiedWordEmbeddings)
    }

    func testValidConfigPasses() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMistralLarge4Configuration.self,
            from: Data(SharpAIMistralLarge4Fixtures.tinyJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testWrongModelTypeFailsValidation() throws {
        let json = """
            {
              "model_type": "mistral3",
              "text_config": {"hidden_size": 32}
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIMistralLarge4Configuration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testWrongArchitectureFailsValidation() throws {
        let json = """
            {
              "model_type": "mistral_large4",
              "architectures": ["SomeOtherForConditionalGeneration"],
              "text_config": {"hidden_size": 32}
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIMistralLarge4Configuration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testMissingTextConfigFailsDecoding() {
        let json = """
            { "model_type": "mistral_large4" }
            """

        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIMistralLarge4Configuration.self, from: Data(json.utf8)))
    }
}
