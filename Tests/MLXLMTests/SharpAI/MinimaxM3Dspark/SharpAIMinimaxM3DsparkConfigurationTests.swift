// SharpAI fork overlay tests for the minimax_m3_dspark config. Not part of upstream.
// Offline only: inline JSON, no checkpoint.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMinimaxM3DsparkConfigurationTests: XCTestCase {

    private let validJSON = """
        {
          "architectures": ["DSparkMiniMaxDraftModel"],
          "model_type": "minimax_m3_dspark"
        }
        """

    private func decode(_ json: String) throws -> SharpAIMinimaxM3DsparkConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIMinimaxM3DsparkConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesModelTypeAndArchitecturesFromInlineJSON() throws {
        let configuration = try decode(validJSON)

        XCTAssertEqual(configuration.modelType, "minimax_m3_dspark")
        XCTAssertEqual(configuration.architectures, ["DSparkMiniMaxDraftModel"])
    }

    func testMissingArchitecturesDefaultsToEmpty() throws {
        let json = """
            { "model_type": "minimax_m3_dspark" }
            """
        let configuration = try decode(json)

        XCTAssertEqual(configuration.architectures, [])
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testMissingModelTypeFailsToDecode() {
        let json = """
            { "architectures": ["DSparkMiniMaxDraftModel"] }
            """

        XCTAssertThrowsError(try decode(json))
    }

    func testValidateAcceptsExpectedModelType() throws {
        let configuration = try decode(validJSON)

        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testValidateRejectsWrongModelType() throws {
        let json = validJSON.replacingOccurrences(
            of: "\"model_type\": \"minimax_m3_dspark\"", with: "\"model_type\": \"llama\"")
        let configuration = try decode(json)

        XCTAssertThrowsError(try configuration.validateModelConfiguration()) { error in
            guard let typed = error as? SharpAIMinimaxM3DsparkError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testValidateRejectsWrongArchitecture() throws {
        let json = validJSON.replacingOccurrences(
            of: "DSparkMiniMaxDraftModel", with: "LlamaForCausalLM")
        let configuration = try decode(json)

        XCTAssertThrowsError(try configuration.validateModelConfiguration()) { error in
            guard let typed = error as? SharpAIMinimaxM3DsparkError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }
}
