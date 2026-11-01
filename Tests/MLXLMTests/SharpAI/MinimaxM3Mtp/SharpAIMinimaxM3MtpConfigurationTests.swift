// SharpAI fork overlay tests for the minimax_m3_mtp configuration. Not part of upstream.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

private enum SharpAIMinimaxM3MtpConfigFixtures {
    static let tinyJSON = """
        {
          "model_type": "minimax_m3_mtp",
          "architectures": ["MiniMaxM3MTP"],
          "hidden_size": 64
        }
        """
}

final class SharpAIMinimaxM3MtpConfigurationTests: XCTestCase {

    func testDecodesModelTypeAndArchitecture() throws {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIMinimaxM3MtpConfiguration.self,
            from: Data(SharpAIMinimaxM3MtpConfigFixtures.tinyJSON.utf8))

        XCTAssertEqual(configuration.modelType, "minimax_m3_mtp")
        XCTAssertEqual(configuration.architectures, ["MiniMaxM3MTP"])
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testMissingModelTypeDefaultsToMtpKey() throws {
        let json = #"{"architectures": ["MiniMaxM3MTP"]}"#
        let configuration = try JSONDecoder.json5().decode(
            SharpAIMinimaxM3MtpConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(configuration.modelType, "minimax_m3_mtp")
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testMissingArchitecturesIsAccepted() throws {
        let json = #"{"model_type": "minimax_m3_mtp"}"#
        let configuration = try JSONDecoder.json5().decode(
            SharpAIMinimaxM3MtpConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(configuration.architectures, [])
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testWrongModelTypeFailsValidation() throws {
        let json = #"{"model_type": "minimax_m3_vl", "architectures": ["MiniMaxM3MTP"]}"#
        let configuration = try JSONDecoder.json5().decode(
            SharpAIMinimaxM3MtpConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try configuration.validateModelConfiguration()) { error in
            guard let typed = error as? SharpAIMinimaxM3MtpError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testWrongArchitectureFailsValidation() throws {
        let json = #"{"model_type": "minimax_m3_mtp", "architectures": ["LlamaForCausalLM"]}"#
        let configuration = try JSONDecoder.json5().decode(
            SharpAIMinimaxM3MtpConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try configuration.validateModelConfiguration()) { error in
            guard let typed = error as? SharpAIMinimaxM3MtpError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }
}
