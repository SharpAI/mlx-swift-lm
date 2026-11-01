// SharpAI fork overlay tests for the ming_v3_asr configuration. Not part of upstream.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIMingV3AsrConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAIMingV3AsrConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIMingV3AsrConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesModelTypeAndIgnoresUnknownKeys() throws {
        let configuration = try decode(#"{"model_type": "ming_v3_asr", "unknown_key": 1}"#)
        XCTAssertEqual(configuration.modelType, "ming_v3_asr")
    }

    func testValidationRejectsOtherModelType() throws {
        let configuration = try decode(#"{"model_type": "bailing_moe_v3"}"#)
        XCTAssertThrowsError(try configuration.validateModelConfiguration()) { error in
            guard let mingError = error as? SharpAIMingV3AsrError,
                case .invalidConfiguration = mingError
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    /// Known gap: the encoder and text backbone are not written, so validation always fails.
    func testValidationReportsUnimplementedModel() throws {
        let configuration = try decode(#"{"model_type": "ming_v3_asr"}"#)
        XCTAssertThrowsError(try configuration.validateModelConfiguration()) { error in
            guard let mingError = error as? SharpAIMingV3AsrError,
                case .unsupported = mingError
            else {
                XCTFail("expected unsupported, got \(error)")
                return
            }
        }
    }
}
