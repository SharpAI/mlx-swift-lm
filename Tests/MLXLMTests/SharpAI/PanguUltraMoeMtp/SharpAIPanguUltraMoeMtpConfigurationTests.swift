// SharpAI fork overlay tests for the pangu_ultra_moe_mtp configuration. Not part of upstream.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIPanguUltraMoeMtpConfigurationTests: XCTestCase {

    /// Inline shape with the spec's model type and architecture. No checkpoint values are assumed.
    private let fixtureJSON = """
        {
            "architectures": ["OpenPanguMTPModel"],
            "model_type": "pangu_ultra_moe_mtp"
        }
        """

    private func decode(_ json: String) throws -> SharpAIPanguUltraMoeMtpConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIPanguUltraMoeMtpConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesModelTypeAndArchitecture() throws {
        let config = try decode(fixtureJSON)
        XCTAssertEqual(config.modelType, "pangu_ultra_moe_mtp")
        XCTAssertEqual(config.architectures, ["OpenPanguMTPModel"])
    }

    func testValidationAcceptsExpectedConfiguration() throws {
        let config = try decode(fixtureJSON)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingModelTypeDefaultsToExpectedValue() throws {
        let config = try decode("{\"architectures\": [\"OpenPanguMTPModel\"]}")
        XCTAssertEqual(config.modelType, SharpAIPanguUltraMoeMtpConfiguration.expectedModelType)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingArchitecturesDecodesAsEmptyAndValidates() throws {
        let config = try decode("{\"model_type\": \"pangu_ultra_moe_mtp\"}")
        XCTAssertTrue(config.architectures.isEmpty)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsWrongModelType() throws {
        let config = try decode("{\"model_type\": \"pangu_ultra_moe\"}")
        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            switch error as? SharpAIPanguUltraMoeMtpError {
            case .invalidConfiguration(_)?: break
            default: XCTFail("expected invalidConfiguration, got \(error)")
            }
        }
    }

    func testValidationRejectsWrongArchitecture() throws {
        let config = try decode(
            "{\"model_type\": \"pangu_ultra_moe_mtp\", \"architectures\": [\"SomeOtherModel\"]}")
        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            switch error as? SharpAIPanguUltraMoeMtpError {
            case .invalidConfiguration(_)?: break
            default: XCTFail("expected invalidConfiguration, got \(error)")
            }
        }
    }

    func testUnknownKeysAreIgnored() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"model_type\"", with: "\"unmodeled_key\": 1, \"model_type\"")
        let config = try decode(json)
        XCTAssertEqual(config.modelType, "pangu_ultra_moe_mtp")
    }
}
