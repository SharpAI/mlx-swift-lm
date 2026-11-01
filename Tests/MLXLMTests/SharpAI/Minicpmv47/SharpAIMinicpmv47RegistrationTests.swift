// SharpAI fork overlay tests for the minicpmv4_7 creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIMinicpmv47RegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMinicpmv47Overlay.creators["minicpmv4_7"])
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAIMinicpmv47Overlay.creators["minicpmv4_7"] else {
            XCTFail("minicpmv4_7 creator missing")
            return
        }

        let model = try creator(Data(SharpAIMinicpmv47Fixtures.tinyJSON.utf8))
        XCTAssertTrue(model is SharpAIMinicpmv47)
    }

    func testConfigWithoutTextBlockFailsBeforeConstruction() throws {
        guard let creator = SharpAIMinicpmv47Overlay.creators["minicpmv4_7"] else {
            XCTFail("minicpmv4_7 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"model_type\": \"minicpmv4_7\" }".utf8)))
    }

    func testEmptyModelTypeFailsBeforeConstruction() throws {
        guard let creator = SharpAIMinicpmv47Overlay.creators["minicpmv4_7"] else {
            XCTFail("minicpmv4_7 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"model_type\": \"\" }".utf8)))
    }

    /// Formerly skipped. The model class now exists, so this runs a real tiny forward pass.
    func testTinyForwardPass() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv47Configuration.self,
            from: Data(SharpAIMinicpmv47Fixtures.tinyJSON.utf8))
        let model = try SharpAIMinicpmv47(config)

        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)
        let output = model.callAsFunction(
            LMInput.Text(tokens: tokens), cache: nil, state: nil)
        XCTAssertEqual(output.logits.shape, [1, 3, 128])

        // 56px / patch 14 = 4x4 patches. Window merger gives 2x2, MLP merger gives one token.
        let slice = MLXRandom.normal([56, 56, 3])
        let encoded = try model.encodeSlice(slice)
        XCTAssertEqual(encoded.shape, [1, 32])
    }
}
