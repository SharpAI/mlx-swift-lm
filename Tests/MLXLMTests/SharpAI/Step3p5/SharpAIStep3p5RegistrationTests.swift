// SharpAI fork overlay tests for the step3p5 creator. Not part of upstream.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIStep3p5RegistrationTests: XCTestCase {

    private func creator() throws -> (Data) throws -> any LanguageModel {
        guard let creator = SharpAIStep3p5Overlay.creators["step3p5"] else {
            throw XCTSkip("step3p5 creator missing")
        }
        return creator
    }

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIStep3p5Overlay.creators["step3p5"])
    }

    func testValidTinyConfigBuildsModel() throws {
        let model = try creator()(Data(SharpAIStep3p5TestFixture.json().utf8))
        XCTAssertTrue(model is SharpAIStep3p5Model)
    }

    func testMTPOnlyConfigFailsBeforeConstruction() throws {
        let json = SharpAIStep3p5TestFixture.json(["architectures": ["Step3p5MTP"]])

        XCTAssertThrowsError(try creator()(Data(json.utf8))) { error in
            guard case SharpAIStep3p5Error.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        XCTAssertThrowsError(try creator()(Data("{ \"hidden_size\": 32 }".utf8)))
    }
}
