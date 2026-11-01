// SharpAI fork overlay tests for the nemo_speechlm creator. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAINemoSpeechlmRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "nemo_speechlm",
            "architectures": ["Nemotron3_5TranscribeForConditionalGeneration"]
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAINemoSpeechlmOverlay.creators["nemo_speechlm"])
    }

    func testValidConfigFailsUntilForwardIsImplemented() throws {
        guard let creator = SharpAINemoSpeechlmOverlay.creators["nemo_speechlm"] else {
            XCTFail("nemo_speechlm creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard
                case SharpAINemoSpeechlmError.forwardNotImplemented(let type) = error
            else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "nemo_speechlm")
        }
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAINemoSpeechlmOverlay.creators["nemo_speechlm"] else {
            XCTFail("nemo_speechlm creator missing")
            return
        }

        let badJSON = """
            {
                "model_type": "nemo_speechlm",
                "architectures": ["NotATranscribeModel"]
            }
            """
        XCTAssertThrowsError(try creator(Data(badJSON.utf8))) { error in
            if case SharpAINemoSpeechlmError.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }
}
