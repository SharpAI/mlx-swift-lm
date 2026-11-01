// SharpAI fork overlay tests for the ming_v3_asr creator. Not part of upstream.
// No checkpoint, no network, no weights. UNVERIFIED: written without compiling.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIMingV3AsrRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMingV3AsrOverlay.creators["ming_v3_asr"])
    }

    func testCreatorRejectsWrongModelTypeBeforeConstruction() throws {
        guard let creator = SharpAIMingV3AsrOverlay.creators["ming_v3_asr"] else {
            XCTFail("ming_v3_asr creator missing")
            return
        }
        XCTAssertThrowsError(try creator(Data(#"{"model_type": "cohere_asr"}"#.utf8)))
    }

    /// Known gap: no model class yet, so a valid config still fails at load time.
    func testCreatorFailsUntilModelIsWritten() throws {
        guard let creator = SharpAIMingV3AsrOverlay.creators["ming_v3_asr"] else {
            XCTFail("ming_v3_asr creator missing")
            return
        }
        XCTAssertThrowsError(try creator(Data(#"{"model_type": "ming_v3_asr"}"#.utf8))) { error in
            guard let mingError = error as? SharpAIMingV3AsrError,
                case .unsupported = mingError
            else {
                XCTFail("expected unsupported, got \(error)")
                return
            }
        }
    }
}
