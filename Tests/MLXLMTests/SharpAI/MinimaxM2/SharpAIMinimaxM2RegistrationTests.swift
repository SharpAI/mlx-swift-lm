import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIMinimaxM2RegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMinimaxM2Overlay.creators["minimax_m2"])
    }

    func testValidConfigFailsWithBlockedOnBaseError() {
        guard let creator = SharpAIMinimaxM2Overlay.creators["minimax_m2"] else {
            XCTFail("minimax_m2 creator missing")
            return
        }
        XCTAssertThrowsError(try creator(Data(SharpAIMinimaxM2TestFixtures.tinyJSON.utf8))) { error in
            XCTAssertEqual(error as? SharpAIMinimaxM2Error, .blockedOnBase)
        }
    }

    func testInvalidConfigFailsBeforeBlockedError() {
        guard let creator = SharpAIMinimaxM2Overlay.creators["minimax_m2"] else {
            XCTFail("minimax_m2 creator missing")
            return
        }
        XCTAssertThrowsError(try creator(Data("{}".utf8))) { error in
            XCTAssertNotEqual(error as? SharpAIMinimaxM2Error, .blockedOnBase)
        }
    }

    func testErrorDescriptionMentionsModelType() {
        let description = SharpAIMinimaxM2Error.blockedOnBase.errorDescription ?? ""
        XCTAssertTrue(description.contains("minimax_m2"))
    }
}

final class SharpAIMinimaxM2RegistrationKnownIssueTests: XCTestCase {

    // KNOWN GAP: no tiny forward pass exists. The creator throws blockedOnBase for a valid config.
    // Sources checked: thoughtworks/MiniMax-M2.5-Eagle3 config.json declares model_type "llama" and
    // architectures LlamaForCausalLMEagle3, not "minimax_m2". SGLang's llama_eagle3.py (the loader that
    // serves that drafter, raw.githubusercontent.com/sgl-project/sglang) gives the drafter structure:
    // fc over the concatenated aux hidden states, a midlayer decoder layer with hidden_norm, and a
    // draft-vocab lm_head with d2t mapping. The MiniMax M2 target model is not implemented in MLX here.
    // The Swift drafter class and the creator change are not written yet, so this stays an expected
    // failure. Writing it, and changing the blockedOnBase expectation, is the remaining work.
    func testTinyForwardPassKnownIssue() {
        guard let creator = SharpAIMinimaxM2Overlay.creators["minimax_m2"] else {
            XCTFail("minimax_m2 creator missing")
            return
        }
        XCTExpectFailure(
            "KNOWN GAP: no Swift EAGLE3 drafter class yet; the reference is LlamaForCausalLMEagle3 (SGLang llama_eagle3.py), but the creator still throws blockedOnBase and the drafter is not model_type minimax_m2"
        ) {
            do {
                let model = try creator(Data(SharpAIMinimaxM2TestFixtures.tinyJSON.utf8))
                XCTAssertNotNil(model)
            } catch {
                XCTFail("tiny forward pass unavailable: \(error)")
            }
        }
    }
}
