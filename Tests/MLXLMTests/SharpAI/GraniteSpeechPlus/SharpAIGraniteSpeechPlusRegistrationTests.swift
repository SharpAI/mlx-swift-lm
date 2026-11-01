// SharpAI fork overlay tests for the granite_speech_plus creator. Not part of upstream.
// The encoder, Q-Former projector and Granite decoder are implemented, so a valid tiny config builds
// a model and runs a forward pass with deterministic weights.
// UNVERIFIED: this file was written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIGraniteSpeechPlusRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIGraniteSpeechPlusOverlay.creators["granite_speech_plus"])
    }

    func testInvalidConfigFailsAtValidation() throws {
        guard let creator = SharpAIGraniteSpeechPlusOverlay.creators["granite_speech_plus"] else {
            XCTFail("granite_speech_plus creator missing")
            return
        }
        let json = SharpAIGraniteSpeechPlusFixtures.tinyJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 2", with: "\"num_key_value_heads\": 3")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard let typed = error as? SharpAIGraniteSpeechPlusError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAIGraniteSpeechPlusOverlay.creators["granite_speech_plus"] else {
            XCTFail("granite_speech_plus creator missing")
            return
        }

        let model = try creator(Data(SharpAIGraniteSpeechPlusFixtures.tinyJSON.utf8))
        XCTAssertTrue(model is SharpAIGraniteSpeechPlusForConditionalGeneration)
    }

    func testTinyForwardPassWithRandomWeights() throws {
        guard let creator = SharpAIGraniteSpeechPlusOverlay.creators["granite_speech_plus"] else {
            XCTFail("granite_speech_plus creator missing")
            return
        }
        let model = try XCTUnwrap(
            try creator(Data(SharpAIGraniteSpeechPlusFixtures.tinyJSON.utf8))
                as? SharpAIGraniteSpeechPlusForConditionalGeneration)

        // 40 frames of 160 mel bins. Window 15 gives 3 blocks, and each block yields 15 / 5 = 3
        // queries, so 9 audio positions of the text width 32.
        let frames = 40
        let features = MLXArray(
            (0 ..< frames * 160).map { Float(sin(Double($0) * 0.01)) }, [1, frames, 160])

        let encoded = model.encoder(features)
        XCTAssertEqual(encoded.shape, [1, frames, 32])

        let audio = model.encodeAudio(features)
        XCTAssertEqual(audio.shape, [1, 9, 32])
        XCTAssertFalse(any(isNaN(audio)).item(Bool.self))

        // Text-only forward: token ids in, logits over the 64-token vocabulary out.
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5]).expandedDimensions(axis: 0)
        let logits = model(tokens, cache: nil)
        XCTAssertEqual(logits.shape, [1, 5, 64])
        XCTAssertFalse(any(isNaN(logits)).item(Bool.self))
    }
}
