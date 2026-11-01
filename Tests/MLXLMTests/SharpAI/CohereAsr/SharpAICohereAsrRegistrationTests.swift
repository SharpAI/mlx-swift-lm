// SharpAI fork overlay tests for the cohere_asr creator and a tiny-shape forward pass.
// Random-init weights only (no checkpoint, no network). Not part of upstream.
// UNVERIFIED: this file was written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAICohereAsrRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAICohereAsrOverlay.creators["cohere_asr"])
    }

    func testInvalidConfigFailsBeforeConstruction() {
        guard let creator = SharpAICohereAsrOverlay.creators["cohere_asr"] else {
            XCTFail("cohere_asr creator missing")
            return
        }
        // Validation rejects the activation. The default-sized model is never built.
        XCTAssertThrowsError(try creator(Data(#"{"hidden_act": "tanh"}"#.utf8)))
    }

    func testTinyForwardPassPrepareThenStep() throws {
        guard let creator = SharpAICohereAsrOverlay.creators["cohere_asr"] else {
            XCTFail("cohere_asr creator missing")
            return
        }
        let model = try creator(Data(SharpAICohereAsrSanitizeTests.tinyJSON.utf8))

        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 2)

        // 32 mel frames of 16 bins subsample to 4 encoder frames.
        let features = MLXArray.zeros([1, 32, 16])
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let input = LMInput(
            text: LMInput.Text(tokens: tokens),
            audio: LMInput.ProcessedAudio(features: features))

        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())
        guard case .logits(let output) = result else {
            XCTFail("prepare should return logits for the prompt")
            return
        }
        XCTAssertEqual(output.logits.shape, [1, 3, 32])
        XCTAssertNotNil(output.state)

        let next = MLXArray([Int32(4)]).reshaped(1, 1)
        let step = model.callAsFunction(
            LMInput.Text(tokens: next), cache: cache, state: output.state)
        XCTAssertEqual(step.logits.shape, [1, 1, 32])
    }

    func testPrepareWithoutAudioThrows() throws {
        guard let creator = SharpAICohereAsrOverlay.creators["cohere_asr"] else {
            XCTFail("cohere_asr creator missing")
            return
        }
        let model = try creator(Data(SharpAICohereAsrSanitizeTests.tinyJSON.utf8))
        let cache = try model.newCache(parameters: nil)
        let input = LMInput(tokens: MLXArray([Int32(1), 2]).reshaped(1, 2))

        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters()))
    }
}
