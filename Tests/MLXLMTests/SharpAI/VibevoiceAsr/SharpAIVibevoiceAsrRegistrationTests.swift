// SharpAI fork overlay tests for the vibevoice_asr creator and a tiny-shape forward pass.
// Random-init weights only (no checkpoint, no network). Not part of upstream.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIVibevoiceAsrRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIVibevoiceAsrOverlay.creators["vibevoice_asr"])
    }

    func testInvalidConfigFailsBeforeConstruction() {
        guard let creator = SharpAIVibevoiceAsrOverlay.creators["vibevoice_asr"] else {
            XCTFail("vibevoice_asr creator missing")
            return
        }
        // The audio token id must index the text vocabulary, so this is rejected.
        let json = #"{"audio_token_id": 99999, "text_config": {"vocab_size": 16}}"#
        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testTinyForwardPassMergesAudioIntoPrompt() throws {
        guard let creator = SharpAIVibevoiceAsrOverlay.creators["vibevoice_asr"] else {
            XCTFail("vibevoice_asr creator missing")
            return
        }
        let model = try creator(Data(SharpAIVibevoiceAsrSanitizeTests.tinyJSON.utf8))

        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 1)

        // 6400 samples is two frames at the 3200-sample hop. Two placeholder tokens match them.
        let samples = MLXArray.zeros([1, 6400])
        let tokens = MLXArray([Int32(1), 9, 9, 3]).reshaped(1, 4)
        let input = LMInput(
            text: LMInput.Text(tokens: tokens),
            audio: LMInput.ProcessedAudio(features: samples))

        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())
        guard case .logits(let output) = result else {
            XCTFail("prepare should return logits for a prompt with audio")
            return
        }
        XCTAssertEqual(output.logits.shape, [1, 4, 16])

        let next = MLXArray([Int32(4)]).reshaped(1, 1)
        let step = model.callAsFunction(
            LMInput.Text(tokens: next), cache: cache, state: output.state)
        XCTAssertEqual(step.logits.shape, [1, 1, 16])
    }

    func testPartialHopIsZeroPaddedToOneMoreFrame() throws {
        guard let creator = SharpAIVibevoiceAsrOverlay.creators["vibevoice_asr"] else {
            XCTFail("vibevoice_asr creator missing")
            return
        }
        let model = try creator(Data(SharpAIVibevoiceAsrSanitizeTests.tinyJSON.utf8))
        let cache = try model.newCache(parameters: nil)

        // 3201 samples pads to two hops, so two placeholders are needed.
        let input = LMInput(
            text: LMInput.Text(tokens: MLXArray([Int32(1), 9, 9]).reshaped(1, 3)),
            audio: LMInput.ProcessedAudio(features: MLXArray.zeros([1, 3201])))
        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())
        guard case .logits(let output) = result else {
            XCTFail("prepare should return logits")
            return
        }
        XCTAssertEqual(output.logits.shape, [1, 3, 16])
    }

    func testPlaceholderCountMismatchThrows() throws {
        guard let creator = SharpAIVibevoiceAsrOverlay.creators["vibevoice_asr"] else {
            XCTFail("vibevoice_asr creator missing")
            return
        }
        let model = try creator(Data(SharpAIVibevoiceAsrSanitizeTests.tinyJSON.utf8))
        let cache = try model.newCache(parameters: nil)

        // One placeholder but two audio frames.
        let input = LMInput(
            text: LMInput.Text(tokens: MLXArray([Int32(1), 9, 3]).reshaped(1, 3)),
            audio: LMInput.ProcessedAudio(features: MLXArray.zeros([1, 6400])))
        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters()))
    }

    func testPrepareWithoutAudioUsesTextPath() throws {
        guard let creator = SharpAIVibevoiceAsrOverlay.creators["vibevoice_asr"] else {
            XCTFail("vibevoice_asr creator missing")
            return
        }
        let model = try creator(Data(SharpAIVibevoiceAsrSanitizeTests.tinyJSON.utf8))
        let cache = try model.newCache(parameters: nil)
        let input = LMInput(tokens: MLXArray([Int32(1), 2]).reshaped(1, 2))

        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())
        guard case .tokens = result else {
            XCTFail("text-only input should defer to the token path")
            return
        }
    }
}
