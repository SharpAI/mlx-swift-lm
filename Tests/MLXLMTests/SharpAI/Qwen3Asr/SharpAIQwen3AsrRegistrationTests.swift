// SharpAI fork overlay tests for the qwen3_asr creator and a tiny-shape forward pass.
// Random-init weights only (no checkpoint, no network). Not part of upstream.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIQwen3AsrRegistrationTests: XCTestCase {

    /// 100 mel frames give one chunk, which gives 13 encoder rows.
    static let placeholderCount = 13

    private func loadCreator() throws -> (Data) throws -> any LanguageModel {
        guard let creator = SharpAIQwen3AsrOverlay.creators["qwen3_asr"] else {
            throw XCTSkip("qwen3_asr creator missing")
        }
        return creator
    }

    private func promptTokens(placeholders: Int) -> MLXArray {
        var ids: [Int32] = [1, 7]
        ids += Array(repeating: 9, count: placeholders)
        ids += [8, 2]
        return MLXArray(ids).reshaped(1, ids.count)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        let creator = try loadCreator()
        // d_model is not divisible by the head count.
        XCTAssertThrowsError(
            try creator(Data(#"{"audio_config": {"d_model": 16, "encoder_attention_heads": 3}}"#.utf8)))
    }

    func testTinyPrepareThenStep() throws {
        let model = try loadCreator()(Data(SharpAIQwen3AsrConfigurationTests.tinyJSON.utf8))

        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 1)

        let tokens = promptTokens(placeholders: Self.placeholderCount)
        let input = LMInput(
            text: LMInput.Text(tokens: tokens),
            audio: LMInput.ProcessedAudio(features: MLXArray.zeros([1, 128, 100])))

        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())
        guard case .logits(let output) = result else {
            XCTFail("prepare should return logits for the prompt")
            return
        }
        XCTAssertEqual(output.logits.shape, [1, tokens.dim(1), 64])

        let next = MLXArray([Int32(3)]).reshaped(1, 1)
        let step = model.callAsFunction(LMInput.Text(tokens: next), cache: cache, state: nil)
        XCTAssertEqual(step.logits.shape, [1, 1, 64])
    }

    func testTextOnlyPromptRunsWithoutAudio() throws {
        let model = try loadCreator()(Data(SharpAIQwen3AsrConfigurationTests.tinyJSON.utf8))
        let cache = try model.newCache(parameters: nil)
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)

        let result = try model.prepare(
            LMInput(text: LMInput.Text(tokens: tokens)),
            cache: cache, state: nil, prefill: PrefillParameters())
        guard case .logits(let output) = result else {
            XCTFail("prepare should return logits")
            return
        }
        XCTAssertEqual(output.logits.shape, [1, 3, 64])
    }

    func testPlaceholdersWithoutAudioThrow() throws {
        let model = try loadCreator()(Data(SharpAIQwen3AsrConfigurationTests.tinyJSON.utf8))
        let cache = try model.newCache(parameters: nil)

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: LMInput.Text(tokens: promptTokens(placeholders: 2))),
                cache: cache, state: nil, prefill: PrefillParameters()))
    }

    func testPlaceholderCountMustMatchEncoderRows() throws {
        let model = try loadCreator()(Data(SharpAIQwen3AsrConfigurationTests.tinyJSON.utf8))
        let cache = try model.newCache(parameters: nil)
        let input = LMInput(
            text: LMInput.Text(tokens: promptTokens(placeholders: Self.placeholderCount - 1)),
            audio: LMInput.ProcessedAudio(features: MLXArray.zeros([1, 128, 100])))

        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters()))
    }
}
