// SharpAI fork overlay tests for audioflamingo3 registration and a tiny forward pass.
// Random weights only. No network and no checkpoint.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIAudioflamingo3ModelTests: XCTestCase {

    /// Placeholder id in the tiny config (`audio_token_id`).
    private let audioToken: Int32 = 30

    private func tinyModel() throws -> SharpAIAudioflamingo3 {
        let config = try SharpAIAudioflamingo3TestConfig.decode(
            SharpAIAudioflamingo3TestConfig.tiny())
        return SharpAIAudioflamingo3(config)
    }

    /// `[chunks=1, mel=4, frames=16]` gives 4 audio tokens.
    private func tinyAudio() -> LMInput.ProcessedAudio {
        .init(features: MLXRandom.normal([1, 4, 16]))
    }

    func testRegistrationBuildsModelFromConfigData() throws {
        let creator = try XCTUnwrap(SharpAIAudioflamingo3Overlay.creators["audioflamingo3"])
        let model = try creator(Data(SharpAIAudioflamingo3TestConfig.tiny().utf8))
        XCTAssertTrue(model is SharpAIAudioflamingo3)
    }

    func testRegistrationRejectsInvalidConfig() throws {
        let creator = try XCTUnwrap(SharpAIAudioflamingo3Overlay.creators["audioflamingo3"])
        let json = SharpAIAudioflamingo3TestConfig.tiny(scaleEmbedding: true)
        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testTextOnlyForwardReturnsVocabularyLogits() throws {
        let model = try tinyModel()
        let tokens = MLXArray([5, 7, 9] as [Int32]).expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        let result = try model.prepare(
            LMInput(text: .init(tokens: tokens)), cache: cache, state: nil,
            prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from prepare")
        }
        // prepare returns logits for the unprocessed tail, so the position count is at most 3.
        XCTAssertEqual(output.logits.dim(0), 1)
        XCTAssertTrue((1 ... 3).contains(output.logits.dim(1)))
        XCTAssertEqual(output.logits.dim(2), 32)
    }

    func testTinyAudioForwardMergesEmbeddingsIntoPlaceholders() throws {
        let model = try tinyModel()
        // Prompt: 1 text token, 4 audio placeholders, 1 text token.
        let tokens = MLXArray([5, audioToken, audioToken, audioToken, audioToken, 7] as [Int32])
            .expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        let result = try model.prepare(
            LMInput(text: .init(tokens: tokens), audio: tinyAudio()), cache: cache, state: nil,
            prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from prepare")
        }
        // Logits cover the unprocessed tail of the 6 merged positions.
        XCTAssertEqual(output.logits.dim(0), 1)
        XCTAssertTrue((1 ... 6).contains(output.logits.dim(1)))
        XCTAssertEqual(output.logits.dim(2), 32)
    }

    func testPlaceholderCountMustMatchAudioTokens() throws {
        let model = try tinyModel()
        // Two placeholders, but the 16-frame clip yields 4 audio tokens.
        let tokens = MLXArray([5, audioToken, audioToken, 7] as [Int32])
            .expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: tokens), audio: tinyAudio()), cache: cache,
                state: nil, prefill: PrefillParameters())
        ) { error in
            guard
                case SharpAIAudioflamingo3Error.audioTokenMismatch(
                    let placeholders, let features)? = error as? SharpAIAudioflamingo3Error
            else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(placeholders, 2)
            XCTAssertEqual(features, 4)
        }
    }

    func testAudioFeaturesMustUseMelFirstLayout() throws {
        let model = try tinyModel()
        // Frames-first input [1, 16, 4] has the wrong mel dimension.
        let wrongLayout = LMInput.ProcessedAudio(features: MLXRandom.normal([1, 16, 4]))
        let tokens = MLXArray([5, audioToken, audioToken, audioToken, audioToken, 7] as [Int32])
            .expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: tokens), audio: wrongLayout), cache: cache,
                state: nil, prefill: PrefillParameters()))
    }
}
