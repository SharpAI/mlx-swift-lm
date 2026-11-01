// SharpAI fork overlay tests for glmasr registration and a tiny forward pass.
// Random weights only. No network and no checkpoint.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIGlmasrModelTests: XCTestCase {

    /// Placeholder id in the tiny config (`audio_token_id`).
    private let audioToken: Int32 = 30

    private func tinyModel() throws -> SharpAIGlmasr {
        let config = try SharpAIGlmasrTestConfig.decode(SharpAIGlmasrTestConfig.tiny())
        return SharpAIGlmasr(config)
    }

    /// `[batch=1, mel=4, frames=15]` gives 8 conv frames and 2 audio tokens.
    private func tinyAudio() -> LMInput.ProcessedAudio {
        .init(features: MLXRandom.normal([1, 4, 15]))
    }

    func testRegistrationBuildsModelFromConfigData() throws {
        let creator = try XCTUnwrap(SharpAIGlmasrOverlay.creators["glmasr"])
        let model = try creator(Data(SharpAIGlmasrTestConfig.tiny().utf8))
        XCTAssertTrue(model is SharpAIGlmasr)
    }

    func testRegistrationRejectsInvalidConfig() throws {
        let creator = try XCTUnwrap(SharpAIGlmasrOverlay.creators["glmasr"])
        let json = SharpAIGlmasrTestConfig.tiny(intermediate: 16)
        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testTextOnlyPrepareReturnsVocabularyLogits() throws {
        let model = try tinyModel()
        let tokens = MLXArray([5, 7, 9] as [Int32]).expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        let result = try model.prepare(
            LMInput(text: .init(tokens: tokens)), cache: cache, state: nil,
            prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from prepare")
        }
        XCTAssertEqual(output.logits.dim(0), 1)
        XCTAssertTrue((1 ... 3).contains(output.logits.dim(1)))
        XCTAssertEqual(output.logits.dim(2), 32)
    }

    func testTinyAudioForwardMergesEmbeddingsIntoPlaceholders() throws {
        let model = try tinyModel()
        // Prompt: 1 text token, 2 audio placeholders, 1 text token.
        let tokens = MLXArray([5, audioToken, audioToken, 7] as [Int32])
            .expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        let result = try model.prepare(
            LMInput(text: .init(tokens: tokens), audio: tinyAudio()), cache: cache, state: nil,
            prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from prepare")
        }
        XCTAssertEqual(output.logits.dim(0), 1)
        XCTAssertTrue((1 ... 4).contains(output.logits.dim(1)))
        XCTAssertEqual(output.logits.dim(2), 32)
    }

    func testPlaceholderCountMustMatchAudioTokens() throws {
        let model = try tinyModel()
        // One placeholder, but the 15-frame clip yields 2 audio tokens.
        let tokens = MLXArray([5, audioToken, 7] as [Int32]).expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: tokens), audio: tinyAudio()), cache: cache,
                state: nil, prefill: PrefillParameters())
        ) { error in
            guard
                case SharpAIGlmasrError.audioTokenMismatch(let placeholders, let features)? =
                    error as? SharpAIGlmasrError
            else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(placeholders, 1)
            XCTAssertEqual(features, 2)
        }
    }

    func testMelAxisMustBeSecondDimension() throws {
        let model = try tinyModel()
        // Frames-first input [1, 15, 4] has the wrong mel dimension.
        let wrongLayout = LMInput.ProcessedAudio(features: MLXRandom.normal([1, 15, 4]))
        let tokens = MLXArray([5, audioToken, audioToken, 7] as [Int32]).expandedDimensions(
            axis: 0)
        let cache = try model.newCache(parameters: nil)

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: tokens), audio: wrongLayout), cache: cache,
                state: nil, prefill: PrefillParameters()))
    }

    func testShortAudioBelowOneTokenGroupIsRejected() throws {
        let model = try tinyModel()
        // 6 frames give 3 conv frames, fewer than one group of 4.
        let short = LMInput.ProcessedAudio(features: MLXRandom.normal([1, 4, 6]))
        let tokens = MLXArray([5, audioToken, 7] as [Int32]).expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: tokens), audio: short), cache: cache,
                state: nil, prefill: PrefillParameters()))
    }
}
