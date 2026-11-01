// SharpAI fork overlay tests for moss_transcribe_diarize registration and a tiny forward pass.
// Random weights only. No network and no checkpoint.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIMossTranscribeDiarizeModelTests: XCTestCase {

    /// Placeholder id in the tiny config (`audio_token_id`).
    private let audioToken: Int32 = 30

    private func tinyModel() throws -> SharpAIMossTranscribeDiarize {
        let config = try SharpAIMossTranscribeDiarizeTestConfig.decode(
            SharpAIMossTranscribeDiarizeTestConfig.tiny())
        return SharpAIMossTranscribeDiarize(config)
    }

    /// `[batch=1, mel=4, frames=15]` gives 8 conv frames and 4 audio tokens (factor 2).
    private func tinyAudio() -> LMInput.ProcessedAudio {
        .init(features: MLXRandom.normal([1, 4, 15]))
    }

    func testRegistrationBuildsModelFromConfigData() throws {
        let creator = try XCTUnwrap(
            SharpAIMossTranscribeDiarizeOverlay.creators["moss_transcribe_diarize"])
        let model = try creator(Data(SharpAIMossTranscribeDiarizeTestConfig.tiny().utf8))
        XCTAssertTrue(model is SharpAIMossTranscribeDiarize)
    }

    func testRegistrationRejectsInvalidConfig() throws {
        let creator = try XCTUnwrap(
            SharpAIMossTranscribeDiarizeOverlay.creators["moss_transcribe_diarize"])
        let json = SharpAIMossTranscribeDiarizeTestConfig.tiny(activation: "relu")
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
        // Balanced prefill runs the leading positions in chunks and keeps the last one for the
        // logits forward. With a 3-token prompt that is 1 to 3 logit positions, not always 3.
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
        XCTAssertEqual(output.logits.dim(0), 1)
        // The merged prompt has 6 positions. Chunked prefill leaves only the tail for the logits
        // forward, so the logits cover 1 to 6 positions.
        XCTAssertTrue((1 ... 6).contains(output.logits.dim(1)))
        XCTAssertEqual(output.logits.dim(2), 32)
    }

    func testPlaceholderCountMustMatchAudioTokens() throws {
        let model = try tinyModel()
        // One placeholder, but the 15-frame clip yields 4 audio tokens.
        let tokens = MLXArray([5, audioToken, 7] as [Int32]).expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: tokens), audio: tinyAudio()), cache: cache,
                state: nil, prefill: PrefillParameters()))
    }

    func testRejectsAudioLongerThanPositionTable() throws {
        let model = try tinyModel()
        // 65 frames give 33 conv frames, one more than `max_source_positions` (32).
        let tokens = MLXArray([5, audioToken, audioToken, audioToken, audioToken] as [Int32])
            .expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)
        let audio = LMInput.ProcessedAudio(features: MLXRandom.normal([1, 4, 65]))

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: tokens), audio: audio), cache: cache,
                state: nil, prefill: PrefillParameters()))
    }
}
