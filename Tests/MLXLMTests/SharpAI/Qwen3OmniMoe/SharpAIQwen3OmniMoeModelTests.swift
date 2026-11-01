// SharpAI fork overlay tests for qwen3_omni_moe registration, audio tower and a tiny forward pass.
// Random weights only. No network and no checkpoint.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIQwen3OmniMoeModelTests: XCTestCase {

    /// Placeholder id for audio in the tiny config.
    private let audioToken: Int32 = 30

    private func tinyConfig() throws -> SharpAIQwen3OmniMoeConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIQwen3OmniMoeConfiguration.self,
            from: Data(SharpAIQwen3OmniMoeTestConfig.tiny().utf8))
    }

    private func tinyModel() throws -> SharpAIQwen3OmniMoeThinker {
        try SharpAIQwen3OmniMoeThinker(try tinyConfig())
    }

    /// `[chunks=3, mel=8, frames=8]` gives one audio token per chunk: 3 tokens in total.
    private func tinyAudio() -> LMInput.ProcessedAudio {
        .init(features: MLXRandom.normal([3, 8, 8]))
    }

    func testAudioAttentionProjectionsAllCarryBias() {
        let attention = SharpAIQwen3OmniMoeAudioAttention(hiddenSize: 8, heads: 2)
        let keys = Set(attention.parameters().flattened().map { $0.0 })
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            XCTAssertTrue(keys.contains("\(projection).bias"), "\(projection).bias missing")
        }
    }

    func testRegistrationBuildsThinkerFromConfigData() throws {
        let creator = try XCTUnwrap(SharpAIQwen3OmniMoeOverlay.creators["qwen3_omni_moe"])
        let model = try creator(Data(SharpAIQwen3OmniMoeTestConfig.tiny().utf8))
        XCTAssertTrue(model is SharpAIQwen3OmniMoeThinker)
    }

    func testRegistrationRejectsInvalidConfig() throws {
        let creator = try XCTUnwrap(SharpAIQwen3OmniMoeOverlay.creators["qwen3_omni_moe"])
        XCTAssertThrowsError(
            try creator(Data(SharpAIQwen3OmniMoeTestConfig.tiny(outputDim: 64).utf8)))
    }

    func testAudioEncoderReturnsOneTokenPerValidChunk() throws {
        let config = try tinyConfig()
        let encoder = SharpAIQwen3OmniMoeAudioEncoder(config.audioConfiguration)

        let full = try encoder(MLXRandom.normal([3, 8, 8]), featureMask: nil)
        XCTAssertEqual(full.shape, [3, 32])

        // The last chunk has no valid frames, so it contributes no tokens.
        var maskValues = [Int32](repeating: 1, count: 24)
        for index in 16 ..< 24 {
            maskValues[index] = 0
        }
        let masked = try encoder(
            MLXRandom.normal([3, 8, 8]), featureMask: MLXArray(maskValues, [3, 8]))
        XCTAssertEqual(masked.shape, [2, 32])
    }

    func testAudioEncoderRejectsChunkLongerThanNWindow() throws {
        let config = try tinyConfig()
        let encoder = SharpAIQwen3OmniMoeAudioEncoder(config.audioConfiguration)
        // 2 * n_window = 8 frames per chunk in the tiny config.
        XCTAssertThrowsError(try encoder(MLXRandom.normal([1, 8, 9]), featureMask: nil))
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
        XCTAssertEqual(output.logits.shape, [1, 3, 64])
    }

    func testAudioForwardMergesEmbeddingsIntoPlaceholders() throws {
        let model = try tinyModel()
        // Three audio placeholders match the three audio tokens from the tiny clip.
        let tokens = MLXArray([5, audioToken, audioToken, audioToken, 7] as [Int32])
            .expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        let result = try model.prepare(
            LMInput(text: .init(tokens: tokens), audio: tinyAudio()), cache: cache, state: nil,
            prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from prepare")
        }
        XCTAssertEqual(output.logits.shape, [1, 5, 64])
    }

    func testPlaceholderCountMustMatchAudioTokens() throws {
        let model = try tinyModel()
        // Two placeholders, but the clip yields three audio tokens.
        let tokens = MLXArray([5, audioToken, audioToken, 7] as [Int32]).expandedDimensions(
            axis: 0)
        let cache = try model.newCache(parameters: nil)

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: tokens), audio: tinyAudio()), cache: cache,
                state: nil, prefill: PrefillParameters())
        ) { error in
            guard let typed = error as? SharpAIQwen3OmniMoeError,
                case .featureTokenMismatch(let placeholders, let features) = typed
            else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(placeholders, 2)
            XCTAssertEqual(features, 3)
        }
    }
}
