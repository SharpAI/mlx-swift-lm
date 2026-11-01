// SharpAI fork overlay tests for voxtral_realtime tiny forward passes. Random weights only.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIVoxtralRealtimeModelTests: XCTestCase {

    private func tinyModel() throws -> SharpAIVoxtralRealtimeForConditionalGeneration {
        SharpAIVoxtralRealtimeForConditionalGeneration(
            try SharpAIVoxtralRealtimeTestConfig.decodeTiny())
    }

    /// Mel-first layout: [batch=1, mel=4, time=16]. Gives 8 frames after the stride-2 conv
    /// and 2 projected audio positions after downsampling by 4.
    private func tinyAudio() -> LMInput.ProcessedAudio {
        .init(features: MLXRandom.normal([1, 4, 16]))
    }

    func testAudioTowerHalvesTimeAndKeepsHidden() throws {
        let model = try tinyModel()
        let out = model.audioTower(MLXRandom.normal([1, 4, 16]))
        XCTAssertEqual(out.shape, [1, 8, 8])
    }

    func testProjectorMergesFramesIntoTextWidth() throws {
        let model = try tinyModel()
        let frames = MLXRandom.normal([1, 8, 8])
        let out = model.projector(frames)
        XCTAssertEqual(out.shape, [1, 2, 8])
    }

    func testPrepareWithAudioReturnsVocabularyLogits() throws {
        let model = try tinyModel()
        let tokens = MLXArray([5, 7, 9] as [Int32]).expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        let result = try model.prepare(
            LMInput(text: .init(tokens: tokens), audio: tinyAudio()), cache: cache, state: nil,
            prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from prepare")
        }
        XCTAssertEqual(output.logits.shape, [1, 3, 32])
        XCTAssertNotNil(output.state)
    }

    func testStepAfterPrepareUsesStoredAudio() throws {
        let model = try tinyModel()
        let tokens = MLXArray([5, 7, 9] as [Int32]).expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        let result = try model.prepare(
            LMInput(text: .init(tokens: tokens), audio: tinyAudio()), cache: cache, state: nil,
            prefill: PrefillParameters())
        guard case .logits(let prompt) = result else {
            return XCTFail("expected logits from prepare")
        }

        let next = MLXArray([11] as [Int32]).expandedDimensions(axis: 0)
        let step = model(
            LMInput.Text(tokens: next), cache: cache, state: prompt.state)
        XCTAssertEqual(step.logits.shape, [1, 1, 32])
    }

    func testTextOnlyForwardReturnsVocabularyLogits() throws {
        let model = try tinyModel()
        let tokens = MLXArray([5, 7, 9] as [Int32]).expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        let logits = model(tokens, cache: cache)
        XCTAssertEqual(logits.shape, [1, 3, 32])
    }
}
