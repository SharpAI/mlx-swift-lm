// SharpAI fork overlay: moondream3 tiny-shape forward tests with random weights.
// No real checkpoint is loaded. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIMoondream3ForwardTests: XCTestCase {

    private func makeModel() throws -> SharpAIMoondream3Model {
        SharpAIMoondream3Model(try SharpAIMoondream3TestFixtures.decode())
    }

    /// Five-token image prefix (BOS plus 2x2 vision rows), then two text tokens.
    private func imageTokens() -> MLXArray {
        MLXArray([Int32(31), 31, 31, 31, 31, 1, 2]).expandedDimensions(axis: 0)
    }

    func testTextOnlyForwardHasVocabularyLogits() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(1), 2, 3]).expandedDimensions(axis: 0)

        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testAnswerDelimiterLogitIsSuppressed() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(1), 2]).expandedDimensions(axis: 0)

        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits[0, 0, 3].item(Float.self), -Float.infinity)
        XCTAssertTrue(logits[0, 0, 4].item(Float.self).isFinite)
    }

    func testSingleCropImagePrepareKeepsPromptLength() throws {
        let model = try makeModel()
        let input = LMInput(
            text: .init(tokens: imageTokens()),
            image: .init(
                pixels: MLXArray.zeros([1, 3, 28, 28]),
                frames: [THW(1, 1, 1)]))

        let cache = try model.newCache(parameters: nil)
        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from prepare")
        }
        XCTAssertEqual(output.logits.shape, [1, 7, 32])
    }

    func testMultiCropImagePrepareUsesTiling() throws {
        let model = try makeModel()
        // One global crop plus a 2x2 grid of local crops.
        let input = LMInput(
            text: .init(tokens: imageTokens()),
            image: .init(
                pixels: MLXArray.zeros([5, 3, 28, 28]),
                frames: [THW(1, 2, 2)]))

        let cache = try model.newCache(parameters: nil)
        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from prepare")
        }
        XCTAssertEqual(output.logits.shape, [1, 7, 32])
    }

    func testTilingMismatchThrows() throws {
        let model = try makeModel()
        let input = LMInput(
            text: .init(tokens: imageTokens()),
            image: .init(
                pixels: MLXArray.zeros([5, 3, 28, 28]),
                frames: [THW(1, 1, 3)]))

        let cache = try model.newCache(parameters: nil)
        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters()))
    }

    func testMultiCropWithoutTilingThrows() throws {
        let model = try makeModel()
        let input = LMInput(
            text: .init(tokens: imageTokens()),
            image: .init(pixels: MLXArray.zeros([5, 3, 28, 28])))

        let cache = try model.newCache(parameters: nil)
        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters()))
    }

    func testPromptShorterThanImagePrefixThrows() throws {
        let model = try makeModel()
        let input = LMInput(
            text: .init(tokens: MLXArray([Int32(31), 1]).expandedDimensions(axis: 0)),
            image: .init(
                pixels: MLXArray.zeros([1, 3, 28, 28]),
                frames: [THW(1, 1, 1)]))

        let cache = try model.newCache(parameters: nil)
        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters()))
    }
}
