import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// Tiny-shape forward passes with random weights. No real checkpoint is loaded.
final class SharpAIPhi4SiglipForwardTests: XCTestCase {

    private func makeModel(_ json: String = SharpAIPhi4SiglipTestFixtures.tinyJSON) throws
        -> SharpAIPhi4SiglipModel
    {
        SharpAIPhi4SiglipModel(try SharpAIPhi4SiglipTestFixtures.decode(json))
    }

    func testTextOnlyForwardHasVocabularyLogits() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(1), 2, 3]).expandedDimensions(axis: 0)

        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testVisionTowerOutputsOneRowPerPatch() throws {
        let model = try makeModel()

        let square = model.model.visionTower(
            MLXArray.zeros([4, 12]), gridHeight: 2, gridWidth: 2)
        XCTAssertEqual(square.shape, [4, 8])

        // A 3x1 grid resamples the 2x2 position table to a different shape.
        let tall = model.model.visionTower(
            MLXArray.zeros([3, 12]), gridHeight: 3, gridWidth: 1)
        XCTAssertEqual(tall.shape, [3, 8])
    }

    func testImagePrepareWritesFeaturesAtImageTokens() throws {
        let model = try makeModel()
        // One 2x2 patch grid gives four image rows, so four `31` tokens in the prompt.
        let tokens = MLXArray([Int32(1), 31, 31, 31, 31, 2]).expandedDimensions(axis: 0)
        let input = LMInput(
            text: .init(tokens: tokens),
            image: .init(
                pixels: MLXArray.zeros([4, 12]), frames: [THW(1, 2, 2)]))

        let cache = try model.newCache(parameters: nil)
        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from prepare")
        }
        XCTAssertEqual(output.logits.shape, [1, 6, 32])
    }

    func testImageTokenCountMismatchThrows() throws {
        let model = try makeModel()
        // Two `31` tokens but four rows of image features.
        let tokens = MLXArray([Int32(31), 1, 31]).expandedDimensions(axis: 0)
        let input = LMInput(
            text: .init(tokens: tokens),
            image: .init(
                pixels: MLXArray.zeros([4, 12]), frames: [THW(1, 2, 2)]))

        let cache = try model.newCache(parameters: nil)
        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters()))
    }

    func testImageWithoutImageTokenIndexThrows() throws {
        let json = SharpAIPhi4SiglipTestFixtures.tinyJSON
            .replacingOccurrences(of: "\"image_token_index\": 31,", with: "")
        let model = try makeModel(json)
        let tokens = MLXArray([Int32(1), 31, 31, 31, 31, 2]).expandedDimensions(axis: 0)
        let input = LMInput(
            text: .init(tokens: tokens),
            image: .init(
                pixels: MLXArray.zeros([4, 12]), frames: [THW(1, 2, 2)]))

        let cache = try model.newCache(parameters: nil)
        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters()))
    }
}
