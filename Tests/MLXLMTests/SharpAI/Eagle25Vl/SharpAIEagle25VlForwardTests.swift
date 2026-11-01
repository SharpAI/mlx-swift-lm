import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// Tiny-shape forward passes with random weights. No real checkpoint is loaded.
final class SharpAIEagle25VlForwardTests: XCTestCase {

    private func makeModel(_ json: String = SharpAIEagle25VlTestFixtures.tinyJSON) throws
        -> SharpAIEagle25VlModel
    {
        SharpAIEagle25VlModel(try SharpAIEagle25VlTestFixtures.decode(json))
    }

    func testTextOnlyForwardHasVocabularyLogits() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(1), 2, 3]).expandedDimensions(axis: 0)

        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testImagePrepareWritesFeaturesAtImageTokens() throws {
        let model = try makeModel()
        // One 28x28 tile gives one image token after pixel shuffle, so one `31` in the prompt.
        let tokens = MLXArray([Int32(1), 31, 2]).expandedDimensions(axis: 0)
        let pixels = MLXArray.zeros([1, 3, 28, 28])
        let input = LMInput(
            text: .init(tokens: tokens), image: .init(pixels: pixels))

        let cache = try model.newCache(parameters: nil)
        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from prepare")
        }
        XCTAssertEqual(output.logits.shape, [1, 3, 32])
    }

    func testImageTokenCountMismatchThrows() throws {
        let model = try makeModel()
        // Two `31` tokens but only one tile of features.
        let tokens = MLXArray([Int32(31), 1, 31]).expandedDimensions(axis: 0)
        let input = LMInput(
            text: .init(tokens: tokens),
            image: .init(pixels: MLXArray.zeros([1, 3, 28, 28])))

        let cache = try model.newCache(parameters: nil)
        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters()))
    }

    func testImageWithoutImageTokenIdThrows() throws {
        let json = SharpAIEagle25VlTestFixtures.tinyJSON
            .replacingOccurrences(of: "\"img_context_token_id\": 31,", with: "")
        let model = try makeModel(json)
        let tokens = MLXArray([Int32(1), 31, 2]).expandedDimensions(axis: 0)
        let input = LMInput(
            text: .init(tokens: tokens),
            image: .init(pixels: MLXArray.zeros([1, 3, 28, 28])))

        let cache = try model.newCache(parameters: nil)
        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters()))
    }

    func testPixelShuffleHalvesGridAndDoublesChannels() {
        // [N=1, W=2, H=2, C=1] shuffles to [1, 1, 1, 4].
        let grid = MLXArray((0 ..< 4).map { Float($0) }).reshaped(1, 2, 2, 1)
        let shuffled = SharpAIEagle25VlModel.pixelShuffle(grid, scale: 0.5)

        XCTAssertEqual(shuffled.shape, [1, 1, 1, 4])
    }
}
