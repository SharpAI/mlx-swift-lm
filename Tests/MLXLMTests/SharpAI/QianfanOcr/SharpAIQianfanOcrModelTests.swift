// SharpAI fork overlay tests for qianfan_ocr registration, pixel shuffle, and a tiny forward pass.
// Random weights only. No network and no checkpoint.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIQianfanOcrModelTests: XCTestCase {

    /// Placeholder id in the tiny config (`image_token_id`).
    private let imageToken: Int32 = 30

    private func tinyModel() throws -> SharpAIQianfanOcr {
        let config = try SharpAIQianfanOcrTestConfig.decode(SharpAIQianfanOcrTestConfig.tiny())
        return SharpAIQianfanOcr(config)
    }

    /// One 28x28 RGB image, channel-last.
    private func tinyPixels(images: Int = 1) -> MLXArray {
        MLXRandom.normal([images, 28, 28, 3])
    }

    func testRegistrationBuildsModelFromConfigData() throws {
        let creator = try XCTUnwrap(SharpAIQianfanOcrOverlay.creators["qianfan_ocr"])
        let model = try creator(Data(SharpAIQianfanOcrTestConfig.tiny().utf8))
        XCTAssertTrue(model is SharpAIQianfanOcr)
    }

    func testRegistrationRejectsInvalidConfig() throws {
        let creator = try XCTUnwrap(SharpAIQianfanOcrOverlay.creators["qianfan_ocr"])
        XCTAssertThrowsError(try creator(Data(#"{"model_type": "internvl"}"#.utf8)))
    }

    /// Hand-derived from the reference `pixel_shuffle` on a 4x4 grid holding 4 * row + col.
    /// Each output token gathers its 2x2 block in (row, column) order.
    func testPixelShuffleGroupsTwoByTwoBlocks() {
        let grid = MLXArray((0 ..< 16).map { Float($0) }).reshaped(1, 4, 4, 1)

        let shuffled = sharpAIQianfanOcrPixelShuffle(grid, factor: 2)

        XCTAssertEqual(shuffled.shape, [1, 2, 2, 4])
        XCTAssertEqual(
            shuffled.asArray(Float.self),
            [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
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
        XCTAssertEqual(output.logits.dim(2), 40)
        XCTAssertEqual(model.kvHeads.count, 2)
    }

    func testTinyImageForwardMergesFeaturesIntoPlaceholder() throws {
        let model = try tinyModel()
        // Prompt: one text token, one image placeholder (one image token), one text token.
        let tokens = MLXArray([5, imageToken, 7] as [Int32]).expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        let result = try model.prepare(
            LMInput(text: .init(tokens: tokens), image: .init(pixels: tinyPixels())),
            cache: cache, state: nil, prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from prepare")
        }
        XCTAssertEqual(output.logits.dim(0), 1)
        XCTAssertTrue((1 ... 3).contains(output.logits.dim(1)))
        XCTAssertEqual(output.logits.dim(2), 40)
    }

    func testPlaceholderCountMustMatchImageTokens() throws {
        let model = try tinyModel()
        // Two placeholders, but one 28px image gives one image token.
        let tokens = MLXArray([5, imageToken, imageToken, 7] as [Int32]).expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: tokens), image: .init(pixels: tinyPixels())),
                cache: cache, state: nil, prefill: PrefillParameters()))
    }

    func testImageSizeMustMatchNativeResolution() throws {
        let model = try tinyModel()
        let tokens = MLXArray([5, imageToken, 7] as [Int32]).expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)
        // 14px is not the native 28px. Dynamic resolution is not implemented.
        let wrongSize = MLXRandom.normal([1, 14, 14, 3])

        XCTAssertThrowsError(
            try model.prepare(
                LMInput(text: .init(tokens: tokens), image: .init(pixels: wrongSize)),
                cache: cache, state: nil, prefill: PrefillParameters()))
    }
}
