// SharpAI fork overlay tests for the paddleocr_vl model class. Not part of upstream.
// Random weights, tiny shapes, no network. Shapes and positions only; values are not checked.
// UNVERIFIED: this file has not been compiled or run.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIPaddleocrVlModelTests: XCTestCase {

    private func makeModel() throws -> SharpAIPaddleocrVl {
        let config = try JSONDecoder.json5().decode(
            SharpAIPaddleocrVlConfiguration.self,
            from: Data(SharpAIPaddleocrVlConfigurationTests.tinyJSON.utf8))
        return SharpAIPaddleocrVl(config)
    }

    func testTextForwardReturnsVocabularyLogits() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)
        let output = model.callAsFunction(
            LMInput.Text(tokens: tokens), cache: nil, state: nil)

        XCTAssertEqual(output.logits.shape, [1, 3, 128])
    }

    func testVisionRotaryUsesRowThenColumnAngles() {
        let rotary = sharpAIPaddleocrVlVisionRotary(height: 2, width: 3, headDim: 8)
        XCTAssertEqual(rotary.cos.shape, [1, 1, 6, 8])
        XCTAssertEqual(rotary.sin.shape, [1, 1, 6, 8])
        // Patch 4 is row 1, column 1; with headDim 8 each axis has two frequencies (1, 0.01).
        let angles = rotary.sin[0, 0, 4].asArray(Float.self)
        let expected: [Float] = [sinf(1), sinf(0.01), sinf(1), sinf(0.01)]
        for index in 0 ..< 4 {
            XCTAssertEqual(angles[index], expected[index], accuracy: 1e-4)
            XCTAssertEqual(angles[index + 4], expected[index], accuracy: 1e-4)
        }
        // Row 0, column 0 is the identity rotation.
        XCTAssertEqual(rotary.sin[0, 0, 0].asArray(Float.self), [Float](repeating: 0, count: 8))
    }

    func testTextPositionsAreSequentialOnEveryAxis() {
        let positions = SharpAIPaddleocrVl.textPositions(start: 4, length: 3)

        XCTAssertEqual(positions.shape, [3, 1, 3])
        XCTAssertEqual(positions.asArray(Int32.self), [4, 5, 6, 4, 5, 6, 4, 5, 6])
    }

    func testMropePositionsForTextStartAtAnchor() throws {
        let (positions, next) = try SharpAIPaddleocrVl.mropePositions(
            tokenIds: [5, 6, 7], frames: [], imageTokenId: 9, mergeSize: 2, anchor: 4)

        XCTAssertEqual(positions.asArray(Int32.self), [4, 5, 6, 4, 5, 6, 4, 5, 6])
        XCTAssertEqual(next, 7)
    }

    func testMropePositionsUseMergedGridForImageBlock() throws {
        // A 4x4 patch grid merges to 2x2 image tokens. The text after it starts past the
        // larger grid offset, at 1 + 2 = 3.
        let (positions, next) = try SharpAIPaddleocrVl.mropePositions(
            tokenIds: [1, 9, 9, 9, 9, 2], frames: [THW(1, 4, 4)],
            imageTokenId: 9, mergeSize: 2, anchor: 0)

        XCTAssertEqual(
            positions.asArray(Int32.self),
            [0, 1, 1, 1, 1, 3, 0, 1, 1, 2, 2, 3, 0, 1, 2, 1, 2, 3])
        XCTAssertEqual(next, 4)
    }

    func testMropePositionsRejectPlaceholderCountMismatch() {
        XCTAssertThrowsError(
            try SharpAIPaddleocrVl.mropePositions(
                tokenIds: [1, 9, 2], frames: [THW(1, 4, 4)],
                imageTokenId: 9, mergeSize: 2, anchor: 0))
    }

    func testInterpolatedPositionsResizeToRequestedGrid() throws {
        let model = try makeModel()
        let embeddings = model.visionTower.embeddings

        XCTAssertEqual(embeddings.interpolatedPositions(height: 4, width: 4).shape, [16, 16])
        XCTAssertEqual(embeddings.interpolatedPositions(height: 6, width: 6).shape, [36, 16])
    }

    func testImageFeaturesMatchMergedPatchGrid() throws {
        let model = try makeModel()
        let pixels = MLXRandom.normal([16, 14, 14, 3])
        let features = try model.imageFeatures(pixels: pixels, frames: [THW(1, 4, 4)])

        XCTAssertEqual(features.shape, [4, 16])
    }

    func testPrepareMergesImageAndCarriesRopeDelta() throws {
        let model = try makeModel()
        let cache = try model.newCache(parameters: nil)
        let tokens = MLXArray([Int32(1), 9, 9, 9, 9, 2]).reshaped(1, 6)
        let image = LMInput.ProcessedImage(
            pixels: MLXRandom.normal([16, 14, 14, 3]), frames: [THW(1, 4, 4)])
        let input = LMInput(text: .init(tokens: tokens), image: image)

        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())
        guard case .logits(let output) = result else {
            return XCTFail("expected logits from prepare")
        }

        XCTAssertEqual(output.logits.dim(-1), 128)
        // Next position is 4, and the cache holds 6 positions, so the delta is -2.
        XCTAssertEqual(output.state?[sharpAIPaddleocrVlRopeDeltaKey], -2)

        let next = model.callAsFunction(
            LMInput.Text(tokens: MLXArray([Int32(3)]).reshaped(1, 1)),
            cache: cache, state: output.state)
        XCTAssertEqual(next.logits.shape, [1, 1, 128])
    }
}
