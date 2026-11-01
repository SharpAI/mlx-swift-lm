// SharpAI fork overlay tests for the minicpmv4_6 model class. Not part of upstream.
// Random weights, tiny shapes, no network. Shapes only; values are not checked.
// UNVERIFIED: this file has not been compiled or run.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIMinicpmv46ModelTests: XCTestCase {

    private func makeModel() throws -> SharpAIMinicpmv46 {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv46Configuration.self,
            from: Data(SharpAIMinicpmv46SanitizeTests.tinyJSON.utf8))
        return try SharpAIMinicpmv46(config)
    }

    func testEncodeSliceProducesOneTokenPerWindowedMergedGrid() throws {
        let model = try makeModel()
        // 56px / patch 14 = 4x4 patches. The window merger gives 2x2, the MLP merger gives 1x1.
        let slice = MLXRandom.normal([56, 56, 3])
        let tokens = try model.encodeSlice(slice)

        XCTAssertEqual(tokens.shape, [1, 32])
    }

    func testEncodeSlicesConcatenatesSliceTokens() throws {
        let model = try makeModel()
        let slices = MLXRandom.normal([2, 56, 56, 3])
        let tokens = try model.encodeSlices(slices)

        XCTAssertEqual(tokens.shape, [2, 32])
    }

    func testRejectsGridThatDoesNotDivideByMergeFactor() throws {
        let model = try makeModel()
        // 42px / 14 = 3 patches per side, which is not divisible by 4.
        let slice = MLXRandom.normal([42, 42, 3])

        XCTAssertThrowsError(try model.encodeSlice(slice))
    }

    func testTextForwardReturnsVocabularyLogits() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)
        let output = model.callAsFunction(
            LMInput.Text(tokens: tokens), cache: nil, state: nil)

        XCTAssertEqual(output.logits.shape, [1, 3, 128])
    }
}
