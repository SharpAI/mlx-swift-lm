import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

/// Tiny-shape forward passes with random weights. No real checkpoint is loaded.
final class SharpAIKanana15VForwardTests: XCTestCase {

    private func makeModel(_ json: String = SharpAIKanana15VTestFixtures.tinyJSON) throws
        -> SharpAIKanana15VModel
    {
        SharpAIKanana15VModel(try SharpAIKanana15VTestFixtures.decode(json))
    }

    func testTextOnlyForwardHasVocabularyLogits() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(1), 2, 3]).expandedDimensions(axis: 0)

        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testTextOnlyPrepareReturnsLogits() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(1), 2, 3]).expandedDimensions(axis: 0)
        let input = LMInput(text: .init(tokens: tokens))

        let cache = try model.newCache(parameters: nil)
        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from prepare")
        }
        XCTAssertEqual(output.logits.shape, [1, 3, 32])
    }

    func testImageInputThrowsProjectorNotImplemented() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(1), 31, 2]).expandedDimensions(axis: 0)
        let input = LMInput(
            text: .init(tokens: tokens),
            image: .init(pixels: MLXArray.zeros([1, 3, 8, 8])))

        let cache = try model.newCache(parameters: nil)
        do {
            _ = try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters())
            XCTFail("expected projectorNotImplemented")
        } catch let error as SharpAIKanana15VError {
            guard case .projectorNotImplemented = error else {
                return XCTFail("wrong error case: \(error)")
            }
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testVisionEncoderOutputsOneRowPerPatch() throws {
        let config = try SharpAIKanana15VTestFixtures.decode()
        let vision = SharpAIKanana15VVisionModel(config.visionConfiguration)

        // 4x4 patch grid. Each patch holds 3 channels x 2 frames x 2 x 2 values.
        let pixels = MLXArray.zeros([16, 24])
        let output = vision(pixels, frames: [THW(1, 4, 4)])

        XCTAssertEqual(output.shape, [16, 8])
    }

    func testFastGeluMatchesQuickGeluFormula() {
        // quick_gelu(x) = x * sigmoid(1.702 * x). Checked at x = 1.
        let value = geluFastApproximate(MLXArray([Float(1.0)])).item(Float.self)
        let expected: Float = 1.0 / (1.0 + Foundation.exp(Float(-1.702)))

        XCTAssertEqual(value, expected, accuracy: 1e-4)
    }

    func testRegistrationCreatesModelFromConfigJSON() throws {
        let creator = try XCTUnwrap(SharpAIKanana15VOverlay.creators["kanana-1.5-v"])
        let model = try creator(Data(SharpAIKanana15VTestFixtures.tinyJSON.utf8))

        XCTAssertTrue(model is SharpAIKanana15VModel)
        XCTAssertNil(SharpAIKanana15VOverlay.creators["kanana-1.5"])
    }

    // Known gap: the projector (dynamic-c-abs, RegStage) is not built.
    // UNVERIFIED: no reference source for its layout. Image path is covered above.
}
