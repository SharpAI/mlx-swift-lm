// SharpAI fork overlay tests for the bagel model: tiny random-weight forward pass and registry
// creator. Not part of upstream. No weights are downloaded.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIBagelForwardTests: XCTestCase {

    private func makeModel() throws -> SharpAIBagel {
        let config = try JSONDecoder.json5().decode(
            SharpAIBagelConfiguration.self,
            from: Data(SharpAIBagelTestFixtures.tinyJSON.utf8))
        return SharpAIBagel(config)
    }

    func testTextOnlyForwardReturnsLogitsPerToken() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)

        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 4, 32])
    }

    func testImageFeaturesMatchPatchGridAndTextWidth() throws {
        let model = try makeModel()
        // NCHW: one 3-channel 28x28 image gives a 2x2 patch grid.
        let pixels = MLXArray.zeros([1, 3, 28, 28])

        let features = try model.imageFeatures(pixels)

        XCTAssertEqual(features.shape, [1, 4, 16])
    }

    func testImageFeaturesRejectGridLargerThanPositionTable() throws {
        let model = try makeModel()
        // 42x42 with 14-pixel patches is a 3x3 grid, larger than vit_max_num_patch_per_side = 2.
        let pixels = MLXArray.zeros([1, 3, 42, 42])

        XCTAssertThrowsError(try model.imageFeatures(pixels))
    }

    func testPrepareRejectsExtraImagePlaceholders() throws {
        let model = try makeModel()
        // The 2x2 grid gives 4 features; the prompt carries 5 placeholders (token 31).
        let tokens = MLXArray([Int32(1), 31, 31, 31, 31, 31, 2]).reshaped(1, 7)
        let image = LMInput.ProcessedImage(pixels: MLXArray.zeros([1, 3, 28, 28]))
        let input = LMInput(text: .init(tokens: tokens), image: image)
        let cache = try model.newCache(parameters: nil)

        XCTAssertThrowsError(try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters()))
    }

    func testRegistryCreatesBagelModel() throws {
        let creator = try XCTUnwrap(SharpAIBagelOverlay.creators["bagel"])
        let model = try creator(Data(SharpAIBagelTestFixtures.tinyJSON.utf8))

        XCTAssertTrue(model is SharpAIBagel)
    }

    func testRegistryRejectsConfigWithoutQKNorm() throws {
        let creator = try XCTUnwrap(SharpAIBagelOverlay.creators["bagel"])
        let json = SharpAIBagelTestFixtures.tinyJSON.replacingOccurrences(
            of: "\"qk_norm\": true", with: "\"qk_norm\": false")

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }
}
