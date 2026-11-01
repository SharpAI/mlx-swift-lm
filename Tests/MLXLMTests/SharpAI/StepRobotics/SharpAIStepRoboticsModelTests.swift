// SharpAI fork overlay tests for the StepVL model. Not part of upstream.
//
// Offline checks only: a tiny config with random weights, one forward pass per path, no network.
// UNVERIFIED: none of these shapes were compiled or run in this change.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIStepRoboticsModelTests: XCTestCase {

    /// Vision: one 28 px image, 14 px patches, so a 2x2 grid that downsamples to 1 token.
    /// Text: width 32, 2 layers, layer 1 is MoE, vocabulary 64, image placeholder id 63.
    private let tinyJSON = """
        {
          "model_type": "step_robotics",
          "projector_bias": true,
          "image_token_id": 63,
          "vision_config": {
            "width": 16, "layers": 1, "heads": 2, "patch_size": 14, "image_size": 28,
            "mlp_ratio": 2.0, "hidden_act": "gelu"
          },
          "text_config": {
            "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
            "intermediate_size": 48, "vocab_size": 64, "head_dim": 8, "share_q_dim": 16,
            "rms_norm_eps": 1e-6, "moe_num_experts": 4, "moe_top_k": 2,
            "moe_intermediate_size": 16, "share_expert_dim": 16, "moe_layers_enum": "1"
          }
        }
        """

    private func tinyModel() throws -> SharpAIStepRoboticsModel {
        let config = try JSONDecoder.json5().decode(
            SharpAIStepRoboticsConfiguration.self, from: Data(tinyJSON.utf8))
        try config.validateModelConfiguration()
        return try SharpAIStepRoboticsModel(config)
    }

    /// NCHW pixels `[1, 3, 28, 28]`.
    private func tinyPixels() -> MLXArray {
        MLXArray.zeros([1, 3, 28, 28])
    }

    func testImageFeaturesHaveOneTokenOfTextWidth() throws {
        let model = try tinyModel()

        let features = try model.imageFeatures(tinyPixels(), dtype: .float32)

        XCTAssertEqual(features.shape, [1, 1, 32])
    }

    func testTextOnlyPrepareReturnsLogitsForEveryPosition() throws {
        let model = try tinyModel()
        let cache = try model.newCache(parameters: nil)
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)])

        let result = try model.prepare(
            LMInput(tokens: tokens), cache: cache, state: nil, prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from the text path")
        }
        XCTAssertEqual(output.logits.shape, [1, 3, 64])
    }

    func testMultimodalEmbeddingsFillThePlaceholderSlot() throws {
        let model = try tinyModel()
        let tokens = MLXArray([Int32(1), Int32(2), Int32(63), Int32(3)]).expandedDimensions(axis: 0)

        let embeddings = try model.multimodalEmbeddings(tokens, pixelValues: tinyPixels())

        XCTAssertEqual(embeddings.shape, [1, 4, 32])
    }

    func testImagePrepareReturnsLogitsForEveryPosition() throws {
        let model = try tinyModel()
        let cache = try model.newCache(parameters: nil)
        let tokens = MLXArray([Int32(1), Int32(2), Int32(63), Int32(3)]).expandedDimensions(axis: 0)
        let input = LMInput(
            text: .init(tokens: tokens), image: .init(pixels: tinyPixels()))

        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected logits from the image path")
        }
        XCTAssertEqual(output.logits.shape, [1, 4, 64])
    }

    func testPlaceholderCountMismatchThrows() throws {
        let model = try tinyModel()
        // No placeholder token, but one image feature row.
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).expandedDimensions(axis: 0)

        XCTAssertThrowsError(
            try model.multimodalEmbeddings(tokens, pixelValues: tinyPixels())
        ) { error in
            guard case SharpAIStepRoboticsError.placeholderMismatch(let placeholders, let features) =
                error
            else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(placeholders, 0)
            XCTAssertEqual(features, 1)
        }
    }

    func testWrongImageSizeThrows() throws {
        let model = try tinyModel()
        let pixels = MLXArray.zeros([1, 3, 30, 30])

        XCTAssertThrowsError(try model.imageFeatures(pixels, dtype: .float32))
    }

    func testWrongChannelCountThrows() throws {
        let model = try tinyModel()
        let pixels = MLXArray.zeros([1, 4, 28, 28])

        XCTAssertThrowsError(try model.imageFeatures(pixels, dtype: .float32))
    }
}
