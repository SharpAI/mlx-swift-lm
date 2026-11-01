// SharpAI fork overlay tests for the OpenVLA model. Not part of upstream.
//
// Offline checks only: a tiny text config and a tiny vision spec with random weights, no network.
// UNVERIFIED: none of these shapes were compiled or run in this change.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

extension SharpAIOpenvlaVisionSpec {
    /// DINOv2-like and SigLIP-like backbones at 28 px with 14 px patches (4 patches).
    static let tiny = SharpAIOpenvlaVisionSpec(
        dino: SharpAIOpenvlaViTSpec(
            hiddenSize: 16, depth: 2, heads: 2, mlpHiddenSize: 32, patchSize: 14,
            imageSize: 28, registerTokens: 2, hasClassToken: true, layerScale: true,
            approximateGelu: false),
        siglip: SharpAIOpenvlaViTSpec(
            hiddenSize: 12, depth: 2, heads: 2, mlpHiddenSize: 24, patchSize: 14,
            imageSize: 28, registerTokens: 0, hasClassToken: false, layerScale: false,
            approximateGelu: true)
    )
}

final class SharpAIOpenvlaModelTests: XCTestCase {

    /// Text width 32, 2 layers, 64-token vocabulary.
    private func tinyConfiguration() throws -> SharpAIOpenvlaConfiguration {
        let json = """
            {
              "model_type": "openvla",
              "text_config": {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 64
              }
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIOpenvlaConfiguration.self, from: Data(json.utf8))
    }

    /// NCHW pixels with the 6 channels the model expects: 3 DINOv2 and 3 SigLIP.
    private func tinyPixels() -> MLXArray {
        MLXArray.zeros([1, 6, 28, 28])
    }

    func testImageFeaturesHaveFusedWidthAndPatchCount() throws {
        let model = SharpAIOpenvla(try tinyConfiguration(), vision: .tiny)

        let features = try model.imageFeatures(tinyPixels(), dtype: .float32)

        // 4 patches, projected to the text width of 32.
        XCTAssertEqual(features.shape, [1, 4, 32])
    }

    func testMultimodalEmbeddingsPlaceImagePatchesAfterBOS() throws {
        let model = SharpAIOpenvla(try tinyConfiguration(), vision: .tiny)
        let tokens = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let embeddings = try model.multimodalEmbeddings(tokens, pixelValues: tinyPixels())

        // BOS, 4 image patches, then the 2 remaining text tokens.
        XCTAssertEqual(embeddings.shape, [1, 7, 32])
    }

    func testPrepareReturnsLogitsForTheFullSequence() throws {
        let model = SharpAIOpenvla(try tinyConfiguration(), vision: .tiny)
        let tokens = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let cache = try model.newCache(parameters: nil)
        let input = LMInput(
            text: .init(tokens: tokens), image: .init(pixels: tinyPixels()))

        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            return XCTFail("expected prepare to return logits")
        }
        XCTAssertEqual(output.logits.shape, [1, 7, 64])
    }

    func testTextOnlyForwardHasVocabularyLogits() throws {
        let model = SharpAIOpenvla(try tinyConfiguration(), vision: .tiny)
        let tokens = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 64])
    }

    func testRejectsPixelsWithoutSixChannels() throws {
        let model = SharpAIOpenvla(try tinyConfiguration(), vision: .tiny)
        let threeChannel = MLXArray.zeros([1, 3, 28, 28])

        XCTAssertThrowsError(try model.imageFeatures(threeChannel, dtype: .float32)) { error in
            XCTAssertEqual(
                error as? SharpAIOpenvlaError, SharpAIOpenvlaError.imageChannelMismatch(3))
        }
    }
}
