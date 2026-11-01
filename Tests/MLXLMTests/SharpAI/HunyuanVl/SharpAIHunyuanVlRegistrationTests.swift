// SharpAI fork overlay tests for the hunyuan_vl creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIHunyuanVlRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "hunyuan_vl",
            "text_config": {
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 512,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "max_position_embeddings": 4096,
                "head_dim": 64,
                "rope_scaling": {"type": "xdrope", "xdrope_section": [8, 12, 12, 0]}
            },
            "vision_config": {
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4
            }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIHunyuanVlOverlay.creators["hunyuan_vl"])
    }

    func testValidConfigFailsUntilForwardIsImplemented() {
        guard let creator = SharpAIHunyuanVlOverlay.creators["hunyuan_vl"] else {
            XCTFail("hunyuan_vl creator missing")
            return
        }

        do {
            _ = try creator(Data(validJSON.utf8))
            XCTFail("creator must not return a model yet")
        } catch let error as SharpAIHunyuanVlError {
            guard case .forwardNotImplemented(let type) = error else {
                XCTFail("unexpected SharpAIHunyuanVlError case")
                return
            }
            XCTAssertEqual(type, "hunyuan_vl")
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testInvalidConfigFailsBeforeForwardError() {
        guard let creator = SharpAIHunyuanVlOverlay.creators["hunyuan_vl"] else {
            XCTFail("hunyuan_vl creator missing")
            return
        }
        let invalid = Data(
            """
            {
                "model_type": "hunyuan_vl",
                "text_config": {
                    "vocab_size": 1024,
                    "hidden_size": 256,
                    "num_hidden_layers": 4,
                    "intermediate_size": 512,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 3,
                    "max_position_embeddings": 4096
                }
            }
            """.utf8)

        XCTAssertThrowsError(try creator(invalid)) { error in
            guard let factoryError = error as? ModelFactoryError,
                case .invalidConfiguration = factoryError
            else {
                XCTFail("expected ModelFactoryError.invalidConfiguration, got \(error)")
                return
            }
        }
    }

    /// Random-weight forward through the text decoder (M-RoPE, Q/K norm, tied head) and the vision
    /// tower with its patch merger. Shapes follow HF `HunYuanVLVisionPatchMerger`: h' * (w' + 1) + 2.
    func testTinyForwardPassWithRandomWeights() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIHunyuanVlConfiguration.self, from: Data(validJSON.utf8))
        try config.validateModelConfiguration()
        let model = SharpAIHunyuanVlModel(config)

        let tokens = MLXArray([Int32(1), 5, 9, 3]).reshaped(1, 4)
        let logits = model(tokens)
        XCTAssertEqual(logits.shape, [1, 4, 1024])
        XCTAssertFalse(MLX.isNaN(logits).any().item(Bool.self), "NaN logits")

        // One 4x4 patch grid, spatial_merge_size 2: h' = w' = 2, so 2 * (2 + 1) + 2 = 8 tokens.
        let pixels = MLXArray.ones([16, 3 * 16 * 16])
        let features = try XCTUnwrap(model.imageFeatures(pixels, grids: [(h: 4, w: 4)]))
        XCTAssertEqual(features.shape, [1, 8, 256])
        XCTAssertFalse(MLX.isNaN(features).any().item(Bool.self), "NaN image features")
    }
}
