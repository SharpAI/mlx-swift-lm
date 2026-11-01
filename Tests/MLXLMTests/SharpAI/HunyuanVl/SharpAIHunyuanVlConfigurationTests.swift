// SharpAI fork overlay tests for hunyuan_vl config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIHunyuanVlConfigurationTests: XCTestCase {

    private let nestedJSON = """
        {
            "model_type": "hunyuan_vl",
            "text_config": {
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 512,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 64,
                "rms_norm_eps": 0.00001,
                "max_position_embeddings": 4096,
                "rope_theta": 10000.0,
                "tie_word_embeddings": true,
                "rope_scaling": {
                    "type": "xdrope",
                    "alpha": 1000,
                    "xdrope_section": [8, 12, 12, 0]
                }
            },
            "vision_config": {
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "patch_size": 16,
                "spatial_merge_size": 2
            }
        }
        """

    func testDecodesNestedTextAndVisionConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIHunyuanVlConfiguration.self, from: Data(nestedJSON.utf8))

        XCTAssertEqual(config.modelType, "hunyuan_vl")
        XCTAssertEqual(config.textDimensions.vocabSize, 1024)
        XCTAssertEqual(config.textDimensions.hiddenSize, 256)
        XCTAssertEqual(config.textDimensions.kvHeads, 2)
        // head_dim 64: the xdrope section [8, 12, 12, 0] must sum to head_dim / 2 = 32 (upstream rule).
        XCTAssertEqual(config.textDimensions.headDim, 64)
        XCTAssertTrue(config.textDimensions.tieWordEmbeddings)
        XCTAssertEqual(config.visionDimensions?.depth, 2)
        XCTAssertEqual(config.visionDimensions?.patchSize, 16)
    }

    func testParsesXDropeSectionFromRopeScaling() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIHunyuanVlConfiguration.self, from: Data(nestedJSON.utf8))

        XCTAssertEqual(config.textDimensions.ropeScaling?.type, "xdrope")
        XCTAssertEqual(config.textDimensions.ropeScaling?.xdropeSection, [8, 12, 12, 0])
    }

    func testDecodesFlatTextConfigWithDefaults() throws {
        let flat = """
            {
                "model_type": "hunyuan_vl",
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 512,
                "num_attention_heads": 8,
                "max_position_embeddings": 4096
            }
            """
        // Defaults follow HunYuanVLTextConfig (transformers configuration_hunyuan_vl.py).
        let config = try JSONDecoder.json5().decode(
            SharpAIHunyuanVlConfiguration.self, from: Data(flat.utf8))

        XCTAssertEqual(config.textDimensions.kvHeads, 8)
        XCTAssertEqual(config.textDimensions.headDim, 32)
        XCTAssertEqual(config.textDimensions.ropeTheta, 10_000)
        XCTAssertTrue(config.textDimensions.useQkNorm)
        // Upstream default is tie_word_embeddings = True.
        XCTAssertTrue(config.textDimensions.tieWordEmbeddings)
        XCTAssertNil(config.visionDimensions)
        XCTAssertNil(config.textDimensions.ropeScaling)
    }

    func testValidConfigPassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIHunyuanVlConfiguration.self, from: Data(nestedJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRejectsNonDivisibleKVHeads() throws {
        let bad = nestedJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 2", with: "\"num_key_value_heads\": 3")
        let config = try JSONDecoder.json5().decode(
            SharpAIHunyuanVlConfiguration.self, from: Data(bad.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard let factoryError = error as? ModelFactoryError,
                case .invalidConfiguration = factoryError
            else {
                XCTFail("expected ModelFactoryError.invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testRejectsEmptyXDropeSection() throws {
        let bad = nestedJSON.replacingOccurrences(
            of: "\"xdrope_section\": [8, 12, 12, 0]", with: "\"xdrope_section\": []")
        let config = try JSONDecoder.json5().decode(
            SharpAIHunyuanVlConfiguration.self, from: Data(bad.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
