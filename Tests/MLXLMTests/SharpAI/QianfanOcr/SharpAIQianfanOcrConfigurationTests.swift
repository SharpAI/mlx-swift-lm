// SharpAI fork overlay tests for qianfan_ocr configuration. Not part of upstream.
// Inline configs only. No network and no checkpoint.

import Foundation
import MLXLMCommon
@testable import MLXVLM
import XCTest

/// Inline configs for qianfan_ocr tests.
enum SharpAIQianfanOcrTestConfig {

    /// The baidu/Qianfan-OCR config.json (fixture in scripts/conformance/offline/fixtures).
    static let hubShape = """
        {
          "architectures": ["QianfanOCRForConditionalGeneration"],
          "downsample_ratio": 0.5,
          "image_token_id": 151671,
          "model_type": "qianfan_ocr",
          "select_layer": -1,
          "tie_word_embeddings": false,
          "text_config": {
            "architectures": ["Qwen3ForCausalLM"],
            "attention_bias": false,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "max_position_embeddings": 32768,
            "model_type": "qwen3",
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_scaling": null,
            "rope_theta": 5000000,
            "vocab_size": 153678
          },
          "vision_config": {
            "architectures": ["QianfanOCRVisionModel"],
            "attention_dropout": 0.0,
            "drop_path_rate": 0.1,
            "hidden_act": "gelu",
            "hidden_size": 1024,
            "image_size": 448,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-06,
            "model_type": "qianfan_ocr_vision",
            "norm_type": "layer_norm",
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 24,
            "patch_size": 14,
            "qkv_bias": true
          }
        }
        """

    /// Small shapes for random-weight forward passes. The image grid is 2x2, so one image gives
    /// one token after the 2x pixel shuffle. The placeholder id 30 is inside the vocabulary.
    static func tiny(
        imageTokenId: Int = 30,
        textHeads: Int = 2,
        textKvHeads: Int = 1,
        tieWordEmbeddings: Bool = false
    ) -> String {
        """
        {
          "model_type": "qianfan_ocr",
          "image_token_id": \(imageTokenId),
          "downsample_ratio": 0.5,
          "vision_config": {
            "model_type": "qianfan_ocr_vision",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "image_size": 28,
            "patch_size": 14,
            "num_channels": 3
          },
          "text_config": {
            "model_type": "qwen3",
            "vocab_size": 40,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": \(textHeads),
            "num_key_value_heads": \(textKvHeads),
            "head_dim": 8,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000,
            "tie_word_embeddings": \(tieWordEmbeddings)
          }
        }
        """
    }

    static func decode(_ json: String) throws -> SharpAIQianfanOcrConfiguration {
        try JSONDecoder.json5().decode(SharpAIQianfanOcrConfiguration.self, from: Data(json.utf8))
    }
}

final class SharpAIQianfanOcrConfigurationTests: XCTestCase {

    func testHubConfigDecodesVisionAndTextFields() throws {
        let config = try SharpAIQianfanOcrTestConfig.decode(SharpAIQianfanOcrTestConfig.hubShape)

        let vision = config.visionConfiguration
        XCTAssertEqual(vision.imageSize, 448)
        XCTAssertEqual(vision.patchSize, 14)
        XCTAssertEqual(vision.hiddenSize, 1024)
        XCTAssertEqual(vision.hiddenLayers, 24)
        XCTAssertEqual(vision.attentionHeads, 16)
        XCTAssertEqual(vision.intermediateSize, 4096)
        XCTAssertTrue(vision.qkvBias)
        XCTAssertTrue(vision.useMeanPooling)

        let text = config.textConfiguration
        XCTAssertEqual(text.modelType, "qwen3")
        XCTAssertEqual(text.hiddenSize, 2560)
        XCTAssertEqual(text.hiddenLayers, 36)
        XCTAssertEqual(text.attentionHeads, 32)
        XCTAssertEqual(text.kvHeads, 8)
        XCTAssertEqual(text.headDim, 128)
        XCTAssertEqual(text.vocabularySize, 153_678)
        XCTAssertEqual(text.ropeTheta, 5_000_000, accuracy: 1)
        XCTAssertFalse(text.tieWordEmbeddings)

        XCTAssertEqual(config.imageTokenId, 151_671)
        XCTAssertEqual(config.downsampleRatio, 0.5, accuracy: 1e-6)
        XCTAssertEqual(config.downsampleFactor, 2)
        XCTAssertEqual(config.projectorHiddenAct, "gelu")
    }

    func testHubConfigValidates() throws {
        let config = try SharpAIQianfanOcrTestConfig.decode(SharpAIQianfanOcrTestConfig.hubShape)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingBlocksFallBackToDefaults() throws {
        let config = try SharpAIQianfanOcrTestConfig.decode(#"{"model_type": "qianfan_ocr"}"#)
        XCTAssertEqual(config.imageTokenId, 151_671)
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 1024)
        XCTAssertEqual(config.textConfiguration.modelType, "qwen3")
        // head_dim falls back to hidden_size / num_attention_heads when the key is absent.
        XCTAssertEqual(config.textConfiguration.headDim, 2560 / 32)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRejectsWrongModelType() throws {
        let config = try SharpAIQianfanOcrTestConfig.decode(#"{"model_type": "internvl"}"#)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsNonQwen3TextModel() throws {
        let json = #"{"model_type": "qianfan_ocr", "text_config": {"model_type": "llama"}}"#
        let config = try SharpAIQianfanOcrTestConfig.decode(json)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsNonIntegerDownsampleRatio() throws {
        let json = #"{"model_type": "qianfan_ocr", "downsample_ratio": 0.3}"#
        let config = try SharpAIQianfanOcrTestConfig.decode(json)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsHeadsNotDivisibleByKeyValueHeads() throws {
        let json = SharpAIQianfanOcrTestConfig.tiny(textHeads: 3, textKvHeads: 2)
        let config = try SharpAIQianfanOcrTestConfig.decode(json)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testTinyConfigValidates() throws {
        let config = try SharpAIQianfanOcrTestConfig.decode(SharpAIQianfanOcrTestConfig.tiny())
        XCTAssertNoThrow(try config.validateModelConfiguration())
        XCTAssertEqual(config.downsampleFactor, 2)
    }
}
