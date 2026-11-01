// SharpAI fork overlay tests for the granite_speech_plus configuration. Not part of upstream.
// Inline JSON only. No checkpoint, no network.
// The encoder and projector sizes in the fixture are tiny so that a random-weight model is cheap.
// Field names follow transformers granite_speech / granite_speech_plus (see the overlay sources).
// UNVERIFIED: this file was written without compiling. Run it before relying on it.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

enum SharpAIGraniteSpeechPlusFixtures {
    /// Tiny config with yarn in `rope_parameters`, plus a tiny conformer and Q-Former. The encoder
    /// concatenates layer 1 (`cat_hidden_layers`), so the encoder output is 2 * 16 = 32 wide, which
    /// is the projector's `encoder_hidden_size`. The projector maps to the text width of 32.
    static let tinyJSON = """
        {
          "model_type": "granite_speech_plus",
          "text_config": {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "vocab_size": 64,
            "logits_scaling": 1.0,
            "attention_multiplier": 0.25,
            "embedding_multiplier": 1.0,
            "residual_multiplier": 1.0,
            "max_position_embeddings": 128,
            "attention_bias": false,
            "mlp_bias": false,
            "tie_word_embeddings": true,
            "rope_parameters": {
              "rope_type": "yarn",
              "factor": 8.0,
              "original_max_position_embeddings": 64,
              "rope_theta": 5000000.0
            }
          },
          "encoder_config": {
            "input_dim": 160,
            "num_layers": 2,
            "hidden_dim": 16,
            "feedforward_mult": 2,
            "num_heads": 2,
            "output_dim": 4,
            "context_size": 4,
            "max_pos_emb": 8,
            "conv_kernel_size": 3,
            "conv_expansion_factor": 2,
            "cat_hidden_layers": [1]
          },
          "projector_config": {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "encoder_hidden_size": 32,
            "layer_norm_eps": 1e-12,
            "cross_attention_frequency": 2
          },
          "downsample_rate": 5,
          "window_size": 15,
          "audio_token_index": 63
        }
        """
}

final class SharpAIGraniteSpeechPlusConfigurationTests: XCTestCase {

    func testDecodesTextConfigAndModelType() throws {
        let data = Data(SharpAIGraniteSpeechPlusFixtures.tinyJSON.utf8)
        let config = try JSONDecoder.json5().decode(
            SharpAIGraniteSpeechPlusConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "granite_speech_plus")
        XCTAssertEqual(config.textConfiguration.hiddenSize, 32)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 2)
        XCTAssertEqual(config.textConfiguration.attentionHeads, 4)
        XCTAssertEqual(config.textConfiguration.kvHeads, 2)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 64)
        XCTAssertTrue(config.textConfiguration.tieWordEmbeddings)
    }

    func testRopeParametersYarnIsKept() throws {
        let data = Data(SharpAIGraniteSpeechPlusFixtures.tinyJSON.utf8)
        let config = try JSONDecoder.json5().decode(
            SharpAIGraniteSpeechPlusConfiguration.self, from: data)
        let text = config.textConfiguration

        XCTAssertEqual(text.ropeTheta, 5_000_000)
        XCTAssertEqual(text.ropeScaling?["rope_type"], StringOrNumber.string("yarn"))
        XCTAssertEqual(text.ropeScaling?["factor"]?.asFloat(), 8.0)
        XCTAssertEqual(text.ropeScaling?["original_max_position_embeddings"]?.asInt(), 64)
        XCTAssertNil(text.ropeScaling?["rope_theta"])
    }

    func testLegacyRopeScalingIsUsedWhenRopeParametersAbsent() throws {
        let json = """
            {
              "model_type": "granite_speech_plus",
              "text_config": {
                "hidden_size": 32, "num_hidden_layers": 2, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-5,
                "vocab_size": 64, "logits_scaling": 1.0, "attention_multiplier": 0.25,
                "embedding_multiplier": 1.0, "residual_multiplier": 1.0,
                "max_position_embeddings": 128, "attention_bias": false, "mlp_bias": false,
                "tie_word_embeddings": true, "rope_theta": 10000.0,
                "rope_scaling": {"type": "linear", "factor": 2.0}
              }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIGraniteSpeechPlusConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.textConfiguration.ropeTheta, 10_000)
        XCTAssertEqual(config.textConfiguration.ropeScaling?["type"], StringOrNumber.string("linear"))
    }

    func testValidationRejectsKvHeadsThatDoNotDivideAttentionHeads() throws {
        let json = SharpAIGraniteSpeechPlusFixtures.tinyJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 2", with: "\"num_key_value_heads\": 3")
        let config = try JSONDecoder.json5().decode(
            SharpAIGraniteSpeechPlusConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testMissingTextConfigFailsDecode() {
        let json = #"{"model_type": "granite_speech_plus"}"#
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIGraniteSpeechPlusConfiguration.self, from: Data(json.utf8)))
    }

    func testEncoderAndProjectorDecodeFromFixture() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIGraniteSpeechPlusConfiguration.self,
            from: Data(SharpAIGraniteSpeechPlusFixtures.tinyJSON.utf8))

        XCTAssertEqual(config.encoderConfiguration.hiddenDim, 16)
        XCTAssertEqual(config.encoderConfiguration.catHiddenLayers, [1])
        XCTAssertEqual(config.encoderConfiguration.outputWidth, 32)
        XCTAssertEqual(config.projectorConfiguration.encoderHiddenSize, 32)
        XCTAssertEqual(config.projectorConfiguration.crossAttentionFrequency, 2)
        XCTAssertEqual(config.windowSize, 15)
        XCTAssertEqual(config.downsampleRate, 5)
        XCTAssertEqual(config.audioTokenIndex, 63)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testProjectorWidthMustMatchConcatenatedEncoderOutput() throws {
        let json = SharpAIGraniteSpeechPlusFixtures.tinyJSON.replacingOccurrences(
            of: "\"encoder_hidden_size\": 32", with: "\"encoder_hidden_size\": 16")
        let config = try JSONDecoder.json5().decode(
            SharpAIGraniteSpeechPlusConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard let typed = error as? SharpAIGraniteSpeechPlusError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }
}
