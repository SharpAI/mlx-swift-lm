// SharpAI fork overlay tests for audioflamingo3 configuration. Not part of upstream.

import Foundation
import MLXLMCommon
@testable import MLXVLM
import XCTest

/// Inline configs for audioflamingo3 tests. No weights are read.
enum SharpAIAudioflamingo3TestConfig {

    /// Values from nvidia/audio-flamingo-3-hf config.json (fixtures/audioflamingo3).
    /// `layer_types` is omitted.
    static let fixture = """
        {
          "architectures": ["AudioFlamingo3ForConditionalGeneration"],
          "audio_config": {
            "activation_dropout": 0.0,
            "activation_function": "gelu",
            "attention_dropout": 0.0,
            "dropout": 0.0,
            "hidden_size": 1280,
            "initializer_range": 0.02,
            "intermediate_size": 5120,
            "layerdrop": 0.0,
            "max_source_positions": 1500,
            "model_type": "audioflamingo3_encoder",
            "num_attention_heads": 20,
            "num_hidden_layers": 32,
            "num_mel_bins": 128,
            "scale_embedding": false
          },
          "audio_token_id": 151669,
          "dtype": "float32",
          "model_type": "audioflamingo3",
          "projector_bias": true,
          "projector_hidden_act": "gelu",
          "text_config": {
            "attention_dropout": 0.0,
            "bos_token_id": 151670,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 3584,
            "initializer_range": 0.02,
            "intermediate_size": 18944,
            "max_position_embeddings": 32768,
            "max_window_layers": 28,
            "model_type": "qwen2",
            "num_attention_heads": 28,
            "num_hidden_layers": 28,
            "num_key_value_heads": 4,
            "pad_token_id": 151671,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
              "rope_theta": 1000000.0,
              "rope_type": "default"
            },
            "sliding_window": null,
            "use_cache": false,
            "use_sliding_window": false,
            "vocab_size": 151672
          },
          "transformers_version": "5.0.0rc1"
        }
        """

    /// A 1-layer model with 4 mel bins and 16 frames. 16 frames give 4 audio tokens.
    static func tiny(tied: Bool = false, scaleEmbedding: Bool = false) -> String {
        """
        {
          "model_type": "audioflamingo3",
          "audio_token_id": 30,
          "projector_bias": true,
          "projector_hidden_act": "gelu",
          "audio_config": {
            "model_type": "audioflamingo3_encoder",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_mel_bins": 4,
            "max_source_positions": 16,
            "scale_embedding": \(scaleEmbedding)
          },
          "text_config": {
            "model_type": "qwen2",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "rms_norm_eps": 1e-6,
            "rope_parameters": { "rope_theta": 10000.0, "rope_type": "default" },
            "tie_word_embeddings": \(tied)
          }
        }
        """
    }

    static func decode(_ json: String) throws -> SharpAIAudioflamingo3Configuration {
        try JSONDecoder.json5().decode(
            SharpAIAudioflamingo3Configuration.self, from: Data(json.utf8))
    }
}

final class SharpAIAudioflamingo3ConfigurationTests: XCTestCase {

    func testDecodesFixtureDimensions() throws {
        let config = try SharpAIAudioflamingo3TestConfig.decode(
            SharpAIAudioflamingo3TestConfig.fixture)

        XCTAssertEqual(config.modelType, "audioflamingo3")
        XCTAssertEqual(config.audioTokenId, 151_669)
        XCTAssertTrue(config.projectorBias)
        XCTAssertEqual(config.projectorHiddenAct, "gelu")

        let audio = config.audioConfiguration
        XCTAssertEqual(audio.hiddenSize, 1280)
        XCTAssertEqual(audio.intermediateSize, 5120)
        XCTAssertEqual(audio.hiddenLayers, 32)
        XCTAssertEqual(audio.attentionHeads, 20)
        XCTAssertEqual(audio.numMelBins, 128)
        XCTAssertEqual(audio.maxSourcePositions, 1500)
        XCTAssertFalse(audio.scaleEmbedding)

        let text = config.textConfiguration
        XCTAssertEqual(text.hiddenSize, 3584)
        XCTAssertEqual(text.hiddenLayers, 28)
        XCTAssertEqual(text.intermediateSize, 18944)
        XCTAssertEqual(text.attentionHeads, 28)
        XCTAssertEqual(text.kvHeads, 4)
        XCTAssertEqual(text.vocabularySize, 151_672)
        XCTAssertEqual(text.ropeTheta, 1_000_000)
        XCTAssertFalse(text.tieWordEmbeddings)
    }

    func testFixtureConfigurationValidates() throws {
        let config = try SharpAIAudioflamingo3TestConfig.decode(
            SharpAIAudioflamingo3TestConfig.fixture)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRopeThetaReadsRopeParameters() throws {
        let json = SharpAIAudioflamingo3TestConfig.tiny()
            .replacingOccurrences(of: "10000.0", with: "50000.0")
        let config = try SharpAIAudioflamingo3TestConfig.decode(json)
        XCTAssertEqual(config.textConfiguration.ropeTheta, 50_000)
    }

    func testRopeThetaReadsTopLevelKeyWhenPresent() throws {
        let json = SharpAIAudioflamingo3TestConfig.tiny()
            .replacingOccurrences(
                of: "\"rope_parameters\": { \"rope_theta\": 10000.0, \"rope_type\": \"default\" },",
                with: "\"rope_theta\": 20000.0,")
        let config = try SharpAIAudioflamingo3TestConfig.decode(json)
        XCTAssertEqual(config.textConfiguration.ropeTheta, 20_000)
    }

    func testRejectsScaledAudioEmbedding() throws {
        let config = try SharpAIAudioflamingo3TestConfig.decode(
            SharpAIAudioflamingo3TestConfig.tiny(scaleEmbedding: true))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsHeadCountThatDoesNotDivideHidden() throws {
        // Both towers get 3 heads for hidden size 8, so both fail validation.
        let json = SharpAIAudioflamingo3TestConfig.tiny()
            .replacingOccurrences(
                of: "\"num_attention_heads\": 2,", with: "\"num_attention_heads\": 3,")
        let config = try SharpAIAudioflamingo3TestConfig.decode(json)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testPooledLengthMatchesConvAndPoolStrides() {
        // 3000 mel frames (30 s) -> 1500 after conv -> 750 tokens after pooling.
        XCTAssertEqual(sharpAIAudioflamingo3PooledLength(3000), 750)
        // 16 frames -> 8 -> 4.
        XCTAssertEqual(sharpAIAudioflamingo3PooledLength(16), 4)
        XCTAssertEqual(sharpAIAudioflamingo3PooledLength(2), 0)
        XCTAssertEqual(sharpAIAudioflamingo3PooledLength(0), 0)
    }
}
