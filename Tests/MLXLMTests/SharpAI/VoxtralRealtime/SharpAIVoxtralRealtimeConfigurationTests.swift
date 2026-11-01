// SharpAI fork overlay tests for voxtral_realtime configuration decoding. No network.

import Foundation
import XCTest

@testable import MLXVLM

final class SharpAIVoxtralRealtimeConfigurationTests: XCTestCase {

    /// Copy of scripts/conformance/offline/fixtures/voxtral_realtime config.
    private let fixtureJSON = """
        {
          "architectures": ["VoxtralRealtimeForConditionalGeneration"],
          "audio_config": {
            "activation_function": "gelu",
            "attention_dropout": 0.0,
            "head_dim": 64,
            "hidden_act": "silu",
            "hidden_size": 1280,
            "initializer_range": 0.02,
            "intermediate_size": 5120,
            "max_position_embeddings": 1500,
            "model_type": "voxtral_realtime_encoder",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "num_mel_bins": 128,
            "rms_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
            "sliding_window": 750,
            "vocab_size": 131072
          },
          "audio_length_per_tok": 8,
          "default_num_delay_tokens": 6,
          "downsample_factor": 4,
          "dtype": "bfloat16",
          "hidden_size": 3072,
          "model_type": "voxtral_realtime",
          "projector_hidden_act": "gelu",
          "text_config": {
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 3072,
            "initializer_range": 0.02,
            "intermediate_size": 9216,
            "max_position_embeddings": 131072,
            "model_type": "voxtral_realtime_text",
            "num_attention_heads": 32,
            "num_hidden_layers": 26,
            "num_key_value_heads": 8,
            "pad_token_id": null,
            "rms_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
            "sliding_window": 8192,
            "tie_word_embeddings": true,
            "use_cache": true,
            "vocab_size": 131072
          },
          "transformers_version": "5.2.0.dev0"
        }
        """

    func testFixtureDecodesAudioAndTextSections() throws {
        let config = try JSONDecoder().decode(
            SharpAIVoxtralRealtimeConfiguration.self, from: Data(fixtureJSON.utf8))

        XCTAssertEqual(config.modelType, "voxtral_realtime")
        XCTAssertEqual(config.audioConfiguration.hiddenSize, 1280)
        XCTAssertEqual(config.audioConfiguration.hiddenLayers, 32)
        XCTAssertEqual(config.audioConfiguration.numMelBins, 128)
        XCTAssertEqual(config.audioConfiguration.slidingWindow, 750)
        XCTAssertEqual(config.audioConfiguration.ropeTheta, 1_000_000)
        XCTAssertEqual(config.textConfiguration.hiddenSize, 3072)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 26)
        XCTAssertEqual(config.textConfiguration.kvHeads, 8)
        XCTAssertEqual(config.textConfiguration.headDim, 128)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 131072)
        XCTAssertTrue(config.textConfiguration.tieWordEmbeddings)
        XCTAssertEqual(config.hiddenSize, 3072)
        XCTAssertEqual(config.downsampleFactor, 4)
        XCTAssertEqual(config.defaultNumDelayTokens, 6)
        XCTAssertEqual(config.projectorHiddenAct, "gelu")
    }

    func testFixtureUsesDefaultAdaNormDimension() throws {
        let config = try JSONDecoder().decode(
            SharpAIVoxtralRealtimeConfiguration.self, from: Data(fixtureJSON.utf8))
        // Not in the fixture. UNVERIFIED default.
        XCTAssertEqual(config.adaNormDimension, 32)
    }

    func testFixturePassesValidation() throws {
        let config = try JSONDecoder().decode(
            SharpAIVoxtralRealtimeConfiguration.self, from: Data(fixtureJSON.utf8))
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testTinyConfigDecodesAndValidates() throws {
        let config = try SharpAIVoxtralRealtimeTestConfig.decodeTiny()
        XCTAssertEqual(config.audioConfiguration.numMelBins, 4)
        XCTAssertEqual(config.textConfiguration.kvHeads, 1)
        XCTAssertEqual(config.adaNormDimension, 4)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testHeadCountMustBeMultipleOfKvHeads() throws {
        // 2 heads with 3 kv heads does not divide.
        let config = try SharpAIVoxtralRealtimeTestConfig.decodeTiny(numKvHeads: 3)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testHiddenSizeMustMatchTextHiddenSize() throws {
        let config = try SharpAIVoxtralRealtimeTestConfig.decodeTiny(topHiddenSize: 16)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
