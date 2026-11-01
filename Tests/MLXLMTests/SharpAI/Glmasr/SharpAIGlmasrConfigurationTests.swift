// SharpAI fork overlay tests for glmasr configuration. Not part of upstream.

import Foundation
import MLXLMCommon
@testable import MLXVLM
import XCTest

/// Inline configs for glmasr tests. No weights are read.
enum SharpAIGlmasrTestConfig {

    /// Values from transformers' GlmAsrConfig defaults (zai-org/GLM-ASR-Nano-2512 shape).
    static let nanoShape = """
        {
          "model_type": "glmasr",
          "audio_token_id": 59260,
          "projector_hidden_act": "gelu",
          "audio_config": {
            "model_type": "glmasr_encoder",
            "hidden_size": 1280,
            "intermediate_size": 5120,
            "num_hidden_layers": 32,
            "num_attention_heads": 20,
            "num_mel_bins": 128,
            "max_position_embeddings": 1500,
            "partial_rotary_factor": 0.5,
            "hidden_act": "gelu"
          },
          "text_config": {
            "model_type": "llama",
            "vocab_size": 59264,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
          }
        }
        """

    /// A 1-layer model with 4 mel bins. Encoder: 8 hidden, 2 heads of width 4, RoPE on 2 channels.
    /// 15 frames give 8 conv frames, which is 2 audio tokens.
    static func tiny(intermediate: Int = 32, textAct: String = "llama") -> String {
        """
        {
          "model_type": "glmasr",
          "audio_token_id": 30,
          "projector_hidden_act": "gelu",
          "audio_config": {
            "model_type": "glmasr_encoder",
            "hidden_size": 8,
            "intermediate_size": \(intermediate),
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_mel_bins": 4,
            "max_position_embeddings": 16,
            "partial_rotary_factor": 0.5
          },
          "text_config": {
            "model_type": "\(textAct)",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
            "rms_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
          }
        }
        """
    }

    static func decode(_ json: String) throws -> SharpAIGlmasrConfiguration {
        try JSONDecoder.json5().decode(SharpAIGlmasrConfiguration.self, from: Data(json.utf8))
    }
}

final class SharpAIGlmasrConfigurationTests: XCTestCase {

    func testNanoShapeDecodesToUpstreamDefaults() throws {
        let config = try SharpAIGlmasrTestConfig.decode(SharpAIGlmasrTestConfig.nanoShape)

        XCTAssertEqual(config.modelType, "glmasr")
        XCTAssertEqual(config.audioTokenId, 59260)
        XCTAssertEqual(config.audioConfiguration.hiddenSize, 1280)
        XCTAssertEqual(config.audioConfiguration.intermediateSize, 5120)
        XCTAssertEqual(config.audioConfiguration.hiddenLayers, 32)
        XCTAssertEqual(config.audioConfiguration.attentionHeads, 20)
        XCTAssertEqual(config.audioConfiguration.kvHeads, 20)
        XCTAssertEqual(config.audioConfiguration.numMelBins, 128)
        XCTAssertEqual(config.audioConfiguration.partialRotaryFactor, 0.5)
        XCTAssertEqual(config.audioConfiguration.ropeTheta, 10_000)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 59264)
        XCTAssertEqual(config.textConfiguration.hiddenSize, 2048)
        XCTAssertEqual(config.textConfiguration.headDim, 128)
        XCTAssertEqual(config.textConfiguration.kvHeads, 4)
        XCTAssertFalse(config.textConfiguration.tieWordEmbeddings)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testEmptyConfigUsesAllDefaults() throws {
        let config = try SharpAIGlmasrTestConfig.decode("{}")

        XCTAssertEqual(config.audioConfiguration.hiddenSize, 1280)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 28)
        XCTAssertEqual(config.audioTokenId, 59260)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testTinyConfigDecodesAndValidates() throws {
        let config = try SharpAIGlmasrTestConfig.decode(SharpAIGlmasrTestConfig.tiny())

        XCTAssertEqual(config.audioTokenId, 30)
        XCTAssertEqual(config.audioConfiguration.numMelBins, 4)
        XCTAssertEqual(config.audioConfiguration.intermediateSize, 32)
        XCTAssertEqual(config.textConfiguration.headDim, 4)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRejectsIntermediateNotFourTimesHidden() throws {
        let config = try SharpAIGlmasrTestConfig.decode(
            SharpAIGlmasrTestConfig.tiny(intermediate: 16))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsNonLlamaTextDecoder() throws {
        let config = try SharpAIGlmasrTestConfig.decode(
            SharpAIGlmasrTestConfig.tiny(textAct: "qwen2"))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsNonGeluAudioActivation() throws {
        let json = SharpAIGlmasrTestConfig.tiny().replacingOccurrences(
            of: "\"partial_rotary_factor\": 0.5",
            with: "\"partial_rotary_factor\": 0.5, \"hidden_act\": \"silu\"")
        let config = try SharpAIGlmasrTestConfig.decode(json)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
