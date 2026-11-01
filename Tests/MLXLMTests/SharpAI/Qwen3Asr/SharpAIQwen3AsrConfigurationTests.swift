// SharpAI fork overlay tests for qwen3_asr configuration decoding. Not part of upstream.
// No network and no weights. The JSON is inline and mirrors the two hub config layouts.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIQwen3AsrConfigurationTests: XCTestCase {

    /// Flat transformers layout with tiny sizes. Used by the sanitize and forward tests too.
    static let tinyJSON = """
        {
          "model_type": "qwen3_asr",
          "audio_token_id": 9,
          "audio_start_token_id": 7,
          "audio_end_token_id": 8,
          "timestamp_token_id": 10,
          "tie_word_embeddings": true,
          "audio_config": {
            "num_mel_bins": 128,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "d_model": 16,
            "n_window": 50,
            "n_window_infer": 800,
            "downsample_hidden_size": 8,
            "output_dim": 32,
            "max_source_positions": 64
          },
          "text_config": {
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000,
            "max_position_embeddings": 128,
            "vocab_size": 64
          }
        }
        """

    /// Hub layout of Qwen/Qwen3-ASR-0.6B, trimmed to the keys this decoder reads.
    static let asrHubJSON = """
        {
          "architectures": ["Qwen3ASRForConditionalGeneration"],
          "model_type": "qwen3_asr",
          "thinker_config": {
            "model_type": "qwen3_asr",
            "audio_config": {
              "model_type": "qwen3_asr_audio_encoder",
              "d_model": 896,
              "encoder_attention_heads": 14,
              "encoder_ffn_dim": 3584,
              "encoder_layers": 18,
              "downsample_hidden_size": 480,
              "n_window": 50,
              "n_window_infer": 800,
              "num_mel_bins": 128,
              "output_dim": 1024,
              "max_source_positions": 1500,
              "activation_function": "gelu"
            },
            "text_config": {
              "model_type": "qwen3",
              "hidden_size": 1024,
              "intermediate_size": 3072,
              "num_hidden_layers": 28,
              "num_attention_heads": 16,
              "num_key_value_heads": 8,
              "head_dim": 128,
              "rms_norm_eps": 1e-06,
              "rope_theta": 1000000,
              "max_position_embeddings": 65536,
              "vocab_size": 151936,
              "tie_word_embeddings": true
            },
            "audio_token_id": 151676,
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "tie_word_embeddings": true
          }
        }
        """

    /// Hub layout of Qwen/Qwen3-ForcedAligner-0.6B, trimmed. Timestamp keys sit at the top level.
    static let alignerHubJSON = """
        {
          "architectures": ["Qwen3ASRForConditionalGeneration"],
          "timestamp_token_id": 151705,
          "timestamp_segment_time": 80,
          "model_type": "qwen3_asr",
          "thinker_config": {
            "model_type": "qwen3_forced_aligner",
            "classify_num": 5000,
            "audio_config": {
              "d_model": 1024,
              "encoder_attention_heads": 16,
              "encoder_ffn_dim": 4096,
              "encoder_layers": 24,
              "downsample_hidden_size": 480,
              "n_window": 50,
              "n_window_infer": 800,
              "num_mel_bins": 128,
              "output_dim": 1024,
              "max_source_positions": 1500
            },
            "text_config": {
              "model_type": "qwen3",
              "hidden_size": 1024,
              "intermediate_size": 3072,
              "num_hidden_layers": 28,
              "num_attention_heads": 16,
              "num_key_value_heads": 8,
              "head_dim": 128,
              "rms_norm_eps": 1e-06,
              "rope_theta": 1000000,
              "max_position_embeddings": 8192,
              "vocab_size": 152064,
              "tie_word_embeddings": false
            },
            "audio_token_id": 151676,
            "tie_word_embeddings": false
          }
        }
        """

    func testHubAsrLayoutDecodesThinkerConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIQwen3AsrConfiguration.self, from: Data(Self.asrHubJSON.utf8))

        XCTAssertFalse(config.forcedAligner)
        XCTAssertEqual(config.audioTokenId, 151676)
        XCTAssertEqual(config.audioConfiguration.dModel, 896)
        XCTAssertEqual(config.audioConfiguration.outputDim, 1024)
        XCTAssertEqual(config.textConfiguration.hiddenSize, 1024)
        XCTAssertEqual(config.textConfiguration.kvHeads, 8)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testHubAlignerLayoutSelectsForcedAligner() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIQwen3AsrConfiguration.self, from: Data(Self.alignerHubJSON.utf8))

        XCTAssertTrue(config.forcedAligner)
        XCTAssertEqual(config.classifyNum, 5000)
        XCTAssertEqual(config.timestampTokenId, 151705)
        XCTAssertEqual(config.timestampSegmentTime, 80)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 152064)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testFlatLayoutDecodes() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIQwen3AsrConfiguration.self, from: Data(Self.tinyJSON.utf8))

        XCTAssertFalse(config.forcedAligner)
        XCTAssertEqual(config.audioStartTokenId, 7)
        XCTAssertEqual(config.audioEndTokenId, 8)
        XCTAssertEqual(config.audioConfiguration.nWindowInfer, 800)
        XCTAssertEqual(config.textConfiguration.headDim, 8)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testOutputDimMustMatchTextHiddenSize() throws {
        let mismatched = Self.tinyJSON.replacingOccurrences(
            of: "\"output_dim\": 32", with: "\"output_dim\": 16")
        let config = try JSONDecoder.json5().decode(
            SharpAIQwen3AsrConfiguration.self, from: Data(mismatched.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testWindowMustBeMultipleOfChunk() throws {
        let badWindow = Self.tinyJSON.replacingOccurrences(
            of: "\"n_window_infer\": 800", with: "\"n_window_infer\": 150")
        let config = try JSONDecoder.json5().decode(
            SharpAIQwen3AsrConfiguration.self, from: Data(badWindow.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRegistryKeysModelTypeQwen3Asr() {
        XCTAssertNotNil(SharpAIQwen3AsrOverlay.creators["qwen3_asr"])
    }
}
