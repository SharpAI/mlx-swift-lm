// SharpAI fork overlay tests for vibevoice_asr configuration decoding. Inline JSON only, no network.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIVibevoiceAsrConfigurationTests: XCTestCase {

    /// Trimmed copy of the fixture microsoft__VibeVoice-ASR-HF.json (layer_types list removed).
    static let fixtureJSON = """
        {
          "model_type": "vibevoice_asr",
          "architectures": ["VibeVoiceAsrForConditionalGeneration"],
          "audio_token_id": 151648,
          "audio_bos_token_id": 151646,
          "audio_eos_token_id": 151647,
          "acoustic_tokenizer_chunk_size": 1440000,
          "acoustic_tokenizer_encoder_config": {
            "channels": 1,
            "depths": [3, 3, 3, 3, 3, 3, 8],
            "downsampling_ratios": [2, 2, 4, 5, 5, 8],
            "ffn_expansion": 4,
            "hidden_act": "gelu",
            "hidden_size": 64,
            "kernel_size": 7,
            "layer_scale_init_value": 1e-06,
            "model_type": "vibevoice_acoustic_tokenizer_encoder",
            "num_filters": 32,
            "rms_norm_eps": 1e-05,
            "vae_std": 0.625
          },
          "semantic_tokenizer_encoder_config": {
            "channels": 1,
            "depths": [3, 3, 3, 3, 3, 3, 8],
            "downsampling_ratios": [2, 2, 4, 5, 5, 8],
            "ffn_expansion": 4,
            "hidden_act": "gelu",
            "hidden_size": 128,
            "kernel_size": 7,
            "layer_scale_init_value": 1e-06,
            "model_type": "vibevoice_acoustic_tokenizer_encoder",
            "num_filters": 32,
            "rms_norm_eps": 1e-05,
            "vae_std": 0.625
          },
          "text_config": {
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "max_position_embeddings": 131072,
            "model_type": "qwen2",
            "num_attention_heads": 28,
            "num_hidden_layers": 28,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
            "sliding_window": null,
            "tie_word_embeddings": false,
            "use_sliding_window": false,
            "vocab_size": 152064
          },
          "dtype": "bfloat16",
          "transformers_version": "5.3.0.dev0"
        }
        """

    func testFixtureDecodesTopLevelAndNestedFields() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIVibevoiceAsrConfiguration.self, from: Data(Self.fixtureJSON.utf8))

        XCTAssertEqual(config.audioTokenId, 151648)
        XCTAssertEqual(config.audioBosTokenId, 151646)
        XCTAssertEqual(config.audioEosTokenId, 151647)
        XCTAssertEqual(config.acousticTokenizerChunkSize, 1_440_000)

        XCTAssertEqual(config.acousticTokenizer.hiddenSize, 64)
        XCTAssertEqual(config.semanticTokenizer.hiddenSize, 128)
        XCTAssertEqual(config.acousticTokenizer.numFilters, 32)
        XCTAssertEqual(config.acousticTokenizer.vaeStd, 0.625, accuracy: 1e-6)
        XCTAssertEqual(config.acousticTokenizer.hopLength, 3200)

        XCTAssertEqual(config.text.hiddenSize, 3584)
        XCTAssertEqual(config.text.hiddenLayers, 28)
        XCTAssertEqual(config.text.attentionHeads, 28)
        XCTAssertEqual(config.text.kvHeads, 4)
        XCTAssertEqual(config.text.headDim, 128)
        XCTAssertEqual(config.text.ropeTheta, 1_000_000, accuracy: 1)
        XCTAssertFalse(config.text.tieWordEmbeddings)
    }

    func testFixtureValidates() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIVibevoiceAsrConfiguration.self, from: Data(Self.fixtureJSON.utf8))
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingSemanticConfigUsesHiddenSize128() throws {
        let json = #"{"audio_token_id": 3, "text_config": {"vocab_size": 16}}"#
        let config = try JSONDecoder.json5().decode(
            SharpAIVibevoiceAsrConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(config.semanticTokenizer.hiddenSize, 128)
        XCTAssertEqual(config.acousticTokenizer.hiddenSize, 64)
        XCTAssertEqual(config.text.hiddenSize, 3584)
    }

    func testMismatchedDownsamplingRatiosAreRejected() throws {
        let json = """
            {
              "semantic_tokenizer_encoder_config": {
                "downsampling_ratios": [2, 2, 4, 5, 5, 4],
                "depths": [3, 3, 3, 3, 3, 3, 8]
              }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIVibevoiceAsrConfiguration.self, from: Data(json.utf8))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testDepthsMustHaveOneMoreEntryThanRatios() throws {
        let json = """
            {
              "acoustic_tokenizer_encoder_config": {
                "downsampling_ratios": [2, 2],
                "depths": [1, 1]
              },
              "semantic_tokenizer_encoder_config": {
                "downsampling_ratios": [2, 2],
                "depths": [1, 1]
              }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIVibevoiceAsrConfiguration.self, from: Data(json.utf8))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testGroupedQueryHeadsMustDivide() throws {
        let json = #"{"text_config": {"num_attention_heads": 4, "num_key_value_heads": 3, "hidden_size": 8}}"#
        let config = try JSONDecoder.json5().decode(
            SharpAIVibevoiceAsrConfiguration.self, from: Data(json.utf8))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testHopLengthIsProductOfRatios() {
        let tokenizer = SharpAIVibevoiceAsrTokenizerConfiguration()
        XCTAssertEqual(tokenizer.hopLength, 2 * 2 * 4 * 5 * 5 * 8)
    }
}
