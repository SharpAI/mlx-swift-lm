// SharpAI fork overlay tests for funaudiochat configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIFunaudiochatConfigurationTests: XCTestCase {

    func testDecodesModelTypeFromInlineJSON() throws {
        let json = """
            {
              "architectures": ["FunAudioChatForConditionalGeneration"],
              "model_type": "funaudiochat"
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIFunaudiochatConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "funaudiochat")
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRejectsOtherModelType() throws {
        let json = #"{"model_type": "qwen2"}"#
        let config = try JSONDecoder.json5().decode(
            SharpAIFunaudiochatConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    /// Key names and values taken from https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B/blob/main/config.json
    func testDecodesNestedAudioAndTextConfigFromRealKeys() throws {
        let json = """
            {
              "model_type": "funaudiochat",
              "audio_token_index": 151669,
              "audio_config": {
                "model_type": "funaudiochat_audio_encoder",
                "encoder_layers": 32,
                "d_model": 1280,
                "encoder_attention_heads": 20,
                "encoder_ffn_dim": 5120,
                "num_mel_bins": 128,
                "output_dim": 4096,
                "group_size": 5,
                "n_window": 100,
                "max_source_positions": 1500,
                "codebook_size": 6565,
                "enable_audio_invert_tower": true,
                "continuous_features_mode": "replace"
              },
              "text_config": {
                "model_type": "qwen3",
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-06,
                "rope_theta": 5000000,
                "hidden_act": "silu",
                "attention_bias": false,
                "audio_bos_index": 151670,
                "audio_eos_index": 151671,
                "sil_index": 151673
              }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIFunaudiochatConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.audioTokenIndex, 151669)

        let audio = try XCTUnwrap(config.audioConfig)
        XCTAssertEqual(audio.encoderLayers, 32)
        XCTAssertEqual(audio.dModel, 1280)
        XCTAssertEqual(audio.encoderAttentionHeads, 20)
        XCTAssertEqual(audio.encoderFfnDim, 5120)
        XCTAssertEqual(audio.numMelBins, 128)
        XCTAssertEqual(audio.outputDim, 4096)
        XCTAssertEqual(audio.groupSize, 5)
        XCTAssertEqual(audio.nWindow, 100)
        XCTAssertEqual(audio.maxSourcePositions, 1500)
        XCTAssertEqual(audio.codebookSize, 6565)
        XCTAssertEqual(audio.continuousFeaturesMode, "replace")

        let text = try XCTUnwrap(config.textConfig)
        XCTAssertEqual(text.hiddenSize, 4096)
        XCTAssertEqual(text.intermediateSize, 12288)
        XCTAssertEqual(text.numHiddenLayers, 36)
        XCTAssertEqual(text.numAttentionHeads, 32)
        XCTAssertEqual(text.numKeyValueHeads, 8)
        XCTAssertEqual(text.headDim, 128)
        XCTAssertEqual(text.vocabSize, 151936)
        XCTAssertEqual(text.maxPositionEmbeddings, 262144)
        XCTAssertEqual(text.rmsNormEps, 1e-6, accuracy: 1e-12)
        XCTAssertEqual(text.ropeTheta, 5_000_000, accuracy: 1e-6)
        XCTAssertEqual(text.hiddenAct, "silu")
        XCTAssertFalse(text.attentionBias)
        XCTAssertEqual(text.audioBosIndex, 151670)
        XCTAssertEqual(text.audioEosIndex, 151671)
        XCTAssertEqual(text.silIndex, 151673)
    }
}
