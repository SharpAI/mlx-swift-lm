// SharpAI fork overlay tests for moss_audio configuration. Not part of upstream.
// UNVERIFIED: written without compiling. The audio encoder keys are not covered.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest
@testable import MLXLLM

final class SharpAIMossAudioConfigurationTests: XCTestCase {

    private let tinyJSON = """
        {
          "architectures": ["MossAudioModel"],
          "model_type": "moss_audio",
          "language_config": {
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 100,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rope_theta": 1000000,
            "tie_word_embeddings": true
          }
        }
        """

    func testDecodesQwen3LanguageConfigFromInlineJSON() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMossAudioConfiguration.self, from: Data(tinyJSON.utf8))

        XCTAssertEqual(config.modelType, "moss_audio")
        XCTAssertEqual(config.architectures, ["MossAudioModel"])
        XCTAssertEqual(config.languageConfiguration.hiddenSize, 64)
        XCTAssertEqual(config.languageConfiguration.hiddenLayers, 2)
        XCTAssertEqual(config.languageConfiguration.attentionHeads, 4)
        XCTAssertEqual(config.languageConfiguration.kvHeads, 2)
        XCTAssertEqual(config.languageConfiguration.headDim, 16)
        XCTAssertEqual(config.languageConfiguration.vocabularySize, 100)
        XCTAssertTrue(config.languageConfiguration.tieWordEmbeddings)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingLanguageConfigFailsDecoding() {
        let json = #"{"model_type": "moss_audio", "architectures": ["MossAudioModel"]}"#

        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIMossAudioConfiguration.self, from: Data(json.utf8)))
    }

    func testRejectsOtherModelType() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "\"model_type\": \"moss_audio\"", with: "\"model_type\": \"qwen2\"")
        let config = try JSONDecoder.json5().decode(
            SharpAIMossAudioConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    /// UNVERIFIED: the real architecture string is unknown, so any value is accepted.
    func testAcceptsAnyArchitectureString() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "MossAudioModel", with: "MossAudioForConditionalGeneration")
        let config = try JSONDecoder.json5().decode(
            SharpAIMossAudioConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.architectures, ["MossAudioForConditionalGeneration"])
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    /// With no `audio_config`, the encoder takes the 4B-Instruct defaults, and DeepStack uses
    /// layers 8, 16 and 24 (vLLM's default `deepstack_num_inject_layers` is unset).
    func testAudioDefaultsAndDeepstackCountWithoutAudioConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMossAudioConfiguration.self, from: Data(tinyJSON.utf8))

        XCTAssertEqual(config.audioConfiguration.dModel, 1280)
        XCTAssertEqual(config.audioConfiguration.nWindow, 200)
        XCTAssertEqual(config.audioConfiguration.deepstackEncoderLayerIndexes, [8, 16, 24])
        XCTAssertEqual(config.adapterHiddenSize, 8192)
        XCTAssertEqual(config.deepstackMergerCount, 3)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 100)
    }
}
