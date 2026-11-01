// SharpAI fork overlay tests for nemo_speechlm config decoding. Not part of upstream.
// The JSON in testDecodesRealCheckpointBlocks copies values from the real config
// (https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/resolve/main/config.json).
// The transcribeJSON fixture keeps the older audio_config/text_config shape for the existing tests.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAINemoSpeechlmConfigurationTests: XCTestCase {

    private let transcribeJSON = """
        {
            "model_type": "nemo_speechlm",
            "architectures": ["Nemotron3_5TranscribeForConditionalGeneration"],
            "audio_config": {
                "model_type": "nemo_audio_encoder"
            },
            "text_config": {
                "model_type": "nemotron_h"
            }
        }
        """

    func testDecodesTopLevelFields() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAINemoSpeechlmConfiguration.self, from: Data(transcribeJSON.utf8))

        XCTAssertEqual(config.modelType, "nemo_speechlm")
        XCTAssertEqual(
            config.architectures, ["Nemotron3_5TranscribeForConditionalGeneration"])
        XCTAssertTrue(config.hasAudioConfig)
        XCTAssertTrue(config.hasTextConfig)
    }

    func testMissingModelTypeDefaultsToNemoSpeechlm() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAINemoSpeechlmConfiguration.self, from: Data("{}".utf8))

        XCTAssertEqual(config.modelType, "nemo_speechlm")
        XCTAssertEqual(config.architectures, [])
        XCTAssertFalse(config.hasAudioConfig)
        XCTAssertFalse(config.hasTextConfig)
    }

    func testKnownArchitecturePassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAINemoSpeechlmConfiguration.self, from: Data(transcribeJSON.utf8))
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testUnknownArchitectureFailsValidation() throws {
        let badJSON = transcribeJSON.replacingOccurrences(
            of: "Nemotron3_5TranscribeForConditionalGeneration", with: "SomeOtherArchitecture")
        let config = try JSONDecoder.json5().decode(
            SharpAINemoSpeechlmConfiguration.self, from: Data(badJSON.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    /// Top-level blocks and the `perception.encoder` values as they appear in the real checkpoint config.
    func testDecodesRealCheckpointBlocks() throws {
        let realJSON = """
            {
                "model_type": "nemo_speechlm",
                "architectures": ["Nemotron3_5TranscribeForConditionalGeneration"],
                "perception": {
                    "output_dim": 2688,
                    "encoder": {
                        "feat_in": 128,
                        "d_model": 1280,
                        "n_heads": 10,
                        "n_layers": 32,
                        "drop_rate": 0.1,
                        "qkv_bias": false,
                        "qk_norm": true,
                        "ff_expansion": 4,
                        "pre_block_norm": true,
                        "subsampling_factor": 8,
                        "attn_mode": "full",
                        "self_attention_model": "rope",
                        "rope_base": 10000.0,
                        "rotary_fraction": 0.5
                    }
                },
                "llm_config": {
                    "model_type": "nemotron_h"
                }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAINemoSpeechlmConfiguration.self, from: Data(realJSON.utf8))
        XCTAssertTrue(config.hasPerception)
        XCTAssertTrue(config.hasLLMConfig)
        XCTAssertFalse(config.hasAudioConfig)
        XCTAssertFalse(config.hasTextConfig)

        // Decode the `perception.encoder` block on its own, as the audio encoder does.
        let encoderJSON = try XCTUnwrap(
            (try JSONSerialization.jsonObject(with: Data(realJSON.utf8)) as? [String: Any])?[
                "perception"] as? [String: Any])
        let encoderData = try JSONSerialization.data(
            withJSONObject: try XCTUnwrap(encoderJSON["encoder"]))
        let encoder = try JSONDecoder.json5().decode(
            SharpAINemoSpeechlmEncoderConfiguration.self, from: encoderData)
        XCTAssertEqual(encoder.featIn, 128)
        XCTAssertEqual(encoder.dModel, 1280)
        XCTAssertEqual(encoder.nHeads, 10)
        XCTAssertEqual(encoder.nLayers, 32)
        XCTAssertEqual(encoder.subsamplingFactor, 8)
        XCTAssertEqual(encoder.rotaryFraction, 0.5)
        XCTAssertNoThrow(try encoder.validate())
    }
}
