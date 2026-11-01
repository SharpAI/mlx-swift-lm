// SharpAI fork overlay tests for moss_transcribe_diarize configuration decoding.

import Foundation
@testable import MLXVLM
import XCTest

final class SharpAIMossTranscribeDiarizeConfigurationTests: XCTestCase {

    func testDecodesTinyConfig() throws {
        let config = try SharpAIMossTranscribeDiarizeTestConfig.decode(
            SharpAIMossTranscribeDiarizeTestConfig.tiny())

        XCTAssertEqual(config.modelType, "moss_transcribe_diarize")
        XCTAssertEqual(config.audioTokenId, 30)
        XCTAssertEqual(config.audioConfiguration.hiddenSize, 8)
        XCTAssertEqual(config.audioConfiguration.hiddenLayers, 1)
        XCTAssertEqual(config.audioConfiguration.numMelBins, 4)
        XCTAssertEqual(config.audioConfiguration.maxSourcePositions, 32)
        XCTAssertEqual(config.audioConfiguration.downsampleFactor, 2)
        XCTAssertEqual(config.textConfiguration.headDim, 4)
        XCTAssertEqual(config.textConfiguration.kvHeads, 1)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 32)
        XCTAssertTrue(config.textConfiguration.tieWordEmbeddings)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMinimalConfigUsesDefaults() throws {
        let json = """
            {
              "audio_config": {
                "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2
              },
              "text_config": {
                "hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 16,
                "num_attention_heads": 2, "vocab_size": 32
              }
            }
            """
        let config = try SharpAIMossTranscribeDiarizeTestConfig.decode(json)

        XCTAssertEqual(config.modelType, "moss_transcribe_diarize")
        XCTAssertEqual(config.audioTokenId, 151_669)
        XCTAssertEqual(config.audioConfiguration.numMelBins, 128)
        XCTAssertEqual(config.audioConfiguration.maxSourcePositions, 1500)
        XCTAssertEqual(config.audioConfiguration.activationFunction, "gelu")
        XCTAssertEqual(config.audioConfiguration.downsampleFactor, 1)
        XCTAssertEqual(config.textConfiguration.kvHeads, 2)
        XCTAssertEqual(config.textConfiguration.headDim, 4)
        XCTAssertEqual(config.textConfiguration.ropeTheta, 1_000_000)
        XCTAssertFalse(config.textConfiguration.tieWordEmbeddings)
    }

    func testRopeThetaReadsRopeParameters() throws {
        let json = """
            {
              "audio_config": {
                "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2
              },
              "text_config": {
                "hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 16,
                "num_attention_heads": 2, "vocab_size": 32,
                "rope_parameters": { "rope_theta": 5000 }
              }
            }
            """
        let config = try SharpAIMossTranscribeDiarizeTestConfig.decode(json)

        XCTAssertEqual(config.textConfiguration.ropeTheta, 5000)
    }

    func testRejectsNonGeluActivation() throws {
        let config = try SharpAIMossTranscribeDiarizeTestConfig.decode(
            SharpAIMossTranscribeDiarizeTestConfig.tiny(activation: "relu"))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsHeadsThatDoNotDivideHidden() throws {
        let json = """
            {
              "audio_config": {
                "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 3
              },
              "text_config": {
                "hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 16,
                "num_attention_heads": 2, "vocab_size": 32
              }
            }
            """
        let config = try SharpAIMossTranscribeDiarizeTestConfig.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsKVHeadsThatDoNotDivideHeads() throws {
        let json = """
            {
              "audio_config": {
                "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2
              },
              "text_config": {
                "hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 16,
                "num_attention_heads": 2, "num_key_value_heads": 3, "vocab_size": 32
              }
            }
            """
        let config = try SharpAIMossTranscribeDiarizeTestConfig.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
