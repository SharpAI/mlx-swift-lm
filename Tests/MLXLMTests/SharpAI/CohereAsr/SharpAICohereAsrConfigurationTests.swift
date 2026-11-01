// SharpAI fork overlay tests for the cohere_asr configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAICohereAsrConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAICohereAsrConfiguration {
        try JSONDecoder.json5().decode(
            SharpAICohereAsrConfiguration.self, from: Data(json.utf8))
    }

    func testEmptyConfigUsesTransformersDefaults() throws {
        let config = try decode("{}")

        XCTAssertEqual(config.modelType, "cohere_asr")
        XCTAssertEqual(config.vocabularySize, 16384)
        XCTAssertEqual(config.hiddenSize, 1024)
        XCTAssertEqual(config.hiddenLayers, 8)
        XCTAssertEqual(config.attentionHeads, 8)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.hiddenAct, "relu")
        XCTAssertEqual(config.encoderConfiguration.hiddenSize, 1280)
        XCTAssertEqual(config.encoderConfiguration.hiddenLayers, 48)
        XCTAssertEqual(config.encoderConfiguration.attentionHeads, 8)
        XCTAssertEqual(config.encoderConfiguration.intermediateSize, 5120)
        XCTAssertEqual(config.encoderConfiguration.convKernelSize, 9)
        XCTAssertEqual(config.encoderConfiguration.subsamplingFactor, 8)
        XCTAssertEqual(config.encoderConfiguration.subsamplingConvChannels, 256)
        XCTAssertEqual(config.encoderConfiguration.numMelBins, 128)
        XCTAssertEqual(config.encoderConfiguration.hiddenAct, "silu")
        XCTAssertEqual(config.encoderConfiguration.scaleInput, false)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testNestedEncoderConfigOverridesDefaults() throws {
        let config = try decode(
            """
            {
              "model_type": "cohere_asr",
              "vocab_size": 32,
              "hidden_size": 16,
              "num_hidden_layers": 2,
              "num_attention_heads": 2,
              "intermediate_size": 32,
              "max_position_embeddings": 64,
              "encoder_config": {
                "model_type": "parakeet_encoder",
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 32,
                "conv_kernel_size": 3,
                "subsampling_conv_channels": 8,
                "num_mel_bins": 16
              }
            }
            """)

        XCTAssertEqual(config.vocabularySize, 32)
        XCTAssertEqual(config.headDim, 8)
        XCTAssertEqual(config.maxPositionEmbeddings, 64)
        XCTAssertEqual(config.encoderConfiguration.hiddenSize, 16)
        XCTAssertEqual(config.encoderConfiguration.hiddenLayers, 2)
        XCTAssertEqual(config.encoderConfiguration.convKernelSize, 3)
        XCTAssertEqual(config.encoderConfiguration.subsamplingConvChannels, 8)
        XCTAssertEqual(config.encoderConfiguration.numMelBins, 16)
        // Keys absent from the JSON keep the transformers default.
        XCTAssertEqual(config.encoderConfiguration.subsamplingFactor, 8)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testHeadDimOverrideIsRead() throws {
        let config = try decode(#"{"hidden_size": 64, "num_attention_heads": 4, "head_dim": 32}"#)
        XCTAssertEqual(config.headDim, 32)
    }

    func testUnsupportedDecoderActivationFailsValidation() throws {
        let config = try decode(#"{"hidden_act": "tanh"}"#)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testUnsupportedEncoderActivationFailsValidation() throws {
        let config = try decode(#"{"encoder_config": {"hidden_act": "tanh"}}"#)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testGroupedQueryDecoderAttentionIsRejected() throws {
        let config = try decode(#"{"num_attention_heads": 8, "num_key_value_heads": 2}"#)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testEvenConvKernelFailsValidation() throws {
        let config = try decode(#"{"encoder_config": {"conv_kernel_size": 4}}"#)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
