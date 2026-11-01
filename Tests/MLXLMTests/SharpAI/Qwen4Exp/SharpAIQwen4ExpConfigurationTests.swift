// SharpAI fork overlay tests for the qwen4_exp configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIQwen4ExpConfigurationTests: XCTestCase {

    private func decodeFixture() throws -> SharpAIQwen4ExpConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIQwen4ExpConfiguration.self, from: Data(SharpAIQwen4ExpFixtures.configJSON.utf8))
    }

    func testDecodesTopLevelFields() throws {
        let config = try decodeFixture()

        XCTAssertEqual(config.modelType, "qwen4_exp")
        XCTAssertEqual(config.architectures, ["Qwen4ExpForConditionalGeneration"])
        XCTAssertEqual(config.imageTokenId, 248056)
        XCTAssertEqual(config.videoTokenId, 248057)
        XCTAssertEqual(config.visionStartTokenId, 248053)
        XCTAssertEqual(config.visionEndTokenId, 248054)
        XCTAssertEqual(config.tieWordEmbeddings, false)
    }

    func testDecodesTextDimensionsAndLayerTypes() throws {
        let text = try decodeFixture().textConfig

        XCTAssertEqual(text.hiddenSize, 2560)
        XCTAssertEqual(text.numHiddenLayers, 48)
        XCTAssertEqual(text.numAttentionHeads, 24)
        XCTAssertEqual(text.numKeyValueHeads, 2)
        XCTAssertEqual(text.headDim, 256)
        XCTAssertEqual(text.vocabSize, 248320)
        XCTAssertEqual(text.layerTypes.count, 48)
        XCTAssertEqual(text.layerTypes[0], "linear_attention")
        XCTAssertEqual(text.layerTypes[3], "full_attention")
        XCTAssertEqual(text.layerTypes[47], "full_attention")
        XCTAssertEqual(text.fullAttentionInterval, 4)
    }

    func testDecodesGatedDeltaAndOutputGate() throws {
        let text = try decodeFixture().textConfig

        XCTAssertEqual(text.linearNumKeyHeads, 16)
        XCTAssertEqual(text.linearNumValueHeads, 48)
        XCTAssertEqual(text.linearKeyHeadDim, 128)
        XCTAssertEqual(text.linearValueHeadDim, 128)
        XCTAssertEqual(text.linearConvKernelDim, 4)
        XCTAssertEqual(text.mambaSsmDtype, "float32")
        XCTAssertEqual(text.outputGateType, "sigmoid")
    }

    func testDecodesMoeFields() throws {
        let text = try decodeFixture().textConfig

        XCTAssertEqual(text.numExperts, 512)
        XCTAssertEqual(text.numExpertsPerTok, 10)
        XCTAssertEqual(text.moeIntermediateSize, 640)
        XCTAssertEqual(text.sharedExpertIntermediateSize, 640)
    }

    func testDecodesRopeParameters() throws {
        let rope = try decodeFixture().textConfig.ropeParameters

        XCTAssertEqual(rope.ropeType, "default")
        XCTAssertEqual(rope.ropeTheta, 10_000_000)
        XCTAssertEqual(rope.partialRotaryFactor, 0.25)
        XCTAssertEqual(rope.mropeInterleaved, true)
        XCTAssertEqual(rope.mropeSection, [11, 11, 10])
    }

    func testDecodesGapFieldsForUnimplementedParts() throws {
        let text = try decodeFixture().textConfig

        // Hyper-connections.
        XCTAssertEqual(text.hcCount, 4)
        XCTAssertEqual(text.hcLowrank, 320)
        // N-gram embedding.
        XCTAssertEqual(text.ngramSize, 3)
        XCTAssertEqual(text.ngramVocabSizeBase, 20_000_000)
        XCTAssertEqual(text.headsPerNgram, 8)
        XCTAssertEqual(text.splitNgramParts, 128)
        XCTAssertEqual(text.makeNgramVocabSizeDivisibleBy, 128)
        // Per-layer embedding.
        XCTAssertEqual(text.pleEmbedDim, 2560)
        XCTAssertEqual(text.pleConvKernelSize, 4)
        XCTAssertEqual(text.pleLayerIds, [2])
        // Sparse indexer.
        XCTAssertEqual(text.indexerBudget, 2048)
        XCTAssertEqual(text.indexerCompressRatio, 4)
        XCTAssertEqual(text.indexerHeadDim, 128)
        XCTAssertEqual(text.indexerKvHeads, 1)
        XCTAssertEqual(text.indexerNHeads, 4)
    }

    func testDecodesMtpBlock() throws {
        let text = try decodeFixture().textConfig

        XCTAssertEqual(text.mtpNumHiddenLayers, 1)
        XCTAssertEqual(text.mtpUseDedicatedEmbeddings, false)
        XCTAssertEqual(text.mtp?.hybrid, true)
        XCTAssertEqual(text.mtp?.layerTypes, ["full_attention"])
        XCTAssertEqual(text.mtp?.numHiddenLayers, 1)
        XCTAssertEqual(text.mtp?.ropeTheta, 10_000_000)
        XCTAssertNil(text.mtp?.mtpUseHiddenStateFromLayer)
    }

    func testDecodesVisionConfig() throws {
        let vision = try XCTUnwrap(decodeFixture().visionConfig)

        XCTAssertEqual(vision.modelType, "qwen4_exp")
        XCTAssertEqual(vision.depth, 27)
        XCTAssertEqual(vision.hiddenSize, 1152)
        XCTAssertEqual(vision.numHeads, 16)
        XCTAssertEqual(vision.patchSize, 16)
        XCTAssertEqual(vision.spatialMergeSize, 2)
        XCTAssertEqual(vision.temporalPatchSize, 2)
        XCTAssertEqual(vision.outHiddenSize, 2560)
    }

    func testValidationAcceptsFixture() throws {
        let config = try decodeFixture()

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsOtherModelType() throws {
        let json = SharpAIQwen4ExpFixtures.configJSON.replacingOccurrences(
            of: "\"model_type\": \"qwen4_exp\",", with: "\"model_type\": \"llama\",")
        let config = try JSONDecoder.json5().decode(
            SharpAIQwen4ExpConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testValidationRejectsLayerCountMismatch() throws {
        let json = SharpAIQwen4ExpFixtures.configJSON.replacingOccurrences(
            of: "\"num_hidden_layers\": 48,", with: "\"num_hidden_layers\": 47,")
        let config = try JSONDecoder.json5().decode(
            SharpAIQwen4ExpConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testDecodingFailsWhenRequiredTextFieldIsMissing() {
        let json = """
            {
                "model_type": "qwen4_exp",
                "text_config": { "model_type": "qwen4_exp_text" }
            }
            """

        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIQwen4ExpConfiguration.self, from: Data(json.utf8)))
    }
}
