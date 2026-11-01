// SharpAI fork overlay tests for openpangu_vl config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIOpenpanguVlConfigurationTests: XCTestCase {

    // Values copied from scripts/conformance/offline/fixtures/openpangu_vl/
    // FreedomIntelligence__openPangu-VL-7B.json. Only the fields the config reads are kept.
    private let fixtureJSON = """
        {
            "architectures": ["OpenPanguVLForConditionalGeneration"],
            "bias": true,
            "bos_token_id": 1,
            "eos_token_id": 45892,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "intermediate_size": 12800,
            "max_position_embeddings": 131072,
            "model_type": "openpangu_vl",
            "num_attention_heads": 32,
            "num_hidden_layers": 34,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 64000000.0,
            "tie_word_embeddings": false,
            "use_cache": true,
            "vision_config": {
                "rms_norm_eps": 1e-06,
                "depth": 26,
                "hidden_act": "gelu",
                "hidden_size": 1536,
                "intermediate_size": 4608,
                "num_heads": 16,
                "in_chans": 3,
                "out_hidden_size": 3584,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "spatial_patch_size": 14,
                "window_size": 112,
                "fullatt_block_indexes": [5, 12, 19, 25],
                "tokens_per_second": 2,
                "temporal_patch_size": 2
            },
            "rope_scaling": {
                "rope_type": "default",
                "type": "default",
                "mrope_section": [10, 27, 27],
                "mrope_interleaved": true
            },
            "vocab_size": 153376
        }
        """

    private func decode(_ json: String) throws -> SharpAIOpenpanguVlConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIOpenpanguVlConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesFixtureTextDimensions() throws {
        let config = try decode(fixtureJSON)

        XCTAssertEqual(config.modelType, "openpangu_vl")
        XCTAssertEqual(config.vocabularySize, 153376)
        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.hiddenLayers, 34)
        XCTAssertEqual(config.intermediateSize, 12800)
        XCTAssertEqual(config.attentionHeads, 32)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertTrue(config.attentionBias)
        XCTAssertEqual(config.rmsNormEps, 1e-5)
        XCTAssertEqual(config.ropeTheta, 64_000_000.0)
        XCTAssertEqual(config.maxPositionEmbeddings, 131072)
        XCTAssertFalse(config.tieWordEmbeddings)
    }

    func testDecodesFixtureMropeSection() throws {
        let config = try decode(fixtureJSON)

        XCTAssertEqual(config.mropeSection, [10, 27, 27])
        XCTAssertTrue(config.mropeInterleaved)
    }

    func testDecodesFixtureVisionBlock() throws {
        let config = try decode(fixtureJSON)
        let vision = try XCTUnwrap(config.vision)

        XCTAssertEqual(vision.depth, 26)
        XCTAssertEqual(vision.hiddenSize, 1536)
        XCTAssertEqual(vision.intermediateSize, 4608)
        XCTAssertEqual(vision.numHeads, 16)
        XCTAssertEqual(vision.inChannels, 3)
        XCTAssertEqual(vision.outHiddenSize, 3584)
        XCTAssertEqual(vision.patchSize, 14)
        XCTAssertEqual(vision.spatialMergeSize, 2)
        XCTAssertEqual(vision.temporalPatchSize, 2)
        XCTAssertEqual(vision.windowSize, 112)
        XCTAssertEqual(vision.fullAttentionBlockIndexes, [5, 12, 19, 25])
    }

    func testFixtureConfigPassesValidation() throws {
        let config = try decode(fixtureJSON)

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingKeyValueHeadsDefaultsToAttentionHeads() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 8,", with: "")
        let config = try decode(json)

        XCTAssertEqual(config.kvHeads, 32)
    }

    func testMissingBiasDefaultsToFalse() throws {
        let json = fixtureJSON.replacingOccurrences(of: "\"bias\": true,", with: "")
        let config = try decode(json)

        XCTAssertFalse(config.attentionBias)
    }

    func testMissingRopeScalingLeavesMropeEmpty() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"mrope_section\": [10, 27, 27],", with: "")
        let config = try decode(json)

        XCTAssertTrue(config.mropeSection.isEmpty)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingVisionConfigFailsValidation() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"vision_config\": {", with: "\"vision_config_removed\": {")
        let config = try decode(json)

        XCTAssertNil(config.vision)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testKeyValueHeadsMustDivideAttentionHeads() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 8", with: "\"num_key_value_heads\": 3")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testMropeSectionMustSumToHalfHeadDim() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"mrope_section\": [10, 27, 27]", with: "\"mrope_section\": [10, 27, 26]")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
