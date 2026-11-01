// SharpAI fork overlay tests for kolibri1 config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

/// Inline fixture. Values match scripts/conformance/offline/fixtures/kolibri1. layer_types
/// follows the fixture pattern: full_attention at index 4, 9, 14, ... and sliding elsewhere.
enum SharpAIKolibri1TestFixtures {
    static func layerTypesJSON(numHiddenLayers: Int) -> String {
        (0 ..< numHiddenLayers)
            .map { $0 % 5 == 4 ? "\"full_attention\"" : "\"sliding_attention\"" }
            .joined(separator: ", ")
    }

    static func json(numHiddenLayers: Int = 50, numExperts: Int = 384) -> String {
        """
            {
                "architectures": ["Kolibri1ForCausalLM"],
                "model_type": "kolibri1",
                "hidden_size": 2560,
                "num_hidden_layers": \(numHiddenLayers),
                "num_attention_heads": 48,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "hidden_act": "silu",
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-06,
                "vocab_size": 128000,
                "rope_theta": 10000.0,
                "num_experts": \(numExperts),
                "num_experts_per_tok": 6,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512,
                "norm_topk_prob": false,
                "attention_bias": false,
                "attention_dropout": 0.0,
                "tie_word_embeddings": false,
                "use_cache": true,
                "use_sliding_window": true,
                "sliding_window": 513,
                "layer_types": [\(layerTypesJSON(numHiddenLayers: numHiddenLayers))],
                "bos_token_id": null,
                "eos_token_id": 127906,
                "pad_token_id": 127901,
                "dtype": "bfloat16",
                "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}
            }
            """
    }

    static func configuration(
        numHiddenLayers: Int = 50, numExperts: Int = 384
    ) throws -> SharpAIKolibri1Configuration {
        try JSONDecoder.json5().decode(
            SharpAIKolibri1Configuration.self,
            from: Data(json(numHiddenLayers: numHiddenLayers, numExperts: numExperts).utf8))
    }
}

final class SharpAIKolibri1ConfigurationTests: XCTestCase {

    func testDecodesFixtureValues() throws {
        let config = try SharpAIKolibri1TestFixtures.configuration()

        XCTAssertEqual(config.modelType, "kolibri1")
        XCTAssertEqual(config.architectures, ["Kolibri1ForCausalLM"])
        XCTAssertEqual(config.vocabSize, 128_000)
        XCTAssertEqual(config.hiddenSize, 2560)
        XCTAssertEqual(config.numHiddenLayers, 50)
        XCTAssertEqual(config.numAttentionHeads, 48)
        XCTAssertEqual(config.numKeyValueHeads, 4)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.maxPositionEmbeddings, 262_144)
        XCTAssertEqual(config.rmsNormEps, 1e-6, accuracy: 1e-12)
        XCTAssertEqual(config.ropeTheta, 10_000, accuracy: 1e-3)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertFalse(config.attentionBias)
        XCTAssertEqual(config.eosTokenId, 127_906)
        XCTAssertEqual(config.padTokenId, 127_901)
    }

    func testDecodesMoEAndSlidingWindowFields() throws {
        let config = try SharpAIKolibri1TestFixtures.configuration()

        XCTAssertEqual(config.numExperts, 384)
        XCTAssertEqual(config.numExpertsPerTok, 6)
        XCTAssertEqual(config.moeIntermediateSize, 512)
        XCTAssertEqual(config.sharedExpertIntermediateSize, 512)
        XCTAssertFalse(config.normTopkProb)
        XCTAssertTrue(config.useSlidingWindow)
        XCTAssertEqual(config.slidingWindow, 513)
        XCTAssertEqual(config.layerTypes.count, 50)
    }

    func testLayerTypeHelperFollowsFixturePattern() throws {
        let config = try SharpAIKolibri1TestFixtures.configuration()

        XCTAssertTrue(config.isSlidingAttentionLayer(0))
        XCTAssertFalse(config.isSlidingAttentionLayer(4))
        XCTAssertFalse(config.isSlidingAttentionLayer(49))
        XCTAssertFalse(config.isSlidingAttentionLayer(50))
    }

    func testMissingModelTypeUsesDefault() throws {
        // UNVERIFIED: the default model_type is an assumption, pinned here so a change is visible.
        let json = SharpAIKolibri1TestFixtures.json(numHiddenLayers: 2).replacingOccurrences(
            of: "\"model_type\": \"kolibri1\",", with: "")
        let config = try JSONDecoder.json5().decode(
            SharpAIKolibri1Configuration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "kolibri1")
    }

    func testValidationRejectsLayerTypeCountMismatch() throws {
        let config = try SharpAIKolibri1TestFixtures.configuration(numHiddenLayers: 3)
        let mismatched = SharpAIKolibri1TestFixtures.json(numHiddenLayers: 3)
            .replacingOccurrences(of: "\"num_hidden_layers\": 3", with: "\"num_hidden_layers\": 4")
        let bad = try JSONDecoder.json5().decode(
            SharpAIKolibri1Configuration.self, from: Data(mismatched.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
        XCTAssertThrowsError(try bad.validateModelConfiguration())
    }
}
