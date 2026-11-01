// SharpAI fork overlay tests for longcat_flash config decoding and validation.
// Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAILongcatFlashConfigurationTests: XCTestCase {

    /// LongCat-Flash-Lite shape from the spec fixture. It omits n_routed_experts.
    private let liteJSON = """
        {
            "architectures": ["LongcatFlashNgramForCausalLM"],
            "attention_bias": false,
            "vocab_size": 131072,
            "hidden_size": 3072,
            "ffn_hidden_size": 6144,
            "expert_ffn_hidden_size": 1024,
            "num_layers": 14,
            "num_attention_heads": 32,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "qk_nope_head_dim": 128,
            "mla_scale_q_lora": true,
            "mla_scale_kv_lora": true,
            "routed_scaling_factor": 6.0,
            "rms_norm_eps": 1e-5,
            "rope_theta": 5000000.0,
            "max_position_embeddings": 327680,
            "rope_scaling": {
                "original_max_position_embeddings": 32768,
                "rope_type": "yarn",
                "factor": 10,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 1,
                "mscale_all_dim": 1
            },
            "zero_expert_num": 128,
            "zero_expert_type": "identity",
            "moe_topk": 12,
            "ngram_vocab_size_ratio": 78,
            "emb_neighbor_num": 4,
            "emb_split_num": 4
        }
        """

    private func decode(_ json: String) throws -> SharpAILongcatFlashConfiguration {
        try JSONDecoder.json5().decode(
            SharpAILongcatFlashConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesLiteShapeAndDerivedNgramSizes() throws {
        let config = try decode(liteJSON)

        XCTAssertEqual(config.hiddenSize, 3072)
        XCTAssertEqual(config.numLayers, 14)
        // UNVERIFIED: 512 is the default used when the key is absent.
        XCTAssertEqual(config.nRoutedExperts, 512)
        XCTAssertEqual(config.zeroExpertNum, 128)
        XCTAssertEqual(config.moeTopk, 12)
        XCTAssertEqual(config.ropeScaling?["factor"]?.asFloat(), 10)
        XCTAssertEqual(config.ngramVocabSize, 78 * 131072)
        XCTAssertEqual(config.ngramEmbedderCount, 12)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testExplicitNRoutedExpertsOverridesDefault() throws {
        let json = liteJSON.replacingOccurrences(
            of: "\"rms_norm_eps\": 1e-5,", with: "\"n_routed_experts\": 256,\n            \"rms_norm_eps\": 1e-5,")
        let config = try decode(json)

        XCTAssertEqual(config.nRoutedExperts, 256)
    }

    func testOptionalKeysFallBackToDefaults() throws {
        // model_type, zero_expert_type, router/attention bias and the n-gram keys are absent.
        // The Lite fixture has no model_type. Removed keys are never the last one,
        // so the JSON stays valid.
        let json = liteJSON
            .replacingOccurrences(of: "\"attention_bias\": false,", with: "")
            .replacingOccurrences(of: "\"zero_expert_type\": \"identity\",", with: "")
            .replacingOccurrences(of: "\"ngram_vocab_size_ratio\": 78,", with: "")
            .replacingOccurrences(of: "\"emb_neighbor_num\": 4,", with: "")
        let config = try decode(json)

        XCTAssertEqual(config.modelType, "longcat_flash")
        XCTAssertEqual(config.zeroExpertType, "identity")
        XCTAssertFalse(config.attentionBias)
        XCTAssertFalse(config.routerBias)
        XCTAssertFalse(config.normTopkProb)
        XCTAssertEqual(config.ngramVocabSizeRatio, 78)
        XCTAssertEqual(config.embNeighborNum, 4)
        XCTAssertEqual(config.embSplitNum, 4)
    }

    func testRejectsZeroExpertTypeOtherThanIdentity() throws {
        let json = liteJSON.replacingOccurrences(
            of: "\"zero_expert_type\": \"identity\"", with: "\"zero_expert_type\": \"copy\"")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsHiddenSizeNotDivisibleByNgramEmbedderCount() throws {
        // Tiny fixture with n = 4: 4 * (4 - 1) = 6 embedders, and 16 % 6 != 0.
        let json = SharpAILongcatFlashTestFixtures.json()
            .replacingOccurrences(of: "\"emb_neighbor_num\": 3", with: "\"emb_neighbor_num\": 4")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testTinyFixtureIsValid() throws {
        let config = try SharpAILongcatFlashTestFixtures.configuration()

        XCTAssertNoThrow(try config.validateModelConfiguration())
        XCTAssertEqual(config.ngramVocabSize, 128)
        XCTAssertEqual(config.ngramEmbedderCount, 4)
    }
}
