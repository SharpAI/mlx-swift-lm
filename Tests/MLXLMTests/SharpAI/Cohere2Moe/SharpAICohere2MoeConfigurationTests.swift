// SharpAI fork overlay tests for cohere2_moe config decoding. Not part of upstream.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAICohere2MoeConfigurationTests: XCTestCase {

    private let denseJSON = """
        {
            "model_type": "cohere2_moe",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "layer_norm_eps": 1e-5,
            "vocab_size": 512,
            "logit_scale": 0.5,
            "rope_theta": 50000.0,
            "sliding_window": 4096,
            "tie_word_embeddings": false,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32
        }
        """

    func testDecodesAllFields() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAICohere2MoeConfiguration.self, from: Data(denseJSON.utf8))

        XCTAssertEqual(config.modelType, "cohere2_moe")
        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertEqual(config.intermediateSize, 128)
        XCTAssertEqual(config.attentionHeads, 8)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.vocabularySize, 512)
        XCTAssertEqual(config.logitScale, 0.5)
        XCTAssertEqual(config.ropeTheta, 50_000.0)
        XCTAssertEqual(config.slidingWindow, 4096)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertEqual(config.numExperts, 4)
        XCTAssertEqual(config.numExpertsPerTok, 2)
        XCTAssertEqual(config.moeIntermediateSize, 32)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testOptionalFieldsUseDefaults() throws {
        // Defaults follow transformers v5.10.2 configuration_cohere2_moe.py.
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "vocab_size": 512
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAICohere2MoeConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "cohere2_moe")
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.layerNormEps, 1e-5)
        XCTAssertNil(config.rmsNormEps)
        XCTAssertEqual(config.logitScale, 0.0625)
        XCTAssertEqual(config.ropeTheta, 10_000.0)
        XCTAssertEqual(config.slidingWindow, 4096)
        XCTAssertEqual(config.layerTypes, ["sliding_attention", "sliding_attention"])
        XCTAssertFalse(config.attentionBias)
        XCTAssertEqual(config.firstKDenseReplace, 0)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertNil(config.numExperts)
        XCTAssertEqual(config.numExpertsPerTok, 2)
        XCTAssertEqual(config.numSharedExperts, 0)
        XCTAssertEqual(config.sharedExpertCombinationStrategy, "average")
        XCTAssertEqual(config.expertSelectionFn, "softmax")
        XCTAssertTrue(config.normTopkProb)
    }

    func testLayerTypesDerivedFromPatterns() throws {
        // HF: full_attention when (i + 1) % pattern == 0. Prefix layers use the prefix pattern.
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "vocab_size": 512,
                "sliding_window_pattern": 2,
                "prefix_dense_sliding_window_pattern": 3,
                "first_k_dense_replace": 1
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAICohere2MoeConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(
            config.layerTypes,
            ["sliding_attention", "full_attention", "sliding_attention", "full_attention"])
    }

    func testExplicitNullSlidingWindowDisablesWindow() throws {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "vocab_size": 512,
                "sliding_window": null
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAICohere2MoeConfiguration.self, from: Data(json.utf8))

        XCTAssertNil(config.slidingWindow)
    }

    func testNumLocalExpertsFallback() throws {
        // UNVERIFIED: num_local_experts is an assumed alias for num_experts.
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "vocab_size": 512,
                "num_local_experts": 8
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAICohere2MoeConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.numExperts, 8)
    }

    func testMissingRequiredFieldThrows() {
        let json = """
            { "hidden_size": 64, "num_hidden_layers": 2 }
            """
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAICohere2MoeConfiguration.self, from: Data(json.utf8)))
    }

    func testValidationRejectsHeadMismatch() throws {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": 3,
                "vocab_size": 512
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAICohere2MoeConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
