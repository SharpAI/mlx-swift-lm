// SharpAI fork overlay tests for unlimited_ocr config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIUnlimitedOcrConfigurationTests: XCTestCase {

    private let nestedJSON = """
        {
            "model_type": "unlimited_ocr",
            "language_config": {
                "model_type": "deepseek_v2",
                "vocab_size": 129280,
                "hidden_size": 1280,
                "intermediate_size": 6848,
                "moe_intermediate_size": 896,
                "num_hidden_layers": 12,
                "num_attention_heads": 10,
                "num_key_value_heads": 10,
                "n_routed_experts": 64,
                "n_shared_experts": 2,
                "num_experts_per_tok": 6,
                "first_k_dense_replace": 1,
                "max_position_embeddings": 32768,
                "use_mla": false,
                "kv_lora_rank": null,
                "q_lora_rank": null,
                "sliding_window": 128
            },
            "vision_config": {
                "model_type": "vision"
            },
            "projector_config": {
                "model_type": "mlp_projector"
            }
        }
        """

    func testDecodesNestedLanguageConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIUnlimitedOcrConfiguration.self, from: Data(nestedJSON.utf8))

        XCTAssertEqual(config.modelType, "unlimited_ocr")
        XCTAssertEqual(config.text.vocabSize, 129280)
        XCTAssertEqual(config.text.hiddenSize, 1280)
        XCTAssertEqual(config.text.numHiddenLayers, 12)
        XCTAssertEqual(config.text.numAttentionHeads, 10)
        XCTAssertEqual(config.text.numKeyValueHeads, 10)
        XCTAssertEqual(config.text.nRoutedExperts, 64)
        XCTAssertEqual(config.text.nSharedExperts, 2)
        XCTAssertEqual(config.text.numExpertsPerTok, 6)
        XCTAssertEqual(config.text.firstKDenseReplace, 1)
        XCTAssertEqual(config.text.slidingWindow, 128)
        XCTAssertTrue(config.hasVisionConfig)
        XCTAssertTrue(config.hasProjectorConfig)
    }

    func testNullKvLoraRankAndFalseUseMLAMeanStandardAttention() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIUnlimitedOcrConfiguration.self, from: Data(nestedJSON.utf8))

        // Upstream DeepseekV2Configuration maps null kv_lora_rank to 512; this overlay keeps it nil.
        XCTAssertNil(config.text.kvLoraRank)
        XCTAssertFalse(config.text.usesMLA)
    }

    func testSetKvLoraRankMeansMLA() throws {
        let mlaJSON = nestedJSON.replacingOccurrences(
            of: "\"kv_lora_rank\": null", with: "\"kv_lora_rank\": 512")
        let config = try JSONDecoder.json5().decode(
            SharpAIUnlimitedOcrConfiguration.self, from: Data(mlaJSON.utf8))

        XCTAssertEqual(config.text.kvLoraRank, 512)
        XCTAssertTrue(config.text.usesMLA)
    }

    func testFlatConfigFallsBackToTopLevelText() throws {
        let flatJSON = """
            {
                "model_type": "unlimited_ocr",
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 2
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIUnlimitedOcrConfiguration.self, from: Data(flatJSON.utf8))

        XCTAssertEqual(config.text.hiddenSize, 256)
        XCTAssertEqual(config.text.numKeyValueHeads, 2)
        XCTAssertFalse(config.hasVisionConfig)
        XCTAssertFalse(config.hasProjectorConfig)
    }

    func testValidConfigPassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIUnlimitedOcrConfiguration.self, from: Data(nestedJSON.utf8))
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testHeadCountMustDivideKeyValueHeads() throws {
        let badJSON = nestedJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 10", with: "\"num_key_value_heads\": 3")
        let config = try JSONDecoder.json5().decode(
            SharpAIUnlimitedOcrConfiguration.self, from: Data(badJSON.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testMissingSizesFailValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIUnlimitedOcrConfiguration.self, from: Data("{ \"hidden_size\": 64 }".utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRouterAndRopeDefaultsMatchDeepseekV2Config() throws {
        // The checkpoint omits these; configuration_deepseek_v2.py supplies the defaults.
        let config = try JSONDecoder.json5().decode(
            SharpAIUnlimitedOcrConfiguration.self, from: Data(nestedJSON.utf8))

        XCTAssertEqual(config.text.ropeTheta, 10000)
        XCTAssertEqual(config.text.rmsNormEps, 1e-6)
        XCTAssertEqual(config.text.routedScalingFactor, 1.0)
        XCTAssertFalse(config.text.normTopKProb)
        XCTAssertEqual(config.text.moeLayerFreq, 1)
        XCTAssertEqual(config.text.topkMethod, "greedy")
        XCTAssertEqual(config.text.intermediateSize, 6848)
        XCTAssertEqual(config.text.moeIntermediateSize, 896)
    }

    func testMLAConfigFailsValidation() throws {
        let mlaJSON = nestedJSON.replacingOccurrences(
            of: "\"kv_lora_rank\": null", with: "\"kv_lora_rank\": 512")
        let config = try JSONDecoder.json5().decode(
            SharpAIUnlimitedOcrConfiguration.self, from: Data(mlaJSON.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
