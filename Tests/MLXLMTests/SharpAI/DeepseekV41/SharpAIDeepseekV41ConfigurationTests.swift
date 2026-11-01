// SharpAI fork overlay tests for the deepseek_v41 configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIDeepseekV41ConfigurationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "deepseek_v41",
            "architectures": ["DeepseekV41ForCausalLM"],
            "text_config": { "model_type": "deepseek_v41_text" },
            "vision_config": { "model_type": "deepseek_v41_vision" }
        }
        """

    func testDecodesModelTypeArchitecturesAndSubModelTypes() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekV41Configuration.self, from: Data(validJSON.utf8))

        XCTAssertEqual(config.modelType, "deepseek_v41")
        XCTAssertEqual(config.architectures, ["DeepseekV41ForCausalLM"])
        XCTAssertEqual(config.textModelType, "deepseek_v41_text")
        XCTAssertEqual(config.visionModelType, "deepseek_v41_vision")
    }

    func testMissingFieldsUseDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekV41Configuration.self, from: Data("{}".utf8))

        XCTAssertEqual(config.modelType, "deepseek_v41")
        XCTAssertEqual(config.architectures, [])
        XCTAssertNil(config.textModelType)
        XCTAssertNil(config.visionModelType)
    }

    func testValidationAcceptsDeepseekV41ModelType() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekV41Configuration.self, from: Data(validJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    private func assertTextConfigRejected(_ textFields: String, file: StaticString = #filePath, line: UInt = #line) throws {
        let json = """
            { "model_type": "deepseek_v41", "text_config": { \(textFields) } }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekV41Configuration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration(), file: file, line: line) {
            error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)", file: file, line: line)
                return
            }
        }
    }

    func testValidationRejectsSourceLayerIdsOutsideCompressRatios() throws {
        try assertTextConfigRejected(#""kv_source_layer_ids": [2], "compress_ratios": [0, 2]"#)
        try assertTextConfigRejected(#""kv_source_layer_ids": [1]"#)
        try assertTextConfigRejected(#""index_source_layer_ids": [3], "compress_ratios": [0, 2]"#)
        try assertTextConfigRejected(#""kv_source_layer_ids": [0], "compress_ratios": [0, 2]"#)
        try assertTextConfigRejected(#""index_source_layer_ids": [0], "compress_ratios": [0, 2]"#)
    }

    func testValidationRejectsUnsafeDimensions() throws {
        try assertTextConfigRejected(#""o_groups": 0"#)
        try assertTextConfigRejected(#""num_attention_heads": 6, "o_groups": 4"#)
        try assertTextConfigRejected(#""n_routed_experts": 4, "num_experts_per_tok": 5"#)
        try assertTextConfigRejected(#""num_experts_per_tok": 0"#)
        try assertTextConfigRejected(#""hc_mult": 0"#)
    }

    func testValidationRejectsOtherModelType() throws {
        let json = """
            { "model_type": "deepseek_v4" }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekV41Configuration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    /// text_config values copied from deepseek-ai/DeepSeek-V4.1-Flash-1017 config.json.
    private let fullTargetJSON = """
        {
            "model_type": "deepseek_v41",
            "architectures": ["DeepseekV41ForCausalLM"],
            "text_config": {
                "model_type": "deepseek_v41_text",
                "vocab_size": 129280,
                "hidden_size": 5120,
                "num_hidden_layers": 40,
                "num_attention_heads": 64,
                "head_dim": 512,
                "qk_rope_head_dim": 64,
                "q_lora_rank": 1280,
                "o_lora_rank": 1024,
                "o_groups": 8,
                "moe_intermediate_size": 2304,
                "n_routed_experts": 384,
                "n_shared_experts": 1,
                "num_experts_per_tok": 6,
                "routed_scaling_factor": 1.5,
                "sliding_window": 128,
                "compress_ratios": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                "compress_rope_theta": 160000,
                "kv_source_layer_ids": [2, 8, 14, 20],
                "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
                "index_n_heads": 32,
                "index_head_dim": 128,
                "index_topk": 512,
                "hc_mult": 4,
                "hc_sinkhorn_iters": 20,
                "max_position_embeddings": 1048576,
                "rope_theta": 10000,
                "engram_layer_ids": [1, 14],
                "engram_max_ngram_size": 4,
                "engram_vocab_size": 16000000,
                "engram_n_heads": 8,
                "engram_head_dim": 256,
                "num_nextn_predict_layers": 3,
                "dspark_block_size": 5,
                "dspark_target_layer_ids": [37, 38, 39],
                "dspark_markov_rank": 256,
                "dspark_n_routed_experts": 128,
                "dspark_num_experts_per_tok": 3
            },
            "vision_config": { "model_type": "deepseek_v41_vision" }
        }
        """

    func testFullTargetConfigFieldsDecoded() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekV41Configuration.self, from: Data(fullTargetJSON.utf8))
        let text = try XCTUnwrap(config.text)

        XCTAssertEqual(text.hiddenSize, 5120)
        XCTAssertEqual(text.numHiddenLayers, 40)
        XCTAssertEqual(text.nRoutedExperts, 384)
        XCTAssertEqual(text.nSharedExperts, 1)
        XCTAssertEqual(text.numExpertsPerTok, 6)
        XCTAssertEqual(text.routedScalingFactor, 1.5)
        XCTAssertEqual(text.indexTopk, 512)
        XCTAssertEqual(text.hcMult, 4)
        XCTAssertEqual(text.ropeTheta, 10_000)
        XCTAssertEqual(text.compressRopeTheta, 160_000)
        XCTAssertEqual(text.compressRatios.count, 43)
        XCTAssertEqual(text.compressRatios.first, 0)
        XCTAssertEqual(text.kvSourceLayerIds, [2, 8, 14, 20])
        XCTAssertEqual(text.engramLayerIds, [1, 14])
        XCTAssertEqual(text.engramVocabSize, 16_000_000)
        XCTAssertEqual(text.numNextnPredictLayers, 3)
        XCTAssertEqual(text.dsparkBlockSize, 5)
        XCTAssertEqual(text.dsparkTargetLayerIds, [37, 38, 39])
        XCTAssertEqual(text.dsparkNumExpertsPerTok, 3)
        XCTAssertEqual(config.visionModelType, "deepseek_v41_vision")
    }

    func testTextConfigDefaultsWhenFieldsMissing() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekV41Configuration.self,
            from: Data(#"{ "text_config": { "model_type": "deepseek_v41_text" } }"#.utf8))
        let text = try XCTUnwrap(config.text)

        XCTAssertEqual(text.hiddenSize, 5120)
        XCTAssertEqual(text.numAttentionHeads, 64)
        XCTAssertEqual(text.routedScalingFactor, 1.5)
        XCTAssertEqual(text.compressRopeTheta, 160_000)
        XCTAssertNil(text.nSharedExperts)
        XCTAssertNil(text.dsparkBlockSize)
        XCTAssertEqual(text.engramLayerIds, [])
    }
}
