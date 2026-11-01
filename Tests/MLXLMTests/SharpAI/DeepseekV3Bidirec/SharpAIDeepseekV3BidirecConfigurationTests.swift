// SharpAI fork overlay tests for deepseek_v3_bidirec configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIDeepseekV3BidirecConfigurationTests: XCTestCase {

    private func decode(qLoraRank: String) throws -> SharpAIDeepseekV3BidirecConfiguration {
        let json = """
            {
                "model_type": "deepseek_v3_bidirec",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "n_group": 2,
                "topk_group": 1,
                "kv_lora_rank": 4,
                "q_lora_rank": \(qLoraRank),
                "qk_rope_head_dim": 2,
                "qk_nope_head_dim": 2,
                "v_head_dim": 4
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIDeepseekV3BidirecConfiguration.self, from: Data(json.utf8))
    }

    func testNullQLoraRankDecodesAsNil() throws {
        let config = try decode(qLoraRank: "null")
        XCTAssertNil(config.qLoraRank)
    }

    func testPresentQLoraRankDecodes() throws {
        let config = try decode(qLoraRank: "4")
        XCTAssertEqual(config.qLoraRank, 4)
    }

    func testDefaultsFillOptionalFields() throws {
        let config = try decode(qLoraRank: "null")
        XCTAssertEqual(config.modelType, "deepseek_v3_bidirec")
        XCTAssertEqual(config.numKeyValueHeads, 2)
        XCTAssertEqual(config.routedScalingFactor, 1.0)
        XCTAssertEqual(config.moeLayerFreq, 1)
        XCTAssertEqual(config.firstKDenseReplace, 0)
        XCTAssertFalse(config.attentionBias)
        XCTAssertEqual(config.moeIntermediateSize, 8)
    }

    func testValidConfigPassesValidation() throws {
        let config = try decode(qLoraRank: "null")
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testNonPositiveQLoraRankFailsValidation() throws {
        let config = try decode(qLoraRank: "0")
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testOddRopeHeadDimFailsValidation() throws {
        let json = """
            {
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "kv_lora_rank": 4,
                "qk_rope_head_dim": 3,
                "qk_nope_head_dim": 2,
                "v_head_dim": 4
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekV3BidirecConfiguration.self, from: Data(json.utf8))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
