// SharpAI fork overlay tests for alice_ai config decoding. Not part of upstream.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIAliceAiConfigurationTests: XCTestCase {

    /// Keys from scripts/conformance/offline/fixtures/alice_ai (80B-A3B base config).
    private func fixtureData(layerTypes: [String]? = nil, extra: [String: Any] = [:]) throws -> Data {
        let types =
            layerTypes
            ?? (1 ... 48).map { $0 % 4 == 0 ? "full_attention" : "linear_attention" }
        var json: [String: Any] = [
            "model_type": "alice_ai",
            "architectures": ["AliceAIForCausalLM"],
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "layer_types": types,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "vocab_size": 129024,
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
            "max_position_embeddings": 262144,
            "tie_word_embeddings": false,
            "linear_num_key_heads": 32,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "block_attn_res_block_size": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "router_score_function": "sigmoid",
            "mtp_num_hidden_layers": 1,
            "number_of_conv_states": 3,
        ]
        for (key, value) in extra {
            json[key] = value
        }
        return try JSONSerialization.data(withJSONObject: json)
    }

    func testDecodesCheckpointConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIAliceAiConfiguration.self, from: try fixtureData())
        try config.validateModelConfiguration()

        XCTAssertEqual(config.modelType, "alice_ai")
        XCTAssertEqual(config.hiddenSize, 2048)
        XCTAssertEqual(config.hiddenLayers, 48)
        XCTAssertEqual(config.attentionHeads, 16)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.headDim, 256)
        XCTAssertEqual(config.layerTypes.count, 48)
        XCTAssertEqual(config.layerTypes[0], "linear_attention")
        XCTAssertEqual(config.layerTypes[3], "full_attention")
        XCTAssertEqual(config.attnResBlockSize, 4)
        XCTAssertEqual(config.linearNumKeyHeads, 32)
        XCTAssertEqual(config.linearKeyHeadDim, 128)
        XCTAssertEqual(config.linearConvKernelDim, 4)
        XCTAssertEqual(config.numExperts, 512)
        XCTAssertEqual(config.numExpertsPerTok, 10)
        XCTAssertEqual(config.moeIntermediateSize, 512)
        XCTAssertEqual(config.sharedExpertIntermediateSize, 512)
        XCTAssertEqual(config.vocabularySize, 129024)
        XCTAssertEqual(config.partialRotaryFactor, 0.25, accuracy: 1e-6)
        XCTAssertFalse(config.tieWordEmbeddings)
    }

    func testNormTopkProbDefaultsToTrueWhenAbsent() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIAliceAiConfiguration.self, from: try fixtureData())
        XCTAssertTrue(config.normTopkProb)
    }

    func testRejectsLayerTypeCountMismatch() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIAliceAiConfiguration.self,
            from: try fixtureData(layerTypes: ["linear_attention", "full_attention"]))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsUnknownLayerType() throws {
        var types = (1 ... 48).map { $0 % 4 == 0 ? "full_attention" : "linear_attention" }
        types[0] = "sliding_attention"
        let config = try JSONDecoder.json5().decode(
            SharpAIAliceAiConfiguration.self, from: try fixtureData(layerTypes: types))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsZeroAttnResBlockSize() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIAliceAiConfiguration.self,
            from: try fixtureData(extra: ["block_attn_res_block_size": 0]))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsNonSigmoidRouter() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIAliceAiConfiguration.self,
            from: try fixtureData(extra: ["router_score_function": "softmax"]))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
