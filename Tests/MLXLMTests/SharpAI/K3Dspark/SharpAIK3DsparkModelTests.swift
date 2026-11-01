// SharpAI fork overlay tests for the k3_dspark draft model. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random weights, no network.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIK3DsparkModelTests: XCTestCase {

    /// Tiny shapes. Target hidden is 6 with 2 target layers, so the context input is 12 wide.
    private func tinyJSON(confidence: Bool) -> String {
        """
        {
          "model_type": "k3_dspark",
          "architectures": ["K3DSparkModel"],
          "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
          "num_attention_heads": 2, "q_lora_rank": 8, "kv_lora_rank": 4,
          "qk_nope_head_dim": 4, "qk_rope_head_dim": 2, "v_head_dim": 4,
          "vocab_size": 50, "num_target_layers": 2, "target_hidden_size": 6,
          "target_layer_ids": [0, 1], "markov_rank": 4,
          "enable_confidence_head": \(confidence), "confidence_head_with_markov": true
        }
        """
    }

    private func tinyModel(confidence: Bool = true) throws -> SharpAIK3DsparkDraftModel {
        let config = try JSONDecoder.json5().decode(
            SharpAIK3DsparkConfiguration.self, from: Data(tinyJSON(confidence: confidence).utf8))
        return SharpAIK3DsparkDraftModel(config)
    }

    func testForwardWithoutTargetStateKeepsDraftShape() throws {
        let model = try tinyModel()
        let inputs = MLXRandom.normal([1, 3, 16])

        let out = model(inputs)
        eval(out)

        XCTAssertEqual(out.shape, [1, 3, 16])
    }

    func testForwardWithTargetStateAddsContextKeys() throws {
        let model = try tinyModel()
        let inputs = MLXRandom.normal([1, 3, 16])
        let targetHidden = MLXRandom.normal([1, 5, 12])

        let out = model(inputs, targetHidden: targetHidden, contextOffset: 0, draftOffset: 5)
        eval(out)

        XCTAssertEqual(out.shape, [1, 3, 16])
    }

    func testCombineHiddenStatesProjectsToHiddenSize() throws {
        let model = try tinyModel()
        let targetHidden = MLXRandom.normal([1, 5, 12])

        let context = model.model.combineHiddenStates(targetHidden)
        eval(context)

        XCTAssertEqual(context.shape, [1, 5, 16])
    }

    func testMarkovEmbedAndBiasShapes() throws {
        let model = try tinyModel()
        let ids = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let embed = model.markovEmbed(ids)
        let bias = model.markovBias(embed)
        eval(bias)

        XCTAssertEqual(embed.shape, [1, 3, 4])
        XCTAssertEqual(bias.shape, [1, 3, 50])
    }

    func testConfidenceIsPerPositionProbability() throws {
        let model = try tinyModel(confidence: true)
        let hidden = MLXRandom.normal([1, 3, 16])
        let embed = model.markovEmbed(MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3))

        let confidence = try XCTUnwrap(model.confidence(hidden, markovEmbed: embed))
        eval(confidence)

        XCTAssertEqual(confidence.shape, [1, 3])
        let values = confidence.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy { $0 >= 0 && $0 <= 1 })
    }

    func testConfidenceHeadAbsentWhenDisabled() throws {
        let model = try tinyModel(confidence: false)
        let hidden = MLXRandom.normal([1, 3, 16])
        let embed = model.markovEmbed(MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3))

        XCTAssertNil(model.confidence(hidden, markovEmbed: embed))
    }

    func testParameterKeysUseCheckpointNames() throws {
        let model = try tinyModel()
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("model.context_proj.weight"))
        XCTAssertTrue(keys.contains("model.context_norm.weight"))
        XCTAssertTrue(keys.contains("model.final_norm.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.input_layernorm.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.q_a_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.q_a_layernorm.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.q_b_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.kv_a_proj_with_mqa.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.kv_a_layernorm.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.kv_b_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.o_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.markov_head.markov_w1.weight"))
        XCTAssertTrue(keys.contains("model.markov_head.markov_w2.weight"))
        XCTAssertTrue(keys.contains("model.confidence_head.proj.weight"))
    }

    func testTargetOwnedWeightsAreNotParameters() throws {
        let model = try tinyModel()
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertFalse(keys.contains { $0.contains("embed_tokens") })
        XCTAssertFalse(keys.contains { $0.contains("lm_head") })
    }
}
