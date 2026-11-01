// SharpAI fork overlay tests for the muse_spark_dspark draft model. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random weights, no network.
// UNVERIFIED: these checks cover shapes, parameter names and the output transform. They do not
// compare against vLLM numerics, because no target model or reference output is available.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIMuseSparkDsparkModelTests: XCTestCase {

    /// Tiny shapes. Target hidden is 6 with 2 target layers, so the context input is 12 wide.
    private func tinyJSON(confidence: Bool, residualGate: Bool = true, softCap: String = "null")
        -> String
    {
        """
        {
          "model_type": "muse_spark_dspark",
          "architectures": ["MuseSparkDSparkModel"],
          "hidden_size": 16, "ffn_dim_multiplier": 1.5,
          "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
          "head_dim": 8, "vocab_size": 50, "draft_vocab_size": 40,
          "moe": {"is_enabled": false},
          "global_attn_cfg": [4, 0],
          "target_layer_ids": [1, 3], "num_target_layers": 2, "target_hidden_size": 6,
          "markov_rank": 4,
          "enable_confidence_head": \(confidence), "confidence_head_with_markov": true,
          "use_attn_o_gate": true, "use_attn_o_norm": true,
          "use_residual_gate": \(residualGate),
          "output_soft_cap_temp": \(softCap),
          "draft_position_mode": "rope"
        }
        """
    }

    private func tinyModel(confidence: Bool = true, residualGate: Bool = true, softCap: String = "null")
        throws -> SharpAIMuseSparkDsparkDraftModel
    {
        let config = try JSONDecoder.json5().decode(
            SharpAIMuseSparkDsparkConfiguration.self,
            from: Data(tinyJSON(confidence: confidence, residualGate: residualGate, softCap: softCap).utf8))
        return SharpAIMuseSparkDsparkDraftModel(config)
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

    func testForwardWithoutGateKeepsDraftShape() throws {
        let model = try tinyModel(residualGate: false)
        let inputs = MLXRandom.normal([1, 3, 16])

        let out = model(inputs)
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

    func testDraftLogitsUseDraftVocab() throws {
        let model = try tinyModel()
        let hidden = MLXRandom.normal([1, 3, 16])

        let logits = model.draftLogits(hidden)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 40])
    }

    func testMarkovEmbedAndBiasShapes() throws {
        let model = try tinyModel()
        let ids = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let embed = model.markovEmbed(ids)
        let bias = model.markovBias(embed)
        eval(bias)

        XCTAssertEqual(embed.shape, [1, 3, 4])
        XCTAssertEqual(bias.shape, [1, 3, 40])
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

    func testSoftCapBoundsLogits() throws {
        let model = try tinyModel(softCap: "2.0")
        let logits = MLXArray([100, -100, 3] as [Float]).reshaped(1, 3)

        let capped = model.applyOutputLogitTransform(logits)
        eval(capped)

        let values = capped.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy { abs($0) <= 2.0001 })
    }

    func testParameterKeysUseCheckpointNames() throws {
        let model = try tinyModel()
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("model.fc.weight"))
        XCTAssertTrue(keys.contains("model.hidden_norm.weight"))
        XCTAssertTrue(keys.contains("model.norm.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.o_gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.pre_qkv_proj_norm.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.post_attn_norm.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.add_attn.gate"))
        XCTAssertTrue(keys.contains("model.layers.1.feed_forward.down_proj.weight"))
        XCTAssertTrue(keys.contains("model.markov_head.markov_w1.weight"))
        XCTAssertTrue(keys.contains("model.markov_head.markov_w2.weight"))
        XCTAssertTrue(keys.contains("model.confidence_head.proj.weight"))
        XCTAssertTrue(keys.contains("lm_head.weight"))
        XCTAssertTrue(keys.contains("draft_id_to_target_id"))
    }

    func testTargetOwnedWeightsAreNotParameters() throws {
        let model = try tinyModel()
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertFalse(keys.contains { $0.contains("embed_tokens") })
        XCTAssertFalse(keys.contains { $0.contains("mask_embedding") })
    }

    func testGateParameterAbsentWhenDisabled() throws {
        let model = try tinyModel(residualGate: false)
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertFalse(keys.contains { $0.contains("add_attn.gate") })
    }
}
