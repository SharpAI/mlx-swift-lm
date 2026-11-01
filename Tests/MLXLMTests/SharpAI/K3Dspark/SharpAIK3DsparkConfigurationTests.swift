// SharpAI fork overlay tests for the k3_dspark config. Not part of upstream.
// Offline only: inline JSON, no checkpoint.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIK3DsparkConfigurationTests: XCTestCase {

    /// Copied from scripts/conformance/offline/fixtures/k3_dspark/Inferact__Kimi-K3-DSpark.json.
    private let fixtureJSON = """
        {
          "architectures": ["K3DSparkModel"],
          "model_type": "k3_dspark",
          "hidden_size": 7168,
          "intermediate_size": 14336,
          "num_hidden_layers": 5,
          "num_attention_heads": 64,
          "num_key_value_heads": 64,
          "q_lora_rank": 1536,
          "kv_lora_rank": 512,
          "qk_nope_head_dim": 128,
          "qk_rope_head_dim": 64,
          "v_head_dim": 128,
          "mla_use_nope": false,
          "mla_use_output_gate": false,
          "vocab_size": 163840,
          "rms_norm_eps": 1e-05,
          "max_position_embeddings": 1048576,
          "rope_theta": 50000.0,
          "num_target_layers": 5,
          "target_hidden_size": 7168,
          "target_num_hidden_layers": 93,
          "target_layer_ids": [2, 23, 47, 71, 89],
          "mask_token_id": 163837,
          "bos_token_id": 163584,
          "eos_token_id": 163586,
          "pad_token_id": 163839,
          "markov_rank": 256,
          "markov_head_type": "vanilla",
          "enable_confidence_head": true,
          "confidence_head_with_markov": true,
          "tie_word_embeddings": false,
          "draft_vocab_size": 163840,
          "torch_dtype": "bfloat16",
          "rope_parameters": {
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "rope_theta": 50000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0
          }
        }
        """

    private func decode(_ json: String) throws -> SharpAIK3DsparkConfiguration {
        try JSONDecoder.json5().decode(SharpAIK3DsparkConfiguration.self, from: Data(json.utf8))
    }

    /// Fixture with keys replaced or removed. A nil value removes the key.
    private func fixture(overriding overrides: [String: Any?]) throws -> String {
        var object = try JSONSerialization.jsonObject(with: Data(fixtureJSON.utf8)) as! [String: Any]
        for (key, value) in overrides {
            object[key] = value
        }
        let data = try JSONSerialization.data(withJSONObject: object)
        return String(decoding: data, as: UTF8.self)
    }

    func testDecodesFixtureFields() throws {
        let config = try decode(fixtureJSON)

        XCTAssertEqual(config.modelType, "k3_dspark")
        XCTAssertEqual(config.architectures, ["K3DSparkModel"])
        XCTAssertEqual(config.hiddenSize, 7168)
        XCTAssertEqual(config.intermediateSize, 14336)
        XCTAssertEqual(config.numHiddenLayers, 5)
        XCTAssertEqual(config.numAttentionHeads, 64)
        XCTAssertEqual(config.qLoraRank, 1536)
        XCTAssertEqual(config.kvLoraRank, 512)
        XCTAssertEqual(config.qkNopeHeadDim, 128)
        XCTAssertEqual(config.qkRopeHeadDim, 64)
        XCTAssertEqual(config.vHeadDim, 128)
        XCTAssertEqual(config.vocabSize, 163840)
        XCTAssertEqual(config.draftVocabSize, 163840)
        XCTAssertEqual(config.maxPositionEmbeddings, 1048576)
        XCTAssertEqual(config.ropeTheta, 50000, accuracy: 1)
        XCTAssertEqual(config.numTargetLayers, 5)
        XCTAssertEqual(config.targetHiddenSize, 7168)
        XCTAssertEqual(config.targetLayerIds, [2, 23, 47, 71, 89])
        XCTAssertEqual(config.markovRank, 256)
        XCTAssertTrue(config.enableConfidenceHead)
        XCTAssertTrue(config.confidenceHeadWithMarkov)
    }

    func testDecodesYarnRopeParameters() throws {
        let config = try decode(fixtureJSON)

        XCTAssertEqual(config.ropeScaling?["rope_type"], .string("yarn"))
        XCTAssertEqual(config.ropeScaling?["factor"]?.asFloat(), 32)
        XCTAssertEqual(config.ropeScaling?["mscale_all_dim"]?.asFloat(), 1)
    }

    func testFixtureValidates() throws {
        let config = try decode(fixtureJSON)

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testOptionalFieldsUseDefaults() throws {
        let json = """
            {
              "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
              "num_attention_heads": 2, "q_lora_rank": 8, "kv_lora_rank": 4,
              "qk_nope_head_dim": 4, "qk_rope_head_dim": 2, "v_head_dim": 4,
              "vocab_size": 50, "num_target_layers": 2, "target_hidden_size": 6,
              "target_layer_ids": [0, 1], "markov_rank": 4
            }
            """
        let config = try decode(json)

        XCTAssertEqual(config.modelType, "k3_dspark")
        XCTAssertEqual(config.architectures, [])
        XCTAssertEqual(config.draftVocabSize, 50)
        XCTAssertEqual(config.rmsNormEps, 1e-6, accuracy: 1e-9)
        XCTAssertEqual(config.ropeTheta, 50000)
        XCTAssertNil(config.ropeScaling)
        XCTAssertFalse(config.enableConfidenceHead)
        XCTAssertFalse(config.confidenceHeadWithMarkov)
        XCTAssertFalse(config.mlaUseNope)
        XCTAssertFalse(config.mlaUseOutputGate)
        XCTAssertFalse(config.mlaUseQkNorm)
        XCTAssertFalse(config.dsparkBonusAnchor)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRopeScalingFallsBackToRopeScalingKey() throws {
        let json = try fixture(overriding: [
            "rope_parameters": nil,
            "rope_scaling": ["rope_type": "yarn", "factor": 4.0],
        ])
        let config = try decode(json)

        XCTAssertEqual(config.ropeScaling?["factor"]?.asFloat(), 4)
    }

    func testRejectsOtherModelType() throws {
        let config = try decode(try fixture(overriding: ["model_type": "deepseek_v4"]))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testRejectsUnsupportedMLAOptions() throws {
        let overrides: [[String: Any?]] = [
            ["mla_use_nope": true],
            ["mla_use_output_gate": true],
            ["mla_use_qk_norm": true],
            ["dspark_bonus_anchor": true],
            ["q_lora_rank": nil],
            ["draft_vocab_size": 8192],
            ["target_layer_ids": []],
            ["num_target_layers": 4],
        ]
        for override in overrides {
            let config = try decode(try fixture(overriding: override))
            XCTAssertThrowsError(try config.validateModelConfiguration(), "\(override)") { error in
                guard case ModelFactoryError.invalidConfiguration(let message) = error else {
                    XCTFail("unexpected error: \(error)")
                    return
                }
                XCTAssertTrue(message.contains("MLA DSpark"), message)
            }
        }
    }
}
