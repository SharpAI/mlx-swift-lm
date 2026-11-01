// SharpAI fork overlay tests for the interns1_pro configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIInterns1ProConfigurationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "interns1_pro",
            "architectures": ["InternS1ProForConditionalGeneration"]
        }
        """

    func testDecodesModelTypeAndArchitectures() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns1ProConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertEqual(config.modelType, "interns1_pro")
        XCTAssertEqual(config.architectures, ["InternS1ProForConditionalGeneration"])
        XCTAssertNil(config.textConfig)
    }

    func testMissingModelTypeDefaultsToInterns1Pro() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns1ProConfiguration.self, from: Data("{}".utf8))

        XCTAssertEqual(config.modelType, "interns1_pro")
        XCTAssertEqual(config.architectures, [])
    }

    func testValidationAcceptsInterns1ProModelType() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns1ProConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsOtherModelType() throws {
        let json = """
            { "model_type": "llama" }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns1ProConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    /// text_config and rope_scaling values copied from internlm/Intern-S1-Pro config.json
    /// (verified against https://huggingface.co/internlm/Intern-S1-Pro/resolve/main/config.json).
    func testDecodesReferenceTextConfig() throws {
        let json = """
            {
                "architectures": ["InternS1ProForConditionalGeneration"],
                "model_type": "interns1_pro",
                "tie_word_embeddings": false,
                "text_config": {
                    "attention_bias": false,
                    "attention_dropout": 0.0,
                    "decoder_sparse_step": 1,
                    "dtype": "bfloat16",
                    "head_dim": 128,
                    "hidden_act": "silu",
                    "hidden_size": 4096,
                    "intermediate_size": 12288,
                    "max_position_embeddings": 262144,
                    "mlp_only_layers": [],
                    "model_type": "interns1_pro_text",
                    "moe_intermediate_size": 1536,
                    "norm_topk_prob": true,
                    "num_attention_heads": 64,
                    "num_experts": 512,
                    "num_experts_per_tok": 8,
                    "num_hidden_layers": 94,
                    "num_key_value_heads": 4,
                    "rms_norm_eps": 1e-06,
                    "rope_scaling": {
                        "rope_type": "default",
                        "fope_init_factor": 0.5,
                        "fope_sep_head": true,
                        "num_inv_freq": null
                    },
                    "rope_theta": 5000000,
                    "router_n_groups": 8,
                    "vocab_size": 155008
                }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns1ProConfiguration.self, from: Data(json.utf8))
        let text = try XCTUnwrap(config.textConfig)

        XCTAssertEqual(text.hiddenLayers, 94)
        XCTAssertEqual(text.attentionHeads, 64)
        XCTAssertEqual(text.kvHeads, 4)
        XCTAssertEqual(text.headDim, 128)
        XCTAssertEqual(text.numExperts, 512)
        XCTAssertEqual(text.numExpertsPerToken, 8)
        XCTAssertEqual(text.moeIntermediateSize, 1536)
        XCTAssertEqual(text.routerNGroups, 8)
        XCTAssertEqual(text.ropeTheta, 5_000_000)
        XCTAssertTrue(text.useFope)
        XCTAssertTrue(text.fopeSepHead)
        XCTAssertEqual(text.fopeInitFactor, 0.5)
        XCTAssertNil(text.numInvFreq)
        XCTAssertTrue(text.normTopkProb)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRejectsExpertsNotDivisibleByGroups() throws {
        let json = SharpAIInterns1ProTinyFixture.fope.replacingOccurrences(
            of: "\"router_n_groups\": 2", with: "\"router_n_groups\": 3")
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns1ProConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
