// SharpAI fork overlay tests for the pangu_ultra_moe configuration. Not part of upstream.
// UNVERIFIED: key names come from the usual HF convention, not from a Pangu Ultra MoE config.
// Written without compiling. Run it before relying on it.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIPanguUltraMoeConfigurationTests: XCTestCase {

    private let json = """
        {
            "architectures": ["PanguUltraMoEForCausalLM"],
            "model_type": "pangu_ultra_moe",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 1000,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 4096,
            "tie_word_embeddings": false,
            "num_routed_experts": 16,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 64,
            "num_shared_experts": 1,
            "num_dense_layers": 1
        }
        """

    /// config.json of openpangu/openPangu-Ultra-MoE-718B-V1.1, copied verbatim.
    private let publishedConfigJSON = """
        {
          "architectures": [
            "PanguUltraMoEForCausalLM"
          ],
          "attention_bias": false,
          "auto_map": {
            "AutoConfig": "configuration_openpangu_moe.PanguUltraMoEConfig",
            "AutoModel": "modeling_openpangu_moe.PanguUltraMoEModel",
            "AutoModelForCausalLM": "modeling_openpangu_moe.PanguUltraMoEForCausalLM"
          },
          "num_dense_layers": 3,
          "hidden_act": "silu",
          "hidden_size": 7680,
          "initializer_range": 0.02,
          "intermediate_size": 18432,
          "attention_kv_lora_dim": 512,
          "max_position_embeddings": 131072,
          "model_type": "pangu_ultra_moe",
          "moe_intermediate_size": 2048,
          "num_routed_experts": 256,
          "num_shared_experts": 1,
          "num_attention_heads": 128,
          "num_experts_per_tok": 8,
          "num_hidden_layers": 61,
          "num_key_value_heads": 128,
          "num_mtp_layers": 1,
          "attention_q_lora_dim": 1536,
          "attention_qk_dim": 128,
          "attention_qk_rope_dim": 64,
          "rms_norm_eps": 1e-05,
          "rope_theta": 25600000,
          "routed_scaling_factor": 2.5,
          "sandwich_norm": true,
          "tie_word_embeddings": false,
          "torch_dtype": "bfloat16",
          "transformers_version": "4.48.2",
          "use_cache": true,
          "attention_v_dim": 128,
          "vocab_size": 153600
        }
        """

    private func decode(_ text: String) throws -> SharpAIPanguUltraMoeConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIPanguUltraMoeConfiguration.self, from: Data(text.utf8))
    }

    func testDecodesCoreFields() throws {
        let config = try decode(json)

        XCTAssertEqual(config.modelType, "pangu_ultra_moe")
        XCTAssertEqual(config.hiddenSize, 256)
        XCTAssertEqual(config.hiddenLayers, 4)
        XCTAssertEqual(config.attentionHeads, 8)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.vocabularySize, 1000)
        XCTAssertEqual(config.ropeTheta, 1_000_000, accuracy: 1)
        XCTAssertEqual(config.maxPositionEmbeddings, 4096)
        XCTAssertFalse(config.tieWordEmbeddings)
    }

    func testDecodesMoEFields() throws {
        let config = try decode(json)

        XCTAssertEqual(config.routedExperts, 16)
        XCTAssertEqual(config.expertsPerToken, 4)
        XCTAssertEqual(config.moeIntermediateSize, 64)
        XCTAssertEqual(config.sharedExperts, 1)
        XCTAssertEqual(config.firstDenseLayers, 1)
    }

    func testMissingKVHeadsDefaultsToAttentionHeads() throws {
        let text = json.replacingOccurrences(of: "\"num_key_value_heads\": 2,", with: "")
        let config = try decode(text)
        XCTAssertEqual(config.kvHeads, 8)
    }

    func testMissingOptionalFieldsUseDefaults() throws {
        let text = """
            {
                "model_type": "pangu_ultra_moe",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "vocab_size": 100
            }
            """
        let config = try decode(text)

        XCTAssertEqual(config.rmsNormEps, 1e-5, accuracy: 1e-9)
        // Class default in configuration_openpangu_moe.py (rope_theta=25600000).
        XCTAssertEqual(config.ropeTheta, 25_600_000, accuracy: 1)
        XCTAssertEqual(config.hiddenAct, "silu")
        XCTAssertTrue(config.normTopkProb)
        XCTAssertFalse(config.sandwichNorm)
        XCTAssertNil(config.maxPositionEmbeddings)
        XCTAssertNil(config.routedExperts)
        XCTAssertEqual(config.architectures, [])
        XCTAssertNoThrow(try config.validateModelConfiguration())
        // Missing MoE and attention dimensions are rejected before any model is built.
        XCTAssertThrowsError(try config.validateForForward())
    }

    func testDecodesPublishedCheckpointConfig() throws {
        let config = try decode(publishedConfigJSON)

        XCTAssertEqual(config.hiddenSize, 7680)
        XCTAssertEqual(config.hiddenLayers, 61)
        XCTAssertEqual(config.attentionHeads, 128)
        XCTAssertEqual(config.vocabularySize, 153_600)
        XCTAssertEqual(config.intermediateSize, 18_432)
        XCTAssertEqual(config.moeIntermediateSize, 2048)
        XCTAssertEqual(config.routedExperts, 256)
        XCTAssertEqual(config.sharedExperts, 1)
        XCTAssertEqual(config.expertsPerToken, 8)
        XCTAssertEqual(config.firstDenseLayers, 3)
        XCTAssertEqual(config.routedScalingFactor, 2.5)
        XCTAssertEqual(config.attentionQLoraDim, 1536)
        XCTAssertEqual(config.attentionKVLoraDim, 512)
        XCTAssertEqual(config.attentionQKDim, 128)
        XCTAssertEqual(config.attentionQKRopeDim, 64)
        XCTAssertEqual(config.attentionVDim, 128)
        XCTAssertEqual(config.ropeTheta, 25_600_000, accuracy: 1)
        XCTAssertTrue(config.sandwichNorm)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertNoThrow(try config.validateForForward())
    }

    func testValidationAcceptsFixtureShape() throws {
        XCTAssertNoThrow(try decode(json).validateModelConfiguration())
    }

    func testValidationRejectsWrongModelType() throws {
        let text = json.replacingOccurrences(
            of: "\"model_type\": \"pangu_ultra_moe\"", with: "\"model_type\": \"pangu_embedded\"")
        XCTAssertThrowsError(try decode(text).validateModelConfiguration())
    }

    func testValidationRejectsWrongArchitecture() throws {
        let text = json.replacingOccurrences(
            of: "PanguUltraMoEForCausalLM", with: "SomeOtherForCausalLM")
        XCTAssertThrowsError(try decode(text).validateModelConfiguration())
    }

    func testValidationRejectsHeadsNotDivisibleByKVHeads() throws {
        let text = json.replacingOccurrences(
            of: "\"num_key_value_heads\": 2,", with: "\"num_key_value_heads\": 3,")
        XCTAssertThrowsError(try decode(text).validateModelConfiguration())
    }

    func testDecodingFailsWithoutHiddenSize() {
        let text = """
            {
                "model_type": "pangu_ultra_moe",
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "vocab_size": 100
            }
            """
        XCTAssertThrowsError(try decode(text))
    }
}
