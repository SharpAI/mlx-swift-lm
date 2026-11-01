// SharpAI fork overlay tests for olmo_hybrid config decoding and validation. Not part of upstream.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIOlmoHybridConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAIOlmoHybridConfiguration {
        try JSONDecoder.json5().decode(SharpAIOlmoHybridConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesReleasedStyleConfig() throws {
        let config = try decode(
            """
            {
              "model_type": "olmo_hybrid",
              "hidden_size": 3840,
              "num_hidden_layers": 8,
              "num_attention_heads": 30,
              "num_key_value_heads": 30,
              "intermediate_size": 11008,
              "vocab_size": 100352,
              "max_position_embeddings": 65536,
              "rms_norm_eps": 1e-6,
              "rope_parameters": null,
              "linear_num_key_heads": 30,
              "linear_num_value_heads": 30,
              "linear_key_head_dim": 96,
              "linear_value_head_dim": 192,
              "linear_conv_kernel_dim": 4,
              "linear_allow_neg_eigval": true
            }
            """)
        try config.validateModelConfiguration()

        XCTAssertEqual(config.modelType, "olmo_hybrid")
        XCTAssertEqual(config.hiddenSize, 3840)
        XCTAssertEqual(config.hiddenLayers, 8)
        XCTAssertEqual(config.attentionHeads, 30)
        XCTAssertEqual(config.kvHeads, 30)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.layerTypes[3], "full_attention")
        XCTAssertEqual(config.layerTypes[7], "full_attention")
        XCTAssertEqual(config.layerTypes[0], "linear_attention")
        XCTAssertNil(config.ropeTheta)
        XCTAssertEqual(config.linearKeyHeadDim, 96)
        XCTAssertEqual(config.linearValueHeadDim, 192)
        XCTAssertTrue(config.linearAllowNegEigval)
    }

    func testDefaultsFollowOlmoHybridConfig() throws {
        let config = try decode(
            """
            { "hidden_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2 }
            """)
        try config.validateModelConfiguration()

        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.headDim, 8)
        XCTAssertEqual(config.intermediateSize, 11008)
        XCTAssertEqual(config.hiddenAct, "silu")
        XCTAssertFalse(config.attentionBias)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertEqual(config.linearNumKeyHeads, 2)
        XCTAssertEqual(config.linearNumValueHeads, 2)
        XCTAssertEqual(config.linearKeyHeadDim, 6)
        XCTAssertEqual(config.linearValueHeadDim, 12)
        XCTAssertEqual(config.linearConvKernelDim, 4)
        XCTAssertTrue(config.linearAllowNegEigval)
        XCTAssertEqual(
            config.layerTypes,
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"])
    }

    func testDefaultLayerTypesForceFullAttentionOnLastLayer() throws {
        let config = try decode(
            """
            { "hidden_size": 16, "num_hidden_layers": 3, "num_attention_heads": 2 }
            """)
        XCTAssertEqual(
            config.layerTypes, ["linear_attention", "linear_attention", "full_attention"])
    }

    func testRopeThetaFromRopeParameters() throws {
        let config = try decode(
            """
            {
              "hidden_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2,
              "rope_parameters": { "rope_type": "default", "rope_theta": 500000.0 }
            }
            """)
        try config.validateModelConfiguration()
        XCTAssertEqual(config.ropeTheta, 500_000)
    }

    func testAllowNegEigvalCanBeDisabled() throws {
        let config = try decode(
            """
            { "hidden_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2,
              "linear_allow_neg_eigval": false }
            """)
        XCTAssertFalse(config.linearAllowNegEigval)
    }

    func testRejectsModelWithoutLinearAttention() throws {
        let config = try decode(
            """
            { "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
              "layer_types": ["full_attention", "full_attention"] }
            """)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsModelWithoutFullAttention() throws {
        let config = try decode(
            """
            { "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
              "layer_types": ["linear_attention", "linear_attention"] }
            """)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsNonDefaultRopeType() throws {
        let config = try decode(
            """
            { "hidden_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2,
              "rope_parameters": { "rope_type": "yarn", "rope_theta": 10000.0 } }
            """)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsNonSiluActivation() throws {
        let config = try decode(
            """
            { "hidden_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2,
              "hidden_act": "gelu" }
            """)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsValueHeadsNotMultipleOfKeyHeads() throws {
        let config = try decode(
            """
            { "hidden_size": 16, "num_hidden_layers": 4, "num_attention_heads": 2,
              "linear_num_key_heads": 2, "linear_num_value_heads": 3 }
            """)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
