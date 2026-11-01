// SharpAI fork overlay tests for exaone_moe config decoding and validation. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIExaoneMoeConfigurationTests: XCTestCase {

    private let mixedJSON = """
        {
            "model_type": "exaone_moe",
            "vocab_size": 64,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 4,
            "mlp_layer_types": ["dense", "sparse"],
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "num_shared_experts": 1,
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": true,
            "n_group": 2,
            "topk_group": 1,
            "swiglu_limits": [2.0, 2.0],
            "num_nextn_predict_layers": 1,
            "mtp_layer_types": ["full_attention"]
        }
        """

    private func decode(_ json: String) throws -> SharpAIExaoneMoeConfiguration {
        try JSONDecoder.json5().decode(SharpAIExaoneMoeConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesAllFields() throws {
        let config = try decode(mixedJSON)

        XCTAssertEqual(config.modelType, "exaone_moe")
        XCTAssertEqual(config.vocabularySize, 64)
        XCTAssertEqual(config.hiddenSize, 16)
        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertEqual(config.attentionHeads, 2)
        XCTAssertEqual(config.kvHeads, 1)
        XCTAssertEqual(config.headDim, 8)
        XCTAssertEqual(config.ropeTheta, 10000.0)
        XCTAssertEqual(config.slidingWindow, 4)
        XCTAssertEqual(config.numExperts, 4)
        XCTAssertEqual(config.numExpertsPerTok, 2)
        XCTAssertEqual(config.moeIntermediateSize, 8)
        XCTAssertEqual(config.numSharedExperts, 1)
        XCTAssertEqual(config.routedScalingFactor, 2.5)
        XCTAssertTrue(config.normTopkProb)
        XCTAssertEqual(config.nGroup, 2)
        XCTAssertEqual(config.topkGroup, 1)
        XCTAssertEqual(config.numNextnPredictLayers, 1)
    }

    func testDefaultsWhenOptionalFieldsAbsent() throws {
        let json = """
            {
                "vocab_size": 64,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 4
            }
            """
        let config = try decode(json)

        XCTAssertEqual(config.kvHeads, 4)
        XCTAssertEqual(config.headDim, 4)
        XCTAssertEqual(config.hiddenAct, "silu")
        XCTAssertEqual(config.layerTypes, ["full_attention", "full_attention"])
        XCTAssertEqual(config.numNextnPredictLayers, 1)
        XCTAssertNil(config.mlpLayerTypes)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testLayerHelpers() throws {
        let config = try decode(mixedJSON)

        XCTAssertEqual(config.windowSize(layer: 0, mtp: false), 4)
        XCTAssertNil(config.windowSize(layer: 1, mtp: false))

        // NoPE on the full-attention layer, RoPE on the sliding layer and on MTP.
        XCTAssertTrue(config.usesRope(layer: 0, mtp: false))
        XCTAssertFalse(config.usesRope(layer: 1, mtp: false))
        XCTAssertTrue(config.usesRope(layer: 0, mtp: true))

        XCTAssertEqual(config.swigluLimit(layer: 0), 2.0)
        XCTAssertEqual(config.swigluLimit(layer: 1), 2.0)

        XCTAssertFalse(config.isSparseLayer(0))
        XCTAssertTrue(config.isSparseLayer(1))
    }

    func testNonPositiveSwigluLimitMeansNoClamp() throws {
        let json = mixedJSON.replacingOccurrences(
            of: "\"swiglu_limits\": [2.0, 2.0]", with: "\"swiglu_limits\": [0.0, -1.0]")
        let config = try decode(json)

        XCTAssertNil(config.swigluLimit(layer: 0))
        XCTAssertNil(config.swigluLimit(layer: 1))
    }

    func testMixedConfigValidates() throws {
        XCTAssertNoThrow(try decode(mixedJSON).validateModelConfiguration())
    }

    func testRejectsNonSiluActivation() throws {
        let json = mixedJSON.replacingOccurrences(
            of: "\"num_nextn_predict_layers\": 1",
            with: "\"num_nextn_predict_layers\": 1, \"hidden_act\": \"gelu\"")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIExaoneMoeError.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testRejectsRoutedExpertsWithoutMlpLayerTypes() throws {
        let json = mixedJSON.replacingOccurrences(
            of: "\"mlp_layer_types\": [\"dense\", \"sparse\"],", with: "")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsSlidingLayerWithoutWindow() throws {
        let json = mixedJSON.replacingOccurrences(of: "\"sliding_window\": 4,", with: "")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testZeroAttentionHeadsWithoutHeadDimThrowsInsteadOfTrapping() throws {
        let json = mixedJSON
            .replacingOccurrences(of: "\"num_attention_heads\": 2,", with: "\"num_attention_heads\": 0,")
            .replacingOccurrences(of: "\"head_dim\": 8,", with: "")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsSingleExpertGroups() throws {
        let json = mixedJSON.replacingOccurrences(
            of: "\"n_group\": 2,", with: "\"n_group\": 4,"
        ).replacingOccurrences(of: "\"topk_group\": 1,", with: "\"topk_group\": 2,")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
