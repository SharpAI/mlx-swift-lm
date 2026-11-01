// SharpAI fork overlay tests for the step3p5 configuration. Not part of upstream.
// Inline JSON only. No checkpoint, no network.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIStep3p5ConfigurationTests: XCTestCase {

    private func decode(_ overrides: [String: Any] = [:]) throws -> SharpAIStep3p5Configuration {
        try JSONDecoder.json5().decode(
            SharpAIStep3p5Configuration.self,
            from: Data(SharpAIStep3p5TestFixture.json(overrides).utf8))
    }

    func testDecodesTopLevelFields() throws {
        let config = try decode()

        XCTAssertEqual(config.modelType, "step3p5")
        XCTAssertEqual(config.architectures, ["Step3p5ForCausalLM"])
        XCTAssertEqual(config.hiddenSize, 32)
        XCTAssertEqual(config.hiddenLayers, 3)
        XCTAssertEqual(config.attentionGroups, 2)
        XCTAssertEqual(config.headDim, 8)
        XCTAssertEqual(config.slidingWindow, 4)
        XCTAssertEqual(config.maxPositionEmbeddings, 64)
    }

    func testDefaultsForAbsentKeys() throws {
        let config = try decode([
            "rope_scaling": NSNull(),
            "head_dim": NSNull(),
            "use_head_wise_attn_gate": NSNull(),
            "moe_layers_enum": NSNull(),
            "rope_theta": NSNull(),
        ])

        XCTAssertEqual(config.headDim, 32 / 4)
        XCTAssertFalse(config.useHeadWiseAttnGate)
        XCTAssertEqual(config.ropeBase(layer: 0), 10000)
        XCTAssertNil(config.ropeScaling)
    }

    func testQueryHeadsVaryPerLayer() throws {
        let config = try decode()

        XCTAssertEqual(config.queryHeads(layer: 0), 4)
        XCTAssertEqual(config.queryHeads(layer: 1), 6)
        XCTAssertEqual(config.queryHeads(layer: 2), 4)
    }

    func testNumSlidingAttentionHeadsWinsOverOtherSetting() throws {
        let config = try decode(["num_sliding_attention_heads": 5])

        XCTAssertEqual(config.queryHeads(layer: 1), 5)
        XCTAssertEqual(config.queryHeads(layer: 0), 4)
    }

    func testMoELayersFromEnumIndices() throws {
        let config = try decode()

        XCTAssertFalse(config.isMoELayer(0))
        XCTAssertTrue(config.isMoELayer(1))
        XCTAssertTrue(config.isMoELayer(2))
    }

    func testMLPLayerTypesOverrideEnum() throws {
        let config = try decode(["mlp_layer_types": ["dense", "dense", "sparse"]])

        XCTAssertFalse(config.isMoELayer(1))
        XCTAssertTrue(config.isMoELayer(2))
    }

    func testRopePerLayerTypeMatchesVLLM() throws {
        let config = try decode()

        // Full layers get the llama3 scaling. Sliding layers fall back to default rope.
        XCTAssertEqual(config.ropeType(layer: 0), "llama3")
        XCTAssertEqual(config.ropeType(layer: 1), "default")
        XCTAssertEqual(config.ropeBase(layer: 2), 5_000_000)
        XCTAssertEqual(config.rotaryDimensions(layer: 0), 4)
        XCTAssertEqual(config.rotaryDimensions(layer: 1), 8)
    }

    func testSwigluLimitsIndexByLayer() throws {
        let config = try decode()

        XCTAssertEqual(config.routedSwigluLimit(layer: 1), 7)
        XCTAssertEqual(config.sharedSwigluLimit(layer: 1), 16)
        XCTAssertEqual(config.sharedSwigluLimit(layer: 0), 0)
        XCTAssertEqual(config.routedSwigluLimit(layer: 99), 0)
    }

    func testTinyConfigValidates() throws {
        XCTAssertNoThrow(try decode().validateModelConfiguration())
    }

    func testMTPOnlyArchitectureIsRejected() throws {
        let config = try decode(["architectures": ["Step3p5MTP"]])

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIStep3p5Error.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testUnknownAttentionImplementationIsRejected() throws {
        let config = try decode(["att_impl_type": "MLA"])

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testNonSigmoidRouterIsRejected() throws {
        let config = try decode(["moe_router_activation": "softmax"])

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRouterBiasFlagMustBeTrue() throws {
        let config = try decode(["use_moe_router_bias": false])

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testUnsupportedPartialRotaryFactorIsRejected() throws {
        let config = try decode(["partial_rotary_factors": [0.25, 1.0, 0.5, 0.5]])

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testSlidingLayersNeedAWindow() throws {
        let config = try decode(["sliding_window": NSNull()])

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testTopKMustNotExceedExpertCount() throws {
        let config = try decode(["moe_top_k": 5])

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
