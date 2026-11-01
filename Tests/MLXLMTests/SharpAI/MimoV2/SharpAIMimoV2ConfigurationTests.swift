// SharpAI fork overlay tests for the mimo_v2 config. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMimoV2ConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAIMimoV2Configuration {
        try JSONDecoder.json5().decode(SharpAIMimoV2Configuration.self, from: Data(json.utf8))
    }

    func testDecodesFixtureFields() throws {
        let config = try decode(SharpAIMimoV2TestFixture.tinyJSON())

        XCTAssertEqual(config.modelType, "mimo_v2")
        XCTAssertEqual(config.hiddenLayers, 4)
        XCTAssertEqual(config.headDim, 8)
        XCTAssertEqual(config.vHeadDim, 8)
        XCTAssertEqual(config.swaVHeadDim, 8)
        XCTAssertEqual(config.hybridLayerPattern, [1, 0, 1, 0])
        XCTAssertEqual(config.moeLayerFreq, [0, 1, 1, 1])
        XCTAssertEqual(config.attentionValueScale ?? 0, 0.5, accuracy: 1e-6)
        XCTAssertEqual(config.attentionProjectionLayout, "fused_qkv")
        XCTAssertTrue(config.addSwaAttentionSinkBias)
        XCTAssertEqual(config.numNextnPredictLayers, 1)
    }

    func testDecodesBooleanMoeLayerFreq() throws {
        let json = SharpAIMimoV2TestFixture.tinyJSON().replacingOccurrences(
            of: "[0, 1, 1, 1]", with: "[false, true, true, true]")
        let config = try decode(json)

        XCTAssertEqual(config.moeLayerFreq, [0, 1, 1, 1])
    }

    func testScalarMoeLayerFreqMeansAllDense() throws {
        let json = SharpAIMimoV2TestFixture.tinyJSON().replacingOccurrences(
            of: "\"moe_layer_freq\": [0, 1, 1, 1]", with: "\"moe_layer_freq\": 0")
        let config = try decode(json)

        XCTAssertTrue(config.moeLayerFreq.isEmpty)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testFixturePassesValidation() throws {
        let config = try decode(SharpAIMimoV2TestFixture.tinyJSON())

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testSharedExpertsAreRejected() throws {
        let config = try decode(SharpAIMimoV2TestFixture.tinyJSON(sharedExperts: 1))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIMimoV2Error.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testOmniTowersAreRejectedAtDecode() {
        XCTAssertThrowsError(
            try decode(SharpAIMimoV2TestFixture.tinyJSON(omniTowers: true))
        ) { error in
            guard case SharpAIMimoV2Error.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testMismatchedHybridPatternFailsValidation() throws {
        let json = SharpAIMimoV2TestFixture.tinyJSON().replacingOccurrences(
            of: "[1, 0, 1, 0]", with: "[1, 0, 1]")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testMissingRequiredFieldFailsDecode() {
        XCTAssertThrowsError(try decode("{ \"hidden_size\": 32 }"))
    }

    /// configuration_mimo_v2.py: sliding-window head counts and head dim fall back to the
    /// full-attention values when the swa_* keys are absent.
    func testSWAGeometryFallsBackToFullAttentionWhenUnset() throws {
        let json = SharpAIMimoV2TestFixture.tinyJSON()
            .replacingOccurrences(of: "\"swa_num_attention_heads\": 4,", with: "")
            .replacingOccurrences(of: "\"swa_num_key_value_heads\": 2,", with: "")
            .replacingOccurrences(of: "\"swa_head_dim\": 8,", with: "")
        let config = try decode(json)

        XCTAssertEqual(config.swaAttentionHeads, config.attentionHeads)
        XCTAssertEqual(config.swaKvHeads, config.kvHeads)
        XCTAssertEqual(config.swaHeadDim, config.headDim)
        XCTAssertEqual(config.swaVHeadDim, config.vHeadDim)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    /// configuration_mimo_v2.py: head_dim defaults to hidden_size // num_attention_heads when
    /// absent. The fixture has 32 / 4 = 8, so the decoded value must equal the fixture's head_dim.
    func testHeadDimDefaultsToHiddenSizeOverHeads() throws {
        let json = SharpAIMimoV2TestFixture.tinyJSON()
            .replacingOccurrences(of: "\"head_dim\": 8,", with: "")
        let config = try decode(json)

        XCTAssertEqual(config.headDim, 32 / 4)
        XCTAssertEqual(config.vHeadDim, config.headDim)
    }
}
