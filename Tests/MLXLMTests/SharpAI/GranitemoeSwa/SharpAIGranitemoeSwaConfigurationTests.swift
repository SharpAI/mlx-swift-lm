// SharpAI fork overlay tests for granitemoe_swa config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIGranitemoeSwaConfigurationTests: XCTestCase {

    func testDecodesAllFieldsFromInlineJSON() throws {
        let config = try SharpAIGranitemoeSwaTestFixtures.configuration()

        XCTAssertEqual(config.modelType, "granitemoe_swa")
        XCTAssertEqual(config.vocabularySize, 64)
        XCTAssertEqual(config.hiddenSize, 16)
        XCTAssertEqual(config.intermediateSize, 8)
        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertEqual(config.attentionHeads, 4)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.numLocalExperts, 4)
        XCTAssertEqual(config.numExpertsPerToken, 2)
        XCTAssertEqual(config.sharedIntermediateSize, 8)
        XCTAssertEqual(config.maxPositionEmbeddings, 128)
        XCTAssertEqual(config.ropeTheta, 10_000.0)
        XCTAssertEqual(config.layerTypes, ["sliding_attention", "full_attention"])
        XCTAssertEqual(config.layerRopeTheta, [0.0, 10_000.0])
        XCTAssertEqual(config.slidingWindow, 2)
        XCTAssertEqual(config.attentionMultiplier, 0.25)
        XCTAssertEqual(config.embeddingMultiplier, 2.0)
        XCTAssertEqual(config.residualMultiplier, 0.5)
        XCTAssertEqual(config.logitsScaling, 2.0)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testOptionalFieldsUseDefaults() throws {
        // UNVERIFIED: default values are assumptions from HF Granite, pinned here so a change shows.
        let json = """
            {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_local_experts": 4,
                "num_experts_per_tok": 2
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIGranitemoeSwaConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "granitemoe_swa")
        XCTAssertEqual(config.kvHeads, 4)
        XCTAssertNil(config.sharedIntermediateSize)
        XCTAssertNil(config.maxPositionEmbeddings)
        XCTAssertEqual(config.rmsNormEps, 1e-6)
        XCTAssertEqual(config.ropeTheta, 10_000.0)
        XCTAssertNil(config.ropeScaling)
        XCTAssertNil(config.layerRopeTheta)
        XCTAssertNil(config.layerTypes)
        XCTAssertNil(config.slidingWindow)
        XCTAssertNil(config.attentionSinks)
        XCTAssertNil(config.attentionMultiplier)
        XCTAssertEqual(config.embeddingMultiplier, 1.0)
        XCTAssertEqual(config.residualMultiplier, 1.0)
        XCTAssertEqual(config.logitsScaling, 1.0)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.resolvedLayerTypes, ["full_attention", "full_attention"])
        XCTAssertFalse(config.usesAttentionSinks)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testSinksFollowLayerTypesUnlessOverridden() throws {
        let withLayerTypes = try SharpAIGranitemoeSwaTestFixtures.configuration()
        XCTAssertTrue(withLayerTypes.usesAttentionSinks)

        let explicitOff = try SharpAIGranitemoeSwaTestFixtures.configuration(
            overriding: ["attention_sinks": false])
        XCTAssertFalse(explicitOff.usesAttentionSinks)

        let noLayerTypes = try SharpAIGranitemoeSwaTestFixtures.configuration(
            overriding: ["layer_types": NSNull(), "sliding_window": NSNull(),
                "layer_rope_theta": NSNull()])
        XCTAssertFalse(noLayerTypes.usesAttentionSinks)
    }

    func testRejectsSlidingLayerWithoutWindow() throws {
        let config = try SharpAIGranitemoeSwaTestFixtures.configuration(
            overriding: ["sliding_window": NSNull()])

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsLayerTypeCountMismatch() throws {
        let config = try SharpAIGranitemoeSwaTestFixtures.configuration(
            overriding: ["layer_types": ["sliding_attention"]])

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsMoreActiveExpertsThanExperts() throws {
        let config = try SharpAIGranitemoeSwaTestFixtures.configuration(
            overriding: ["num_experts_per_tok": 5])

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsNegativeLayerRopeTheta() throws {
        let config = try SharpAIGranitemoeSwaTestFixtures.configuration(
            overriding: ["layer_rope_theta": [-1.0, 10_000.0]])

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
