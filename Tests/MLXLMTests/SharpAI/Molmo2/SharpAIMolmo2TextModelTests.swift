// SharpAI fork overlay tests for the molmo2 text decoder. Not part of upstream.
// The text_config below is copied from allenai/Molmo2-O-7B config.json (see
// scripts/conformance/offline/fixtures/molmo2/allenai__Molmo2-O-7B.json). The YaRN check uses the
// closed form of the transformers reference at the two ends of the frequency range.

import Foundation
@testable import MLXVLM
import XCTest

final class SharpAIMolmo2TextModelTests: XCTestCase {

    private let realTextConfigJSON = """
        {
            "additional_vocab_size": 128,
            "attention_dropout": 0.0,
            "embedding_dropout": 0.0,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "layer_norm_eps": 1e-06,
            "max_position_embeddings": 65536,
            "model_type": "molmo2_text",
            "norm_after": true,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "qk_norm_type": "olmo",
            "qkv_bias": false,
            "residual_dropout": 0.0,
            "rope_scaling": {
                "attention_factor": 1.2079441541679836,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 8.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "yarn"
            },
            "rope_scaling_layers": [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30],
            "rope_theta": 500000.0,
            "use_qk_norm": true,
            "vocab_size": 100278
        }
        """

    private func decodeRealTextConfig() throws -> SharpAIMolmo2TextArguments {
        try JSONDecoder().decode(
            SharpAIMolmo2TextArguments.self, from: Data(realTextConfigJSON.utf8))
    }

    func testDecodesRealMolmo2TextConfig() throws {
        let args = try decodeRealTextConfig()

        XCTAssertEqual(args.hiddenSize, 4096)
        XCTAssertEqual(args.layers, 32)
        XCTAssertEqual(args.attentionHeads, 32)
        XCTAssertEqual(args.kvHeads, 32)
        XCTAssertEqual(args.headDim, 128)
        XCTAssertEqual(args.vocabularySize, 100278)
        XCTAssertEqual(args.additionalVocabularySize, 128)
        XCTAssertFalse(args.qkvBias)
        XCTAssertTrue(args.normAfter)
        XCTAssertTrue(args.useQKNorm)
        XCTAssertEqual(args.qkNormType, "olmo")
        XCTAssertEqual(args.ropeTheta, 500_000)
        XCTAssertEqual(args.ropeScalingLayers?.count, 24)
        XCTAssertEqual(args.ropeScalingLayers?.last, 30)

        let scaling = try XCTUnwrap(args.ropeScaling)
        XCTAssertEqual(scaling.factor, 8)
        XCTAssertEqual(scaling.originalMaxPositionEmbeddings, 8192)
        XCTAssertEqual(scaling.betaFast, 32)
        XCTAssertEqual(scaling.betaSlow, 1)
        XCTAssertTrue(scaling.truncate)
    }

    func testYarnFrequencyEndpointsMatchClosedForm() throws {
        let args = try decodeRealTextConfig()
        let scaling = try XCTUnwrap(args.ropeScaling)
        let yarn = sharpAIMolmo2YarnInverseFrequencies(
            headDim: args.headDim, base: args.ropeTheta, scaling: scaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)

        XCTAssertEqual(yarn.inverseFrequencies.count, 64)
        // Index 0 is below the correction range, so it keeps the unscaled frequency 1.
        XCTAssertEqual(yarn.inverseFrequencies[0], 1, accuracy: 1e-6)
        // Index 63 is above the range, so it takes the interpolated frequency theta^(-126/128) / 8.
        let expectedLast = 1 / (8 * pow(500_000.0, 126.0 / 128.0))
        XCTAssertEqual(Double(yarn.inverseFrequencies[63]), expectedLast, accuracy: expectedLast * 1e-5)
        // attention_factor is given in the config and is passed through unchanged.
        XCTAssertEqual(Double(yarn.attentionFactor), 1.2079441541679836, accuracy: 1e-6)
    }

    func testRejectsUnsupportedRopeType() {
        let json = realTextConfigJSON.replacingOccurrences(
            of: "\"rope_type\": \"yarn\"", with: "\"rope_type\": \"linear\"")

        XCTAssertThrowsError(
            try JSONDecoder().decode(SharpAIMolmo2TextArguments.self, from: Data(json.utf8)))
    }
}
