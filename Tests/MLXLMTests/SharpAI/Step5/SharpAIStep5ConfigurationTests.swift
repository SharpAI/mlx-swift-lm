// SharpAI fork overlay tests for step5 config decoding. Not part of upstream.
// Fixture values are synthetic, except where a test says it mirrors the official
// Step-5-Preview-BF16 text_config (stepfun-ai, config.json).

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIStep5ConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAIStep5Configuration {
        try JSONDecoder.json5().decode(
            SharpAIStep5Configuration.self, from: Data(json.utf8))
    }

    func testDecodesExplicitFields() throws {
        let config = try decode(
            """
            {
                "model_type": "step5",
                "architectures": ["Step5ForCausalLM"],
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "rope_theta": 500000.0,
                "max_position_embeddings": 128,
                "tie_word_embeddings": true,
                "hidden_act": "silu"
            }
            """)

        XCTAssertEqual(config.modelType, "step5")
        XCTAssertEqual(config.vocabularySize, 64)
        XCTAssertEqual(config.hiddenSize, 16)
        XCTAssertEqual(config.intermediateSize, 32)
        XCTAssertEqual(config.hiddenLayers, 3)
        XCTAssertEqual(config.attentionHeads, 4)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.headDim, 4)
        XCTAssertEqual(config.rmsNormEps, 1e-6, accuracy: 1e-12)
        XCTAssertEqual(config.ropeTheta, 500_000, accuracy: 1)
        XCTAssertEqual(config.maxPositionEmbeddings, 128)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.hiddenAct, "silu")
    }

    func testAppliesDefaultsWhenOptionalFieldsAbsent() throws {
        let config = try decode(
            """
            {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4
            }
            """)

        XCTAssertEqual(config.modelType, "step5")
        XCTAssertEqual(config.kvHeads, 4)
        XCTAssertEqual(config.rmsNormEps, 1e-5, accuracy: 1e-12)
        XCTAssertEqual(config.ropeTheta, 10_000, accuracy: 1e-3)
        XCTAssertNil(config.maxPositionEmbeddings)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertNil(config.hiddenAct)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingRequiredFieldThrows() {
        XCTAssertThrowsError(
            try decode(
                """
                {
                    "vocab_size": 64,
                    "hidden_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4
                }
                """))
    }

    func testRejectsHiddenSizeNotDivisibleByHeads() throws {
        let config = try decode(
            """
            {
                "vocab_size": 64,
                "hidden_size": 18,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4
            }
            """)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            XCTAssertTrue(error is SharpAIStep5Error)
        }
    }

    func testRejectsHeadsNotDivisibleByKVHeads() throws {
        let config = try decode(
            """
            {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 3
            }
            """)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            XCTAssertTrue(error is SharpAIStep5Error)
        }
    }

    /// Mirrors the official text_config field shapes: explicit head_dim 192 with 64 heads on
    /// hidden 4096 (so head_dim is not hidden / heads), num_attention_groups for the KV head
    /// count, a per-layer rope_theta list longer than num_hidden_layers (MTP entries), and
    /// moe_layers_enum as a comma string.
    func testDecodesOfficialStyleFields() throws {
        let config = try decode(
            """
            {
                "vocab_size": 128896,
                "hidden_size": 4096,
                "intermediate_size": 13824,
                "num_hidden_layers": 2,
                "num_attention_heads": 64,
                "num_attention_groups": 4,
                "head_dim": 192,
                "rope_theta": [10000.0, 10000000.0, 10000.0, 10000.0],
                "partial_rotary_factors": [1.0, 0.3333333333333333, 1.0, 1.0],
                "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention"],
                "sliding_window": 512,
                "moe_layers_enum": "1",
                "moe_num_experts": 352,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 1536,
                "share_expert_dim": 1536,
                "swiglu_limits": [7.0, 7.0],
                "swiglu_limits_shared": [7.0, 7.0],
                "rope_scaling": {"rope_type": "llama3", "factor": 1.0}
            }
            """)

        XCTAssertEqual(config.headDim, 192)
        XCTAssertEqual(config.kvHeads, 4)
        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertEqual(config.ropeThetaPerLayer, [10000, 10_000_000])
        XCTAssertEqual(config.layerTypes, ["sliding_attention", "full_attention"])
        XCTAssertEqual(config.moeLayers, [1])
        XCTAssertEqual(config.moeTopK, 8)
        XCTAssertEqual(config.rotaryDimensions(layer: 0), 192)
        XCTAssertEqual(config.rotaryDimensions(layer: 1), 64)
        XCTAssertEqual(config.slidingWindowFor(0), 512)
        XCTAssertNil(config.slidingWindowFor(1))
        XCTAssertNoThrow(try config.validateModelConfiguration())
        // Layer 1 is full attention with no DSA sparse config, so the reference path is absent
        // and the architecture check refuses it.
        XCTAssertThrowsError(try config.validateSupportedArchitecture()) { error in
            guard case SharpAIStep5Error.unsupported = error else {
                return XCTFail("expected unsupported, got \(error)")
            }
        }
    }

    /// Per-layer swiglu limit lists may carry trailing MTP entries beyond num_hidden_layers.
    func testTruncatesSwigluLimitsToHiddenLayers() throws {
        let config = try decode(
            """
            {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "swiglu_limits": [7.0, 6.0, 5.0],
                "swiglu_limits_shared": [4.0, 3.0, 2.0]
            }
            """)

        XCTAssertEqual(config.swigluLimits, [7.0, 6.0])
        XCTAssertEqual(config.swigluLimitsShared, [4.0, 3.0])
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }
}
