// SharpAI fork overlay tests for step1 config decoding. Not part of upstream.
// The real-config test uses the values in stepfun-ai/Step-Audio-EditX config.json. The other
// fixtures are synthetic.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIStep1ConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAIStep1Configuration {
        try JSONDecoder.json5().decode(
            SharpAIStep1Configuration.self, from: Data(json.utf8))
    }

    func testDecodesExplicitFields() throws {
        let config = try decode(
            """
            {
                "model_type": "step1",
                "architectures": ["Step1ForCausalLM"],
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

        XCTAssertEqual(config.modelType, "step1")
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

        XCTAssertEqual(config.modelType, "step1")
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

    func testDecodesStepAudioEditXConfigWithAttentionGroups() throws {
        // Values from stepfun-ai/Step-Audio-EditX config.json. KV heads use num_attention_groups.
        let config = try decode(
            """
            {
                "architectures": ["Step1ForCausalLM"],
                "model_type": "step1",
                "bos_token_id": 1,
                "pad_token_id": 0,
                "eos_token_id": 3,
                "hidden_size": 3072,
                "intermediate_size": 8192,
                "num_attention_heads": 48,
                "num_attention_groups": 4,
                "num_hidden_layers": 32,
                "max_seq_len": 32768,
                "vocab_size": 74752,
                "rms_norm_eps": 1e-05,
                "tie_word_embeddings": false,
                "torch_dtype": "bfloat16",
                "use_cache": true
            }
            """)

        XCTAssertEqual(config.hiddenSize, 3072)
        XCTAssertEqual(config.attentionHeads, 48)
        XCTAssertEqual(config.kvHeads, 4)
        XCTAssertEqual(config.headDim, 64)
        XCTAssertEqual(config.hiddenLayers, 32)
        XCTAssertEqual(config.vocabularySize, 74752)
        XCTAssertEqual(config.rmsNormEps, 1e-5, accuracy: 1e-12)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertNoThrow(try config.validateModelConfiguration())
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
            XCTAssertTrue(error is SharpAIStep1Error)
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
            XCTAssertTrue(error is SharpAIStep1Error)
        }
    }
}
