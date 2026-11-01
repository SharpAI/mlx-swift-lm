// SharpAI fork overlay tests for the minimax_m3_mtp creator. Not part of upstream.
// The creator validates the config and the MTP head shape, then throws: it cannot return a
// LanguageModel yet, because the head needs the target's embeddings, hidden states and lm_head.
// The head's real forward pass is tested in SharpAIMinimaxM3MtpHeadTests.swift.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

private enum SharpAIMinimaxM3MtpRegistrationFixtures {
    /// Only the MTP identity keys. The text fields are absent.
    static let minimalJSON = """
        {
          "model_type": "minimax_m3_mtp",
          "architectures": ["MiniMaxM3MTP"]
        }
        """

    /// Complete tiny config: M3 text fields plus `num_mtp_modules`. Last decoder index is MoE and sparse.
    static let tinyJSON = """
        {
          "model_type": "minimax_m3_mtp",
          "architectures": ["MiniMaxM3MTP"],
          "hidden_size": 32,
          "intermediate_size": 16,
          "num_hidden_layers": 4,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "vocab_size": 64,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000,
          "rotary_dim": 4,
          "num_local_experts": 4,
          "num_experts_per_tok": 2,
          "n_shared_experts": 1,
          "shared_intermediate_size": 16,
          "num_mtp_modules": 2,
          "sparse_attention_config": {
            "use_sparse_attention": true,
            "sparse_index_dim": 8,
            "sparse_num_index_heads": 2,
            "sparse_topk_blocks": 2,
            "sparse_block_size": 4,
            "sparse_local_block": 1,
            "sparse_score_type": "max"
          }
        }
        """
}

final class SharpAIMinimaxM3MtpRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMinimaxM3MtpOverlay.creators["minimax_m3_mtp"])
    }

    func testWrongArchitectureFailsAtValidation() throws {
        guard let creator = SharpAIMinimaxM3MtpOverlay.creators["minimax_m3_mtp"] else {
            XCTFail("minimax_m3_mtp creator missing")
            return
        }
        let json = SharpAIMinimaxM3MtpRegistrationFixtures.tinyJSON.replacingOccurrences(
            of: "MiniMaxM3MTP", with: "LlamaForCausalLM")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard let typed = error as? SharpAIMinimaxM3MtpError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testMissingTextFieldsFailAsInvalidConfiguration() throws {
        guard let creator = SharpAIMinimaxM3MtpOverlay.creators["minimax_m3_mtp"] else {
            XCTFail("minimax_m3_mtp creator missing")
            return
        }

        XCTAssertThrowsError(
            try creator(Data(SharpAIMinimaxM3MtpRegistrationFixtures.minimalJSON.utf8))
        ) { error in
            guard let typed = error as? SharpAIMinimaxM3MtpError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testValidConfigStopsAtConstructionAsUnsupported() throws {
        guard let creator = SharpAIMinimaxM3MtpOverlay.creators["minimax_m3_mtp"] else {
            XCTFail("minimax_m3_mtp creator missing")
            return
        }

        XCTAssertThrowsError(
            try creator(Data(SharpAIMinimaxM3MtpRegistrationFixtures.tinyJSON.utf8))
        ) { error in
            guard let typed = error as? SharpAIMinimaxM3MtpError,
                case .unsupported = typed
            else {
                XCTFail("expected unsupported, got \(error)")
                return
            }
        }
    }

    // Was an XCTSkip, then a KNOWN GAP over the stub creator. The head is now implemented and its
    // forward pass is tested in SharpAIMinimaxM3MtpHeadTests. The creator still returns no model,
    // because a LanguageModel needs the target's embeddings, hidden states and lm_head, and no
    // target-side speculative path or public M3 checkpoint with mtp.* weights exists here.
    // When the creator returns a model, this assertion passes and XCTExpectFailure reports an
    // unexpected success, which fails the test.
    func testTinyForwardPassWithRandomWeights() {
        XCTExpectFailure(
            "KNOWN GAP: minimax_m3_mtp creator returns no LanguageModel. The MTP head SharpAIMinimaxM3MtpHead is implemented, but a LanguageModel needs the target's embeddings, hidden states and lm_head: the target-side speculative path is not in the repo, and the public MiniMax-M3 checkpoint has no mtp.* tensors to load."
        ) {
            let creator = SharpAIMinimaxM3MtpOverlay.creators["minimax_m3_mtp"]
            XCTAssertNotNil(creator, "minimax_m3_mtp creator missing")
            let model = try? creator?(Data(SharpAIMinimaxM3MtpRegistrationFixtures.tinyJSON.utf8))
            XCTAssertNotNil(model, "tiny minimax_m3_mtp model should construct with random weights")
        }
    }
}
