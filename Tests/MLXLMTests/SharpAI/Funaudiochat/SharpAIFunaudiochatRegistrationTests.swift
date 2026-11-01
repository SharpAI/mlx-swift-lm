// SharpAI fork overlay tests for funaudiochat creator registration. Not part of upstream.

import Foundation
import MLXVLM
import XCTest

final class SharpAIFunaudiochatRegistrationTests: XCTestCase {

    func testCreatorIsKeyedByModelType() {
        XCTAssertNotNil(SharpAIFunaudiochatOverlay.creators["funaudiochat"])
    }

    func testCreatorThrowsUntilModelClassExists() throws {
        let creator = try XCTUnwrap(SharpAIFunaudiochatOverlay.creators["funaudiochat"])
        let data = Data(#"{"model_type": "funaudiochat"}"#.utf8)

        XCTAssertThrowsError(try creator(data))
    }

    /// The public sources exist (HF FunAudioLLM/Fun-Audio-Chat-8B config.json and
    /// model.safetensors.index.json, vLLM vllm/model_executor/models/funaudiochat.py), so this is a
    /// real gap, not a skip: the fork has no model class, and the creator still throws.
    /// Once a model class is written, the creator must return a LanguageModel instead of throwing.
    func testTinyForwardPassKnownIssue() throws {
        let data = Data(#"{"model_type": "funaudiochat"}"#.utf8)

        try XCTExpectFailure(
            "KNOWN GAP: FunAudioChat model class not written. Still missing: (1) the continuous audio tower (conv1/conv2 + 32 encoder layers, d_model 1280, 20 heads, FFN 5120, sinusoidal positions, n_window chunking, ln_post, proj to 4096) from vLLM funaudiochat.py FunAudioChatAudioEncoder; (2) the discrete encoder FunAudioChatDiscreteEncoder (embed_tokens codebook 6565, group-mean by group_size 5, output_matching, continual_output_matching); (3) the Qwen3 text LM wiring (36 layers, d 4096) and scatter of audio features at audio_token_index 151669; (4) the 128-bin log-mel feature extraction. The audio_invert_tower / crq_transformer is not used by vLLM inference and is dropped on load."
        ) {
            let creator = try XCTUnwrap(SharpAIFunaudiochatOverlay.creators["funaudiochat"])
            XCTAssertNoThrow(try creator(data))
        }
    }
}
