// SharpAI fork overlay tests for the dots3_note creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIDots3NoteRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "dots3_note",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "vocab_size": 512,
            "layer_types": ["full_attention", "sliding_attention"],
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "q_lora_rank": 16,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 8,
            "v_head_dim": 8,
            "rope_theta": 10000.0,
            "attention_gate_type": "headwise",
            "sliding_window_size": 513,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 4,
            "swa_q_lora_rank": 16,
            "swa_kv_lora_rank": 16,
            "swa_qk_nope_head_dim": 8,
            "swa_qk_rope_head_dim": 8,
            "swa_v_head_dim": 8,
            "swa_rope_theta": 10000.0,
            "swa_attention_gate_type": "headwise",
            "moe_intermediate_size": 32,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2
        }
        """

    private func decode(_ json: String) throws -> SharpAIDots3NoteConfiguration {
        try JSONDecoder.json5().decode(SharpAIDots3NoteConfiguration.self, from: Data(json.utf8))
    }

    private func weights(_ keys: [String]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: keys.map { ($0, MLXArray([Float(1)])) })
    }

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIDots3NoteOverlay.creators["dots3_note"])
    }

    func testMTPTypeIsNotRegistered() {
        XCTAssertNil(SharpAIDots3NoteOverlay.creators["dots3_note_mtp"])
    }

    func testValidConfigFailsUntilForwardIsImplemented() throws {
        guard let creator = SharpAIDots3NoteOverlay.creators["dots3_note"] else {
            XCTFail("dots3_note creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIDots3NoteError.forwardNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "dots3_note")
        }
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIDots3NoteOverlay.creators["dots3_note"] else {
            XCTFail("dots3_note creator missing")
            return
        }
        let json = """
            { "hidden_size": 64 }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            if case SharpAIDots3NoteError.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }

    /// Was a skip. The checkpoint index has `self_attn.indexer.*` keys on layer 0 (full_attention),
    /// and vLLM's Dots3NoteModel._adapt_weights drops `.indexer.` weights on sliding_attention
    /// layers. Only the key layout is checked here. Top-k indexer math is not ported.
    func testDSAIndexerKeysKeptOnFullAttentionOnly() throws {
        let config = try decode(validJSON)
        let out = SharpAIDots3NoteSanitizer.sanitize(
            weights([
                "model.layers.0.self_attn.indexer.wk.weight",
                "model.layers.1.self_attn.indexer.wk.weight",
            ]),
            configuration: config)

        XCTAssertNotNil(out["model.layers.0.self_attn.indexer.wk.weight"])
        XCTAssertNil(out["model.layers.1.self_attn.indexer.wk.weight"])
    }

    /// Was a skip. vLLM builds `g_proj` on both attention branches when attention_gate_type is
    /// headwise. The gate math is not ported; this checks the gate type per branch and that the
    /// `g_proj` keys survive sanitization on both layer types.
    func testHeadwiseGateTypeAndGateKeysAreKept() throws {
        let config = try decode(validJSON)
        let full = try XCTUnwrap(config.attentionGeometry(at: 0))
        let sliding = try XCTUnwrap(config.attentionGeometry(at: 1))
        XCTAssertEqual(full.gateType, "headwise")
        XCTAssertEqual(sliding.gateType, "headwise")

        let out = SharpAIDots3NoteSanitizer.sanitize(
            weights([
                "model.layers.0.self_attn.g_proj.weight",
                "model.layers.1.self_attn.g_proj.weight",
            ]),
            configuration: config)
        XCTAssertEqual(
            Set(out.keys),
            [
                "model.layers.0.self_attn.g_proj.weight",
                "model.layers.1.self_attn.g_proj.weight",
            ])
    }

    /// Was a skip. Tower keys are kept by the sanitizer. The checkpoint index uses the
    /// `vision_encoder.` and `audio_encoder.` prefixes, and both are tower segments. The
    /// audio layer key below is a synthetic shape; the real one is
    /// `audio_encoder.dots_encoder.speech_encoder.layers.N.*`. Vision and audio towers are not
    /// built, so the creator still throws. That gap is recorded as an expected failure.
    func testVisionAndAudioTowerKeysAreKeptAndTowersAreKnownGap() throws {
        let config = try decode(validJSON)
        let out = SharpAIDots3NoteSanitizer.sanitize(
            weights([
                "audio_encoder.audio_adapter.proj.0.weight",
                "audio_encoder.layers.2.self_attn.q_proj.weight",
                "vision_encoder.blocks.0.attn.qkv.weight",
            ]),
            configuration: config)
        XCTAssertNotNil(out["audio_encoder.audio_adapter.proj.0.weight"])
        XCTAssertNotNil(out["audio_encoder.layers.2.self_attn.q_proj.weight"])
        XCTAssertNotNil(out["vision_encoder.blocks.0.attn.qkv.weight"])

        let creator = try XCTUnwrap(SharpAIDots3NoteOverlay.creators["dots3_note"])
        try XCTExpectFailure(
            "KNOWN GAP: vision tower (42 blocks, pyramid MoE from vLLM nvidia/vision.py, "
                + "vision_moe.py, vision_attention.py) and audio encoder (vLLM nvidia/audio.py, "
                + "audio_encoder.py) are not ported to Swift; the creator still throws.") {
            XCTAssertNoThrow(try creator(Data(self.validJSON.utf8)))
        }
    }

    /// Was a skip. No dots3_note model class exists. Forward needs MLA, the DSA indexer, the
    /// headwise gate and MoE. vLLM has those modules, but no Swift port exists here. The
    /// creator-level assertion that should hold once forward exists is recorded as an expected
    /// failure.
    func testTinyForwardPassSkipped() throws {
        let creator = try XCTUnwrap(SharpAIDots3NoteOverlay.creators["dots3_note"])
        try XCTExpectFailure(
            "KNOWN GAP: dots3_note language model class is not written in Swift. The reference is "
                + "vLLM vllm/models/dots3_note/nvidia/model.py and attention.py (MLA full and SWA "
                + "attention, DSA indexer, headwise g_proj gate, sigmoid noaux_tc MoE). Not ported yet.") {
            XCTAssertNoThrow(try creator(Data(self.validJSON.utf8)))
        }
    }
}
