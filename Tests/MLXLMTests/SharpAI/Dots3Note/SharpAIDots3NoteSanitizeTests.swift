// SharpAI fork overlay tests for dots3_note checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIDots3NoteSanitizeTests: XCTestCase {

    private func configuration(tied: Bool) throws -> SharpAIDots3NoteConfiguration {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "vocab_size": 512,
                "tie_word_embeddings": \(tied),
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
        return try JSONDecoder.json5().decode(
            SharpAIDots3NoteConfiguration.self, from: Data(json.utf8))
    }

    private func weights(_ keys: [String]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: keys.map { ($0, MLXArray([Float(1)])) })
    }

    func testDropsLmHeadWhenEmbeddingsAreTied() throws {
        let config = try configuration(tied: true)
        let out = SharpAIDots3NoteSanitizer.sanitize(
            weights(["model.embed_tokens.weight", "lm_head.weight"]), configuration: config)

        XCTAssertNotNil(out["model.embed_tokens.weight"])
        XCTAssertNil(out["lm_head.weight"])
    }

    func testKeepsLmHeadWhenEmbeddingsAreNotTied() throws {
        let config = try configuration(tied: false)
        let out = SharpAIDots3NoteSanitizer.sanitize(
            weights(["lm_head.weight"]), configuration: config)

        XCTAssertNotNil(out["lm_head.weight"])
    }

    func testKeepsHFKeyNamesUnchanged() throws {
        let config = try configuration(tied: false)
        let keys = [
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.1.mlp.gate.weight",
        ]
        let out = SharpAIDots3NoteSanitizer.sanitize(weights(keys), configuration: config)

        XCTAssertEqual(Set(out.keys), Set(keys))
    }

    func testDropsTextLayerKeysPastHiddenLayers() throws {
        // Verified against: the checkpoint index has the MTP layer at `model.layers.46.*`, which is
        // num_hidden_layers, and vLLM nvidia/mtp.py starts MTP layers at mtp_start_layer_idx =
        // num_hidden_layers. This test uses the tiny config (num_hidden_layers = 2).
        let config = try configuration(tied: false)
        let out = SharpAIDots3NoteSanitizer.sanitize(
            weights([
                "model.layers.1.self_attn.q_a_proj.weight",
                "model.layers.2.self_attn.q_a_proj.weight",
                "model.layers.2.eh_proj.weight",
            ]),
            configuration: config)

        XCTAssertNotNil(out["model.layers.1.self_attn.q_a_proj.weight"])
        XCTAssertNil(out["model.layers.2.self_attn.q_a_proj.weight"])
        XCTAssertNil(out["model.layers.2.eh_proj.weight"])
    }

    func testTowerKeysAreNeverDroppedByLayerIndex() throws {
        // Synthetic guard: the vision_tower key shape is not from a checkpoint. The real vision
        // prefix is `vision_encoder.` (see testVisionEncoderPrefixKeysAreNeverDroppedByLayerIndex).
        let config = try configuration(tied: false)
        let key = "vision_tower.layers.5.attn.qkv.weight"
        let out = SharpAIDots3NoteSanitizer.sanitize(weights([key]), configuration: config)

        XCTAssertNotNil(out[key])
    }

    /// The published index uses `vision_encoder.*`. With the prefix in towerSegments, a
    /// `layers.N` segment under it is not dropped by num_hidden_layers. The `layers.5` key below is
    /// synthetic and only guards the segment check.
    func testVisionEncoderPrefixKeysAreNeverDroppedByLayerIndex() throws {
        let config = try configuration(tied: false)
        let key = "vision_encoder.layers.5.attn.qkv.weight"
        let out = SharpAIDots3NoteSanitizer.sanitize(weights([key]), configuration: config)

        XCTAssertNotNil(out[key])
    }

    func testLayerIndexParsesLayersSegment() {
        XCTAssertEqual(SharpAIDots3NoteSanitizer.layerIndex(of: "model.layers.17.mlp.weight"), 17)
        XCTAssertNil(SharpAIDots3NoteSanitizer.layerIndex(of: "model.embed_tokens.weight"))
    }

    /// Was a skip. The checkpoint index stores routed experts as per-expert keys
    /// (`mlp.experts.N.{gate,up,down}_proj.weight`). They are stacked into `mlp.switch_mlp.<proj>`
    /// the same way DeepseekV3.swift sanitize does. The router key is left alone.
    func testExpertStackingStacksRoutedExpertsPerProjection() throws {
        let config = try configuration(tied: false)
        var keys = ["model.layers.1.mlp.gate.weight"]
        for expert in 0 ..< 2 {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                keys.append("model.layers.1.mlp.experts.\(expert).\(proj).weight")
            }
        }
        let out = SharpAIDots3NoteSanitizer.sanitize(weights(keys), configuration: config)

        XCTAssertEqual(
            out["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [2, 1])
        XCTAssertEqual(
            out["model.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape, [2, 1])
        XCTAssertNil(out["model.layers.1.mlp.experts.0.gate_proj.weight"])
        XCTAssertNotNil(out["model.layers.1.mlp.gate.weight"])
    }

    /// vLLM's Dots3NoteModel._adapt_weights skips weights whose names start with `mtp.`.
    func testDropsMTPPrefixKeys() throws {
        let config = try configuration(tied: false)
        let out = SharpAIDots3NoteSanitizer.sanitize(
            weights(["mtp.layers.0.eh_proj.weight", "model.layers.1.mlp.gate.weight"]),
            configuration: config)

        XCTAssertNil(out["mtp.layers.0.eh_proj.weight"])
        XCTAssertNotNil(out["model.layers.1.mlp.gate.weight"])
    }

    /// The audio adapter prefix `audio_encoder.` is confirmed in the checkpoint index. Its layer keys
    /// must not be cut by num_hidden_layers (here 2).
    func testAudioEncoderKeysAreNeverDroppedByLayerIndex() throws {
        let config = try configuration(tied: false)
        let key = "audio_encoder.layers.5.self_attn.q_proj.weight"
        let out = SharpAIDots3NoteSanitizer.sanitize(weights([key]), configuration: config)

        XCTAssertNotNil(out[key])
    }

    /// Sliding-layer `.indexer.` keys are dropped, matching vLLM's loader.
    func testDropsIndexerKeysOnSlidingLayersOnly() throws {
        let config = try configuration(tied: false)
        let out = SharpAIDots3NoteSanitizer.sanitize(
            weights([
                "model.layers.0.self_attn.indexer.wk.weight",
                "model.layers.1.self_attn.indexer.wk.weight",
            ]),
            configuration: config)

        XCTAssertNotNil(out["model.layers.0.self_attn.indexer.wk.weight"])
        XCTAssertNil(out["model.layers.1.self_attn.indexer.wk.weight"])
    }
}
