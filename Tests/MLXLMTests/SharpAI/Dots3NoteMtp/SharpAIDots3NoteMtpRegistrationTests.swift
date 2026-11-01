// SharpAI fork overlay tests for dots3_note_mtp. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIDots3NoteMtpRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIDots3NoteMtpOverlay.creators["dots3_note_mtp"])
    }

    func testCreatorFailsWithBlockedOnBaseError() {
        guard let creator = SharpAIDots3NoteMtpOverlay.creators["dots3_note_mtp"] else {
            XCTFail("dots3_note_mtp creator missing")
            return
        }
        XCTAssertThrowsError(try creator(Data("{}".utf8))) { error in
            XCTAssertEqual(error as? SharpAIDots3NoteMtpError, .blockedOnBase)
        }
    }

    func testErrorDescriptionMentionsBaseModel() {
        let description = SharpAIDots3NoteMtpError.blockedOnBase.errorDescription ?? ""
        XCTAssertTrue(description.contains("dots3_note"))
    }
}

/// Real-source checks for the dots3_note_mtp head. Names kept from the earlier skipped versions.
/// Sources: dots-studio/dots3-note-prev config.json and model.safetensors.index.json (fetched
/// 2026-10-30), sglang NextN draft (dots3_common/nextn.py, modeling.py load_weights), transformers
/// modeling_dots3_note.py.
final class SharpAIDots3NoteMtpKnownIssueTests: XCTestCase {

    /// Verbatim subset of the checkpoint config.json. It has no MTP-specific keys, so
    /// `num_nextn_predict_layers` must default to 1.
    private static let checkpointConfigJSON = """
        {"hidden_size": 5120, "vocab_size": 152064, "intermediate_size": 13824, "num_hidden_layers": 46, "rms_norm_eps": 1e-05, "hidden_act": "silu", "max_position_embeddings": 524288, "sliding_window_size": 513, "apply_mla_qkv_lora_rescale": true, "swa_num_attention_heads": 64, "swa_num_key_value_heads": 64, "swa_q_lora_rank": 1024, "swa_kv_lora_rank": 1024, "swa_qk_nope_head_dim": 192, "swa_qk_rope_head_dim": 64, "swa_v_head_dim": 128, "swa_rope_theta": 50000.0, "swa_attention_gate_type": "headwise"}
        """

    /// Small geometry with the same structure as the checkpoint (MLA, headwise gate, dense MLP).
    private static func tinyConfig(window: Int = 513) -> SharpAIDots3NoteMtpConfiguration {
        SharpAIDots3NoteMtpConfiguration(
            hiddenSize: 32,
            vocabSize: 64,
            intermediateSize: 48,
            numHiddenLayers: 2,
            slidingWindowSize: window,
            swaNumAttentionHeads: 4,
            swaNumKeyValueHeads: 4,
            swaQLoraRank: 16,
            swaKvLoraRank: 16,
            swaQkNopeHeadDim: 8,
            swaQkRopeHeadDim: 4,
            swaVHeadDim: 8)
    }

    func testConfigurationDecodingKnownIssue() throws {
        let config = try JSONDecoder().decode(
            SharpAIDots3NoteMtpConfiguration.self, from: Data(Self.checkpointConfigJSON.utf8))

        XCTAssertEqual(config.hiddenSize, 5120)
        XCTAssertEqual(config.numHiddenLayers, 46)
        XCTAssertEqual(config.numNextnPredictLayers, 1)
        XCTAssertEqual(config.intermediateSize, 13824)
        XCTAssertEqual(config.swaNumAttentionHeads, 64)
        XCTAssertEqual(config.swaQLoraRank, 1024)
        XCTAssertEqual(config.swaKvLoraRank, 1024)
        XCTAssertEqual(config.swaQkNopeHeadDim, 192)
        XCTAssertEqual(config.swaQkRopeHeadDim, 64)
        XCTAssertEqual(config.swaQkHeadDim, 256)
        XCTAssertEqual(config.swaVHeadDim, 128)
        XCTAssertEqual(config.swaRopeTheta, 50_000, accuracy: 0.5)
        XCTAssertEqual(config.slidingWindowSize, 513)
        XCTAssertTrue(config.applyMlaQkvLoraRescale)

        // Projection widths must equal the checkpoint tensor shapes (safetensors headers, layer 46):
        // q_b_proj [16384, 1024], kv_b_proj [20480, 1024], kv_a_proj_with_mqa [1088, 5120],
        // o_proj [5120, 8192], g_proj [64, 5120].
        XCTAssertEqual(config.swaNumAttentionHeads * config.swaQkHeadDim, 16384)
        XCTAssertEqual(
            config.swaNumAttentionHeads * (config.swaQkNopeHeadDim + config.swaVHeadDim), 20480)
        XCTAssertEqual(config.swaKvLoraRank + config.swaQkRopeHeadDim, 1088)
        XCTAssertEqual(config.swaNumAttentionHeads * config.swaVHeadDim, 8192)
        XCTAssertEqual(config.swaNumAttentionHeads, 64)
    }

    func testConfigurationRejectsUnsupportedGate() {
        let json = Self.checkpointConfigJSON.replacingOccurrences(
            of: "\"headwise\"", with: "\"elementwise\"")
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                SharpAIDots3NoteMtpConfiguration.self, from: Data(json.utf8)))
    }

    func testSanitizeKeyMappingKnownIssue() {
        let mapped: [String: String] = [
            "model.mtp.embed_tokens.weight": "embed_tokens.weight",
            "model.layers.46.enorm.weight": "enorm.weight",
            "model.layers.46.hnorm.weight": "hnorm.weight",
            "model.layers.46.eh_proj.weight": "eh_proj.weight",
            "model.layers.46.shared_head.norm.weight": "shared_head.norm.weight",
            "model.layers.46.input_layernorm.weight": "decoder.input_layernorm.weight",
            "model.layers.46.post_attention_layernorm.weight":
                "decoder.post_attention_layernorm.weight",
            "model.layers.46.self_attn.q_a_proj.weight": "decoder.self_attn.q_a_proj.weight",
            "model.layers.46.self_attn.kv_b_proj.weight": "decoder.self_attn.kv_b_proj.weight",
            "model.layers.46.self_attn.g_proj.weight": "decoder.self_attn.g_proj.weight",
            "model.layers.46.self_attn.k_rope_only_layernorm.weight":
                "decoder.self_attn.k_rope_only_layernorm.weight",
            "model.layers.46.mlp.gate_proj.weight": "decoder.mlp.gate_proj.weight",
        ]
        for (checkpointKey, headKey) in mapped {
            XCTAssertEqual(
                SharpAIDots3NoteMtpSanitize.headKey(checkpointKey, numHiddenLayers: 46), headKey,
                checkpointKey)
        }

        // Target-model tensors, lower decoder layers, and a layer whose index only shares a prefix
        // (460 vs 46) are not owned by the head.
        let notOwned = [
            "model.layers.45.mlp.down_proj.weight",
            "model.layers.460.enorm.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.46.shared_head.head.weight",
        ]
        for key in notOwned {
            XCTAssertNil(
                SharpAIDots3NoteMtpSanitize.headKey(key, numHiddenLayers: 46), key)
        }
    }

    func testSanitizeCoversHeadParameters() {
        let head = SharpAIDots3NoteMtpHead(Self.tinyConfig())
        let headKeys = Set(head.parameters().flattened().map { $0.0 })

        // Rebuild checkpoint-style keys for every head tensor, then sanitize them back.
        var checkpoint = [String: MLXArray]()
        for key in headKeys {
            if key == "embed_tokens.weight" {
                checkpoint["model.mtp.embed_tokens.weight"] = MLXArray(Float(0))
                continue
            }
            let rest = key.hasPrefix("decoder.") ? String(key.dropFirst("decoder.".count)) : key
            checkpoint["model.layers.2." + rest] = MLXArray(Float(0))
        }
        let sanitized = SharpAIDots3NoteMtpSanitize.sanitize(checkpoint, numHiddenLayers: 2)

        XCTAssertEqual(Set(sanitized.keys), headKeys)
    }

    private static func makeInputs(length: Int, hidden: Int, vocab: Int, seed: Int) -> (
        [Int32], [Float]
    ) {
        let ids = (0 ..< length).map { Int32(($0 * 7 + seed) % vocab) }
        let states = (0 ..< length * hidden).map { Float(sin(Double($0 + seed) * 0.37)) }
        return (ids, states)
    }

    private static func runHead(
        _ head: SharpAIDots3NoteMtpHead, ids: [Int32], states: [Float], length: Int, hidden: Int
    ) -> [Float] {
        let out = head(
            inputIds: MLXArray(ids).reshaped(1, length),
            hiddenStates: MLXArray(states).reshaped(1, length, hidden))
        return out.asArray(Float.self)
    }

    func testTinyForwardPassKnownIssue() {
        let config = Self.tinyConfig()
        let head = SharpAIDots3NoteMtpHead(config)
        let length = 6
        let (ids, states) = Self.makeInputs(
            length: length, hidden: config.hiddenSize, vocab: config.vocabSize, seed: 1)

        let out = head(
            inputIds: MLXArray(ids).reshaped(1, length),
            hiddenStates: MLXArray(states).reshaped(1, length, config.hiddenSize))

        XCTAssertEqual(out.shape, [1, length, config.hiddenSize])
        XCTAssertTrue(out.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testTinyForwardIsCausal() {
        let config = Self.tinyConfig()
        let head = SharpAIDots3NoteMtpHead(config)
        let length = 6
        let hidden = config.hiddenSize
        let (ids, states) = Self.makeInputs(
            length: length, hidden: hidden, vocab: config.vocabSize, seed: 1)
        let base = Self.runHead(head, ids: ids, states: states, length: length, hidden: hidden)

        // Change the token and the target state at position 4 only.
        var idsChanged = ids
        idsChanged[4] = (idsChanged[4] + 5) % Int32(config.vocabSize)
        var statesChanged = states
        for d in 0 ..< hidden {
            statesChanged[4 * hidden + d] += 1.0
        }
        let changed = Self.runHead(
            head, ids: idsChanged, states: statesChanged, length: length, hidden: hidden)

        for i in 0 ..< (4 * hidden) {
            XCTAssertEqual(base[i], changed[i], accuracy: 1e-5, "position before 4 at \(i)")
        }
        let tailChanged = (4 * hidden ..< length * hidden).contains {
            abs(base[$0] - changed[$0]) > 1e-6
        }
        XCTAssertTrue(tailChanged, "changing position 4 must affect position 4 itself")
    }

    func testTinyForwardSlidingWindowLimitsReceptiveField() {
        // Window 2: position i attends to i and i-1 only, so position 0 can reach positions 0 and 1.
        let config = Self.tinyConfig(window: 2)
        let head = SharpAIDots3NoteMtpHead(config)
        let length = 5
        let hidden = config.hiddenSize
        let (ids, states) = Self.makeInputs(
            length: length, hidden: hidden, vocab: config.vocabSize, seed: 3)
        let base = Self.runHead(head, ids: ids, states: states, length: length, hidden: hidden)

        var idsChanged = ids
        idsChanged[0] = (idsChanged[0] + 9) % Int32(config.vocabSize)
        var statesChanged = states
        for d in 0 ..< hidden {
            statesChanged[d] += 2.0
        }
        let changed = Self.runHead(
            head, ids: idsChanged, states: statesChanged, length: length, hidden: hidden)

        // Positions 2..4 are outside the window of position 0 and must not change.
        for i in (2 * hidden) ..< (length * hidden) {
            XCTAssertEqual(base[i], changed[i], accuracy: 1e-5, "outside window at \(i)")
        }
        // Position 1 is inside the window, so it must change. This shows the test is not vacuous.
        let inWindowChanged = (hidden ..< 2 * hidden).contains { abs(base[$0] - changed[$0]) > 1e-6 }
        XCTAssertTrue(inWindowChanged)
    }
}
