// SharpAI fork overlay tests for the minicpmv4_7 model class. Not part of upstream.
// Tiny random-weight shapes only, no network. Key layout is checked against the HF names in
// transformers modeling_minicpmv4_7.py. Logit values are not checked.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

enum SharpAIMinicpmv47Fixtures {
    /// Tiny config with the same structure as openbmb/MiniCPM-V-4.7-1B. Special token ids sit below
    /// the 128-token vocabulary.
    static let tinyJSON = """
        {
            "model_type": "minicpmv4_7",
            "insert_layer_id": 0,
            "downsample_mode": "16x",
            "image_token_id": 120,
            "video_token_id": 121,
            "image_start_id": 122,
            "image_end_id": 123,
            "slice_start_id": 124,
            "slice_end_id": 125,
            "newline_id": 126,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 128,
                "tie_word_embeddings": false,
                "full_attention_interval": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
            },
            "vision_config": {
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_channels": 3,
                "image_size": 56,
                "patch_size": 14,
                "layer_norm_eps": 1e-6
            }
        }
        """
}

final class SharpAIMinicpmv47ModelTests: XCTestCase {

    func testRemapsVisionTowerAndMergers() {
        XCTAssertEqual(
            SharpAIMinicpmv47.remapKey("model.vision_tower.embeddings.patch_embedding.weight"),
            "vpm.embeddings.patch_embedding.weight")
        XCTAssertEqual(
            SharpAIMinicpmv47.remapKey("model.vision_tower.encoder.layers.0.mlp.fc1.weight"),
            "vpm.encoder.layers.0.mlp.fc1.weight")
        XCTAssertEqual(
            SharpAIMinicpmv47.remapKey("model.vision_tower.vit_merger.pre_norm.weight"),
            "vit_merger.pre_norm.weight")
        XCTAssertEqual(
            SharpAIMinicpmv47.remapKey("model.merger.mlp.0.linear_1.weight"),
            "merger.mlp.0.linear_1.weight")
    }

    func testRemapsLanguageModelAndLMHead() {
        XCTAssertEqual(
            SharpAIMinicpmv47.remapKey("model.language_model.layers.3.mlp.gate_proj.weight"),
            "language_model.model.layers.3.mlp.gate_proj.weight")
        XCTAssertEqual(
            SharpAIMinicpmv47.remapKey("lm_head.weight"),
            "language_model.lm_head.weight")
    }

    func testDropsMTPAndRotaryKeys() {
        XCTAssertNil(SharpAIMinicpmv47.remapKey("mtp.0.fc.weight"))
        XCTAssertNil(
            SharpAIMinicpmv47.remapKey(
                "model.language_model.layers.0.self_attn.rotary_emb.inv_freq"))
        XCTAssertNil(
            SharpAIMinicpmv47.remapKey("model.vision_tower.embeddings.position_ids"))
    }

    func testPatchEmbeddingIsChannelsLast() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv47Configuration.self, from: Data(SharpAIMinicpmv47Fixtures.tinyJSON.utf8))
        let model = try SharpAIMinicpmv47(config)

        // HF layout (out, in, kH, kW) is moved to MLX layout (out, kH, kW, in).
        let sanitized = model.sanitize(weights: [
            "model.vision_tower.embeddings.patch_embedding.weight": MLXArray.zeros([16, 3, 14, 14])
        ])
        XCTAssertEqual(sanitized["vpm.embeddings.patch_embedding.weight"]?.shape, [16, 14, 14, 3])
    }

    func testSanitizeKeepsLMHeadAndDropsMTP() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv47Configuration.self, from: Data(SharpAIMinicpmv47Fixtures.tinyJSON.utf8))
        let model = try SharpAIMinicpmv47(config)

        let sanitized = model.sanitize(weights: [
            "model.language_model.embed_tokens.weight": MLXArray.zeros([128, 32]),
            "lm_head.weight": MLXArray.zeros([128, 32]),
            "mtp.0.fc.weight": MLXArray.zeros([4, 4]),
        ])

        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["language_model.lm_head.weight"])
        XCTAssertNil(sanitized["mtp.0.fc.weight"])
    }

    /// 4x4 grid, 2x2 windows: windows row-major, patches row-major inside each window.
    func testWindowOrderGroupsTwoByTwoWindows() {
        let order = SharpAIMinicpmv47WindowMerger.windowOrder(height: 4, width: 4)
        XCTAssertEqual(order, [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
    }

    /// Hand-traced canvas M-RoPE for one image: text, `<image>`, four placeholders on an 8x8 patch
    /// grid (2x2 LLM tokens, 16x mode), `</image>`, text. Expected rows follow
    /// `get_rope_index` in modeling_minicpmv4_7.py.
    func testCanvasPositionsForSingleImage() throws {
        let canvas = SharpAIMinicpmv47CanvasPositions(
            imageStartId: 122, imageEndId: 123, sliceStartId: 124, sliceEndId: 125,
            newlineId: 126, mergeFactor: 4)
        let ids = [5, 122, 120, 120, 120, 120, 123, 6]
        let modalities = [0, 0, 1, 1, 1, 1, 0, 0]

        let result = try canvas.compute(
            tokenIds: ids, modalities: modalities, imageGrids: [(8, 8)], videoGrids: [])

        XCTAssertEqual(result.positions[0], [0, 1, 1, 1, 1, 1, 1, 4])
        XCTAssertEqual(result.positions[1], [0, 0, 1, 1, 2, 2, 3, 4])
        XCTAssertEqual(result.positions[2], [0, 0, 1, 2, 1, 2, 3, 4])
        XCTAssertEqual(result.delta, -3)
    }

    /// The canvas delta travels in the request state. A fresh state has none, so one request's delta
    /// cannot shift another request's positions.
    func testRopeDeltaComesFromStateNotModel() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv47Configuration.self, from: Data(SharpAIMinicpmv47Fixtures.tinyJSON.utf8))
        let model = try SharpAIMinicpmv47(config)

        XCTAssertEqual(SharpAIMinicpmv47.ropeDelta(nil), 0)
        XCTAssertEqual(SharpAIMinicpmv47.ropeDelta(LMOutput.State()), 0)

        var state = LMOutput.State()
        state[SharpAIMinicpmv47.ropeDeltaKey] = MLXArray(Int32(-3))
        XCTAssertEqual(SharpAIMinicpmv47.ropeDelta(state), -3)

        let shifted = model.linearPositions(start: 10, length: 2, delta: SharpAIMinicpmv47.ropeDelta(state))
        XCTAssertEqual(shifted[0, 0].asArray(Int32.self), [7, 8])
        let plain = model.linearPositions(start: 10, length: 2, delta: SharpAIMinicpmv47.ropeDelta(nil))
        XCTAssertEqual(plain[2, 0].asArray(Int32.self), [10, 11])
    }

    func testCanvasGridCountMismatchThrows() {
        let canvas = SharpAIMinicpmv47CanvasPositions(
            imageStartId: 122, imageEndId: 123, sliceStartId: 124, sliceEndId: 125,
            newlineId: 126, mergeFactor: 4)

        XCTAssertThrowsError(
            try canvas.compute(
                tokenIds: [5, 122, 120, 120, 120, 120, 123, 6],
                modalities: [0, 0, 1, 1, 1, 1, 0, 0],
                imageGrids: [], videoGrids: [])
        )
    }

    func testPlaceholderAtIndexZeroThrowsInsteadOfTrapping() {
        let canvas = SharpAIMinicpmv47CanvasPositions(
            imageStartId: 122, imageEndId: 123, sliceStartId: 124, sliceEndId: 125,
            newlineId: 126, mergeFactor: 4)

        XCTAssertThrowsError(
            try canvas.compute(
                tokenIds: [120, 120, 120, 120, 123, 6],
                modalities: [1, 1, 1, 1, 0, 0],
                imageGrids: [(8, 8)], videoGrids: [])
        )
    }

    func testTotalMergeFactorFollowsMergerTimes() {
        XCTAssertEqual(SharpAIMinicpmv47.totalMergeFactor(windowMerge: true, mergerTimes: 1), 4)
        XCTAssertEqual(SharpAIMinicpmv47.totalMergeFactor(windowMerge: false, mergerTimes: 1), 2)
        XCTAssertEqual(SharpAIMinicpmv47.totalMergeFactor(windowMerge: true, mergerTimes: 2), 8)
        XCTAssertEqual(SharpAIMinicpmv47.totalMergeFactor(windowMerge: false, mergerTimes: 2), 4)
    }

    func testLinspaceRoundsHalfToEven() {
        // linspace(0, 2, 2) = [0, 2]; linspace(0, 1, 3) = [0, 0.5, 1] rounds to [0, 0, 1].
        XCTAssertEqual(SharpAIMinicpmv47CanvasPositions.linspaceRounded(end: 2, steps: 2), [0, 2])
        XCTAssertEqual(SharpAIMinicpmv47CanvasPositions.linspaceRounded(end: 1, steps: 3), [0, 0, 1])
    }
}
