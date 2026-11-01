// SharpAI fork overlay tests for inkling_mm_model checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIInklingMmModelSanitizeTests: XCTestCase {

    // Key names copied from the thinkingmachines/Inkling model.safetensors.index.json weight_map
    // (read 2026-10-30). The checkpoint uses model.llm.*, model.audio.*, model.visual.* and
    // model.mtp.*, not the transformers module names (model.language_model.*, lm_head).
    // Tensor shapes are not checked against the checkpoint.
    private let checkpointKeys = [
        "model.llm.unembed.weight",
        "model.llm.norm.weight",
        "model.llm.layers.0.attn.q_norm.weight",
        "model.llm.layers.0.attn.k_sconv.weight",
        "model.llm.layers.0.attn.rel_logits_proj.proj",
        "model.llm.layers.0.attn.wq_du.weight",
        "model.llm.layers.0.attn_sconv.weight",
        "model.llm.layers.0.mlp.w13_dn.weight",
        "model.llm.layers.0.mlp.global_scale",
        "model.llm.layers.7.mlp.gate.global_scale",
        "model.llm.layers.10.mlp.experts.w13_weight",
        "model.llm.layers.10.mlp.experts.w2_weight",
        "model.audio.final_norm.weight",
        "model.visual.final_norm.weight",
        "model.mtp.layers.0.input_proj.weight",
        "model.mtp.layers.0.transformer_block.attn.wq_du.weight",
    ]

    private func weights(_ keys: [String]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for key in keys {
            out[key] = MLXArray(Float(0))
        }
        return out
    }

    func testSanitizeDropsOnlyMTPKeys() {
        let result = SharpAIInklingMmModelSanitizer.sanitize(weights(checkpointKeys))

        let expected = checkpointKeys.filter { !$0.hasPrefix("model.mtp.") }.sorted()
        XCTAssertEqual(result.keys.sorted(), expected)
    }

    func testSanitizeKeepsKeyNamesUnchanged() {
        let result = SharpAIInklingMmModelSanitizer.sanitize(weights(checkpointKeys))

        XCTAssertNotNil(result["model.llm.layers.0.attn.k_sconv.weight"])
        XCTAssertNotNil(result["model.llm.layers.0.attn.rel_logits_proj.proj"])
        XCTAssertNotNil(result["model.audio.final_norm.weight"])
        XCTAssertNotNil(result["model.visual.final_norm.weight"])
        XCTAssertNotNil(result["model.llm.unembed.weight"])
    }

    func testSanitizeKeepsUnembedWhenNotTied() {
        let result = SharpAIInklingMmModelSanitizer.sanitize(weights(["model.llm.unembed.weight"]))

        XCTAssertEqual(result.keys.sorted(), ["model.llm.unembed.weight"])
    }

    func testSanitizeOfEmptyDictionaryIsEmpty() {
        XCTAssertTrue(SharpAIInklingMmModelSanitizer.sanitize([:]).isEmpty)
    }

    // The checkpoint already stores MoE experts stacked (model.llm.layers.10.mlp.experts.w13_weight),
    // so no per-expert stacking is needed. Stacked tensors keep their names and shapes.
    func testExpertTensorsAreAlreadyStackedAndKept() {
        let w13 = "model.llm.layers.10.mlp.experts.w13_weight"
        let w2 = "model.llm.layers.10.mlp.experts.w2_weight"
        let input: [String: MLXArray] = [
            w13: MLXArray.zeros([4, 6, 8]),
            w2: MLXArray.zeros([4, 8, 3]),
        ]

        let result = SharpAIInklingMmModelSanitizer.sanitize(input)

        XCTAssertEqual(result.keys.sorted(), [w13, w2].sorted())
        XCTAssertEqual(result[w13]?.shape, [4, 6, 8])
        XCTAssertEqual(result[w2]?.shape, [4, 8, 3])
    }

    // The checkpoint stores short-conv weights as [channels, 1, kernel] (safetensors headers:
    // k_sconv [2048, 1, 4], attn_sconv [6144, 1, 4]). The depthwise layout is the transformers
    // nn.Conv1d weight, and the model uses MLX's [channels, kernel, 1]. The sanitizer remaps it.
    func testShortConvWeightLayoutIsRemappedToMLX() {
        let key = "model.llm.layers.0.attn.k_sconv.weight"
        let input: [String: MLXArray] = [key: MLXArray.zeros([8, 1, 4])]

        let result = SharpAIInklingMmModelSanitizer.sanitize(input)

        XCTAssertEqual(result[key]?.shape, [8, 4, 1])
    }
}
