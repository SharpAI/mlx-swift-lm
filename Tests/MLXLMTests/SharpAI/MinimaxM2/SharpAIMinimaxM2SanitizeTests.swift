import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIMinimaxM2SanitizeTests: XCTestCase {

    func testDefaultTableIsEmptyAndSanitizeIsPassThrough() {
        XCTAssertTrue(SharpAIMinimaxM2Sanitize.keyMapping.isEmpty)

        let weights: [String: MLXArray] = [
            "layers.0.self_attn.q_proj.weight": MLXArray.zeros([16, 16]),
            "fc.weight": MLXArray.zeros([16, 48]),
        ]
        let out = SharpAIMinimaxM2Sanitize.sanitize(weights)

        XCTAssertEqual(Set(out.keys), Set(weights.keys))
    }

    func testPrefixMappingRewritesMatchingKeysOnly() {
        let weights: [String: MLXArray] = [
            "midlayer.mlp.up_proj.weight": MLXArray.zeros([2]),
            "fc.weight": MLXArray.zeros([2]),
        ]
        let out = SharpAIMinimaxM2Sanitize.sanitize(
            weights, mapping: [(from: "midlayer.", to: "layers.0.")])

        XCTAssertNotNil(out["layers.0.mlp.up_proj.weight"])
        XCTAssertNil(out["midlayer.mlp.up_proj.weight"])
        XCTAssertNotNil(out["fc.weight"])
    }

    func testFirstMatchingPrefixWins() {
        let weights: [String: MLXArray] = ["a.b.weight": MLXArray.zeros([2])]
        let out = SharpAIMinimaxM2Sanitize.sanitize(
            weights,
            mapping: [(from: "a.", to: "x."), (from: "a.b.", to: "y.")])

        XCTAssertNotNil(out["x.b.weight"])
        XCTAssertNil(out["y.weight"])
    }
}

final class SharpAIMinimaxM2SanitizeKnownIssueTests: XCTestCase {

    // Generic EAGLE3 drafter layout, sourced from SpecForge's llama3_eagle.py (modules embed_tokens,
    // midlayer, fc, norm, lm_head and buffers t2d, d2t). A speculators-trained checkpoint saved the
    // decoder layer as layers.0.*, which the loader does not read, so the prefix must be remapped to
    // midlayer. This runs with an explicit mapping; the production table stays empty (see below).
    func testGenericEagle3LayoutRemapsDecoderPrefix() {
        let saved: [String: MLXArray] = [
            "fc.weight": MLXArray.zeros([2]),
            "layers.0.self_attn.q_proj.weight": MLXArray.zeros([2]),
            "layers.0.hidden_norm.weight": MLXArray.zeros([2]),
            "layers.0.mlp.up_proj.weight": MLXArray.zeros([2]),
            "norm.weight": MLXArray.zeros([2]),
            "lm_head.weight": MLXArray.zeros([2]),
            "embed_tokens.weight": MLXArray.zeros([2]),
            "d2t": MLXArray.zeros([2]),
            "t2d": MLXArray.zeros([2]),
        ]
        let out = SharpAIMinimaxM2Sanitize.sanitize(
            saved, mapping: [(from: "layers.0.", to: "midlayer.")])

        XCTAssertEqual(
            Set(out.keys),
            [
                "fc.weight",
                "midlayer.self_attn.q_proj.weight",
                "midlayer.hidden_norm.weight",
                "midlayer.mlp.up_proj.weight",
                "norm.weight",
                "lm_head.weight",
                "embed_tokens.weight",
                "d2t",
                "t2d",
            ])
    }

    // KNOWN GAP: the MiniMax-specific drafter tensor names are not available. The public MiniMax M2.x
    // EAGLE3 drafter thoughtworks/MiniMax-M2.5-Eagle3 declares LlamaForCausalLMEagle3, not minimax_m2.
    // Its repo has no model.safetensors.index.json (the earlier 404 was expected): it ships a single
    // model.safetensors (~464 MB), so the tensor names are only in that file's safetensors header.
    // The fetch tools cannot read that header, and downloading the file needs the owner's permission.
    // The SGLang loader rules (midlayer -> layers.0, aux_norm_* -> fc_norm.*) are source-readable, but
    // they describe SGLang's internal module names, not this checkpoint's tensor names. The production
    // keyMapping therefore stays empty until the header is read.
    func testCheckpointKeyMappingKnownIssue() {
        XCTExpectFailure(
            "KNOWN GAP: thoughtworks/MiniMax-M2.5-Eagle3 ships a single model.safetensors with no index; its tensor names need the safetensors header, which fetch tools cannot read and downloading the ~464 MB file needs owner permission; keyMapping is empty"
        ) {
            XCTAssertFalse(SharpAIMinimaxM2Sanitize.keyMapping.isEmpty)
        }
    }
}
