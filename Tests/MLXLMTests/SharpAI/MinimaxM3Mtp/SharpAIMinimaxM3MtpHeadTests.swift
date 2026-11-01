// SharpAI fork overlay tests for the minimax_m3_mtp head (SharpAIMinimaxM3MtpHead).
// Not part of upstream. Random weights, tiny shapes. These check structure and the vLLM rules
// (position-0 embedding zeroing, depth index modulo, compute_logits through the target lm_head).
// They do not check numbers against a checkpoint, because no public M3 checkpoint has mtp.* tensors.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

private enum SharpAIMinimaxM3MtpHeadFixtures {
    static let hidden = 32
    static let vocab = 64

    static let tinyTextJSON = """
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

    /// Same as `tinyTextJSON`, but the last decoder index (3) is a dense MLP layer.
    static let denseLastLayerJSON = tinyTextJSON.replacingOccurrences(
        of: "\"sparse_attention_config\"",
        with: "\"moe_layer_freq\": [1, 1, 1, 0],\n  \"sparse_attention_config\"")

    static func text(_ json: String) throws -> SharpAIMinimaxM3TextConfiguration {
        try JSONDecoder.json5().decode(SharpAIMinimaxM3TextConfiguration.self, from: Data(json.utf8))
    }
}

final class SharpAIMinimaxM3MtpHeadTests: XCTestCase {

    func testForwardAndLogitsHaveExpectedShapes() throws {
        let head = try SharpAIMinimaxM3MtpHead(
            try SharpAIMinimaxM3MtpHeadFixtures.text(SharpAIMinimaxM3MtpHeadFixtures.tinyTextJSON),
            numMtpModules: 2)
        let embeds = MLXRandom.normal([1, 6, SharpAIMinimaxM3MtpHeadFixtures.hidden])
        let previous = MLXRandom.normal([1, 6, SharpAIMinimaxM3MtpHeadFixtures.hidden])
        let lmHead = MLXRandom.normal([SharpAIMinimaxM3MtpHeadFixtures.vocab, SharpAIMinimaxM3MtpHeadFixtures.hidden])

        let hidden = head(inputsEmbeds: embeds, hiddenStates: previous)
        XCTAssertEqual(hidden.shape, [1, 6, SharpAIMinimaxM3MtpHeadFixtures.hidden])

        let logits = head.computeLogits(hidden) { $0.matmul(lmHead.T) }
        XCTAssertEqual(logits.shape, [1, 6, SharpAIMinimaxM3MtpHeadFixtures.vocab])
    }

    func testPositionZeroDoesNotDependOnItsEmbedding() throws {
        let head = try SharpAIMinimaxM3MtpHead(
            try SharpAIMinimaxM3MtpHeadFixtures.text(SharpAIMinimaxM3MtpHeadFixtures.tinyTextJSON),
            numMtpModules: 1)
        let embeds = MLXRandom.normal([1, 6, SharpAIMinimaxM3MtpHeadFixtures.hidden])
        let previous = MLXRandom.normal([1, 6, SharpAIMinimaxM3MtpHeadFixtures.hidden])
        // Same embeddings from position 1 on, a different embedding at position 0.
        let otherFirst = concatenated(
            [MLXRandom.normal([1, 1, SharpAIMinimaxM3MtpHeadFixtures.hidden]), embeds[0..., 1..., 0...]],
            axis: 1)

        let a = head(inputsEmbeds: embeds, hiddenStates: previous)
        let b = head(inputsEmbeds: otherFirst, hiddenStates: previous)

        // Position 0 is zeroed before enorm and attends only to itself, so its output is the same.
        XCTAssertTrue(
            allClose(a[0..., 0 ..< 1, 0...], b[0..., 0 ..< 1, 0...], rtol: 0, atol: 0)
                .item(Bool.self))
    }

    func testDepthIndexWrapsModuloNumMtpLayers() throws {
        let head = try SharpAIMinimaxM3MtpHead(
            try SharpAIMinimaxM3MtpHeadFixtures.text(SharpAIMinimaxM3MtpHeadFixtures.tinyTextJSON),
            numMtpModules: 2)
        let embeds = MLXRandom.normal([1, 5, SharpAIMinimaxM3MtpHeadFixtures.hidden])
        let previous = MLXRandom.normal([1, 5, SharpAIMinimaxM3MtpHeadFixtures.hidden])

        let depth0 = head(inputsEmbeds: embeds, hiddenStates: previous, step: 0)
        let depth2 = head(inputsEmbeds: embeds, hiddenStates: previous, step: 2)

        XCTAssertTrue(allClose(depth0, depth2, rtol: 0, atol: 0).item(Bool.self))
    }

    func testDenseLastDecoderIndexIsRejected() throws {
        let text = try SharpAIMinimaxM3MtpHeadFixtures.text(SharpAIMinimaxM3MtpHeadFixtures.denseLastLayerJSON)

        XCTAssertThrowsError(try SharpAIMinimaxM3MtpHead(text, numMtpModules: 1)) { error in
            guard let typed = error as? SharpAIMinimaxM3MtpError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testZeroMtpModulesIsRejected() throws {
        let text = try SharpAIMinimaxM3MtpHeadFixtures.text(SharpAIMinimaxM3MtpHeadFixtures.tinyTextJSON)

        XCTAssertThrowsError(try SharpAIMinimaxM3MtpHead(text, numMtpModules: 0))
    }
}
