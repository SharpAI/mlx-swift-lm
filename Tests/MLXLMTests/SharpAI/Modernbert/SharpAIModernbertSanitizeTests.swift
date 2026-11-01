// SharpAI fork overlay tests for modernbert key sanitization. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIModernbertSanitizeTests: XCTestCase {

    private func weights(_ keys: [String]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for key in keys {
            result[key] = MLXArray([0 as Float])
        }
        return result
    }

    func testBareCheckpointGetsModelPrefix() {
        let sanitized = SharpAIModernbertSanitizer.sanitize(
            weights([
                "embeddings.tok_embeddings.weight",
                "embeddings.norm.weight",
                "layers.0.attn.Wqkv.weight",
                "layers.1.mlp.Wi.weight",
                "final_norm.weight",
            ]))

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "model.embeddings.tok_embeddings.weight",
                "model.embeddings.norm.weight",
                "model.layers.0.attn.Wqkv.weight",
                "model.layers.1.mlp.Wi.weight",
                "model.final_norm.weight",
            ])
    }

    func testPrefixedCheckpointKeepsNames() {
        let sanitized = SharpAIModernbertSanitizer.sanitize(
            weights([
                "model.layers.2.attn.Wo.weight",
                "model.layers.2.attn_norm.weight",
                "model.final_norm.weight",
            ]))

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "model.layers.2.attn.Wo.weight",
                "model.layers.2.attn_norm.weight",
                "model.final_norm.weight",
            ])
    }

    func testMaskedLMHeadAndBuffersAreDropped() {
        let sanitized = SharpAIModernbertSanitizer.sanitize(
            weights([
                "model.embeddings.tok_embeddings.weight",
                "head.dense.weight",
                "head.norm.weight",
                "decoder.weight",
                "decoder.bias",
                "model.rotary_emb.inv_freq",
                "model.embeddings.position_ids",
            ]))

        XCTAssertEqual(
            Set(sanitized.keys),
            ["model.embeddings.tok_embeddings.weight"])
    }

    /// The ColBERT projection key is a guess (see the sanitize file). It must not reach the encoder.
    func testColbertProjectionIsDropped() {
        let sanitized = SharpAIModernbertSanitizer.sanitize(
            weights([
                "model.final_norm.weight",
                "linear.weight",
            ]))

        XCTAssertEqual(Set(sanitized.keys), ["model.final_norm.weight"])
    }
}
