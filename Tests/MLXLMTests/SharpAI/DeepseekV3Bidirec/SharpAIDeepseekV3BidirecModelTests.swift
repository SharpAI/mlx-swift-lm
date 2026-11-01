// SharpAI fork overlay tests for the deepseek_v3_bidirec model. Not part of upstream.
//
// Tiny random-weight forward passes only. No network and no checkpoint.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIDeepseekV3BidirecModelTests: XCTestCase {

    private func makeConfig(qLoraRank: String) throws -> SharpAIDeepseekV3BidirecConfiguration {
        let json = """
            {
                "model_type": "deepseek_v3_bidirec",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "n_group": 2,
                "topk_group": 1,
                "norm_topk_prob": true,
                "routed_scaling_factor": 1.0,
                "first_k_dense_replace": 1,
                "moe_layer_freq": 1,
                "q_lora_rank": \(qLoraRank),
                "kv_lora_rank": 4,
                "qk_rope_head_dim": 2,
                "qk_nope_head_dim": 2,
                "v_head_dim": 4,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "max_position_embeddings": 128,
                "attention_bias": false
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIDeepseekV3BidirecConfiguration.self, from: Data(json.utf8))
    }

    func testForwardPassProducesLogitsShapeWithoutQLoraRank() throws {
        let model = SharpAIDeepseekV3BidirecModel(try makeConfig(qLoraRank: "null"))
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testForwardPassProducesLogitsShapeWithQLoraRank() throws {
        let model = SharpAIDeepseekV3BidirecModel(try makeConfig(qLoraRank: "4"))
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    /// Attention is non-causal, so a change at the last position must move the logits at
    /// position 0. A causal model would give exactly zero difference there.
    func testAttentionIsNonCausal() throws {
        let model = SharpAIDeepseekV3BidirecModel(try makeConfig(qLoraRank: "null"))
        let early = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let late = MLXArray([1, 2, 7] as [Int32]).reshaped(1, 3)

        let a = model(early, cache: nil)
        let b = model(late, cache: nil)
        eval(a, b)

        let diff = abs(a[0, 0] - b[0, 0]).max().item(Float.self)
        XCTAssertGreaterThan(diff, 1e-6)
    }

    func testForwardPassUpdatesCacheExactlyOnce() throws {
        let model = SharpAIDeepseekV3BidirecModel(try makeConfig(qLoraRank: "4"))
        let cache = try model.newCache(parameters: nil)
        let tokenCount = 5
        let inputs = MLXArray(Array(Int32(1) ... Int32(tokenCount))).reshaped(1, tokenCount)

        let logits = model(inputs, cache: cache)
        eval(logits)

        for (i, layerCache) in cache.enumerated() {
            XCTAssertEqual(
                layerCache.offset, tokenCount,
                "layer \(i) cache advanced by \(layerCache.offset) for \(tokenCount) tokens")
        }
    }

    /// UNVERIFIED: whether KV-cache decoding matches a full forward is not established for a
    /// non-causal model, because a cached step cannot see tokens that are not yet in the cache.
    /// This test only checks that a one-token decode step runs and keeps the output shape.
    func testSingleTokenDecodeStepAfterPrefillKeepsShape() throws {
        let model = SharpAIDeepseekV3BidirecModel(try makeConfig(qLoraRank: "null"))
        let cache = try model.newCache(parameters: nil)
        let prompt = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let next = MLXArray([4] as [Int32]).reshaped(1, 1)

        eval(model(prompt, cache: cache))
        let logits = model(next, cache: cache)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 1, 32])
    }

    func testLoraLayersMatchDecoderStack() throws {
        let model = SharpAIDeepseekV3BidirecModel(try makeConfig(qLoraRank: "null"))
        XCTAssertEqual(model.loraLayers.count, 2)
    }
}
