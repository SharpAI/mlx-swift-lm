// SharpAI fork overlay tests for the hy_v4 model. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random weights, no network.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Tiny hy_v4 config: 3 layers (dense, sparse, sparse), indexer types full/shared/full, 4 routed experts.
enum SharpAIHyV4TinyFixture {
    static let json = """
        {
            "architectures": ["HYV4ForCausalLM"],
            "model_type": "hy_v4",
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 4,
            "rms_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "max_position_embeddings": 128,
            "attention_bias": false,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "q_lora_rank": 16,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 4,
            "v_head_dim": 8,
            "gated_mla": true,
            "learnable_sink": true,
            "index_head_dim": 8,
            "index_n_heads": 2,
            "index_topk": 4,
            "indexer_types": ["full", "shared", "full"],
            "layer_types": ["deepseek_sparse_attention", "deepseek_sparse_attention", "deepseek_sparse_attention"],
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "n_shared_experts": 1,
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 2.827,
            "norm_topk_prob": true,
            "gating_type": "elementwise",
            "mlp_layer_types": ["dense", "sparse", "sparse"],
            "swiglu_limit": 10.0,
            "enable_ihc": true,
            "hc_mult": 2,
            "hc_magnitude": 2.0,
            "hc_eps": 1e-06,
            "num_nextn_predict_layers": 1
        }
        """

    static func makeModel() throws -> SharpAIHyV4Model {
        let config = try JSONDecoder.json5().decode(
            SharpAIHyV4Configuration.self, from: Data(json.utf8))
        return SharpAIHyV4Model(config)
    }
}

final class SharpAIHyV4ModelTests: XCTestCase {

    func testParameterKeysFollowCheckpointLayout() throws {
        let model = try SharpAIHyV4TinyFixture.makeModel()
        let keys = Set(model.parameters().flattened().map(\.0))

        // Dense layer 0 uses a plain MLP. Sparse layers use the router and switch_mlp.
        XCTAssertTrue(keys.contains("model.layers.0.mlp.gate_proj.weight"))
        XCTAssertFalse(keys.contains("model.layers.0.mlp.gate.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.gate.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.gate.e_score_correction_bias"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.shared_experts.gate_proj.weight"))

        // Indexer only on "full" layers. "shared" layer 1 has none.
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.indexer.wq_b.weight"))
        XCTAssertFalse(keys.contains { $0.hasPrefix("model.layers.1.self_attn.indexer.") })
        XCTAssertTrue(keys.contains("model.layers.2.self_attn.indexer.weights_proj.weight"))

        // Attention sinks, output gate, and iHC parameters.
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.sinks"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.attn_hc.fn"))
        XCTAssertTrue(keys.contains("model.layers.0.ffn_hc.scale"))
        XCTAssertTrue(keys.contains("model.hc_head.hc_fn"))
    }

    func testForwardPassProducesFiniteLogitsOfVocabShape() throws {
        let model = try SharpAIHyV4TinyFixture.makeModel()
        let inputs = MLXArray([0, 1, 2, 3, 4] as [Int32]).reshaped(1, 5)
        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 5, 100])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    /// Changing only later tokens must not change earlier logits (causal mask, including
    /// the sparse selection).
    func testLogitsAtEarlierPositionsIgnoreLaterTokens() throws {
        let model = try SharpAIHyV4TinyFixture.makeModel()
        let a = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)
        let b = MLXArray([1, 2, 3, 9, 9] as [Int32]).reshaped(1, 5)

        let logitsA = model(a, cache: nil)
        let logitsB = model(b, cache: nil)
        eval(logitsA, logitsB)

        let earlierA = logitsA[0..., ..<3, 0...]
        let earlierB = logitsB[0..., ..<3, 0...]
        XCTAssertTrue(allClose(earlierA, earlierB, rtol: 1e-4, atol: 1e-5).item(Bool.self))
    }

    /// Prefill of 4 tokens then one cached decode step must match the last logits of the
    /// full 5-token forward. This exercises the latent cache and the indexer key cache.
    func testIncrementalDecodeMatchesFullForward() throws {
        let model = try SharpAIHyV4TinyFixture.makeModel()
        let tokens = MLXArray([3, 7, 11, 19, 23] as [Int32]).reshaped(1, 5)

        let full = model(tokens, cache: nil)
        eval(full)

        let cache: [KVCache] = (0 ..< 3).map { _ in KVCacheSimple() }
        let prefill = model(tokens[0..., ..<4], cache: cache)
        let step = model(tokens[0..., 4...], cache: cache)
        eval(prefill, step)

        XCTAssertEqual(step.shape, [1, 1, 100])
        XCTAssertTrue(
            allClose(full[0..., 4 ..< 5, 0...], step, rtol: 1e-3, atol: 1e-4).item(Bool.self))
    }

    /// The indexer keys live in the caller's cache: a cache rebuilt from its saved state and two
    /// sessions interleaving on one model both match the full forward.
    func testIndexerKeysFollowRestoredAndInterleavedCaches() throws {
        let model = try SharpAIHyV4TinyFixture.makeModel()
        let tokensA = MLXArray([3, 7, 11, 19, 23, 29] as [Int32]).reshaped(1, 6)
        let tokensB = MLXArray([5, 2, 8, 13, 21, 34] as [Int32]).reshaped(1, 6)
        let fullA = model(tokensA, cache: nil)
        let fullB = model(tokensB, cache: nil)
        eval(fullA, fullB)

        let cacheA: [KVCache] = (0 ..< 3).map { _ in KVCacheSimple() }
        let cacheB: [KVCache] = (0 ..< 3).map { _ in KVCacheSimple() }
        eval(model(tokensA[0..., ..<4], cache: cacheA))
        eval(model(tokensB[0..., ..<4], cache: cacheB))

        // Session A resumes from a cache rebuilt out of saved state.
        let restoredA: [KVCache] = cacheA.map { source in
            let restored = KVCacheSimple()
            restored.state = source.state
            return restored
        }
        let stepA = model(tokensA[0..., 4 ..< 5], cache: restoredA)
        let stepB = model(tokensB[0..., 4 ..< 5], cache: cacheB)
        let lastA = model(tokensA[0..., 5...], cache: restoredA)
        let lastB = model(tokensB[0..., 5...], cache: cacheB)
        eval(stepA, stepB, lastA, lastB)

        XCTAssertTrue(
            allClose(fullA[0..., 4 ..< 5, 0...], stepA, rtol: 1e-3, atol: 1e-4).item(Bool.self))
        XCTAssertTrue(
            allClose(fullB[0..., 4 ..< 5, 0...], stepB, rtol: 1e-3, atol: 1e-4).item(Bool.self))
        XCTAssertTrue(
            allClose(fullA[0..., 5...], lastA, rtol: 1e-3, atol: 1e-4).item(Bool.self))
        XCTAssertTrue(
            allClose(fullB[0..., 5...], lastB, rtol: 1e-3, atol: 1e-4).item(Bool.self))
    }

    func testSanitizeSplitsPackedExpertsIntoSwitchMLP() throws {
        let model = try SharpAIHyV4TinyFixture.makeModel()
        let experts = 4
        let inter = 16
        let hidden = 32
        let weights: [String: MLXArray] = [
            "model.layers.1.mlp.experts.gate_up_proj": MLXArray.zeros([experts, 2 * inter, hidden]),
            "model.layers.1.mlp.experts.down_proj": MLXArray.zeros([experts, hidden, inter]),
            "model.layers.3.mlp.experts.down_proj": MLXArray.zeros([experts, hidden, inter]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [experts, inter, hidden])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape,
            [experts, inter, hidden])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape,
            [experts, hidden, inter])
        XCTAssertNil(sanitized["model.layers.1.mlp.experts.gate_up_proj"])
        // Layer 3 is past num_hidden_layers (MTP), so its keys are dropped.
        XCTAssertNil(sanitized["model.layers.3.mlp.switch_mlp.down_proj.weight"])
    }
}
