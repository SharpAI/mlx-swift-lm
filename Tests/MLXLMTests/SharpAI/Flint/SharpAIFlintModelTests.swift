// SharpAI fork overlay tests for flint model structure and a tiny forward pass. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIFlintModelTests: XCTestCase {

    /// Tiny shapes: 3 layers (full, sliding, full), dense layer 0, sliding window 4.
    private func makeModel() throws -> SharpAIFlintModel {
        let config = try JSONDecoder.json5().decode(
            SharpAIFlintConfiguration.self,
            from: Data(
                #"""
                {
                    "model_type": "flint",
                    "vocab_size": 64,
                    "hidden_size": 32,
                    "intermediate_size": 48,
                    "num_hidden_layers": 3,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 8,
                    "use_sliding_window": true,
                    "sliding_window": 4,
                    "layer_types": ["full_attention", "sliding_attention", "full_attention"],
                    "no_rope_layer_indices": [2],
                    "moe_intermediate_size": 16,
                    "shared_expert_intermediate_size": 16,
                    "num_experts_per_tok": 2,
                    "num_experts": 4,
                    "mlp_only_layers": [0]
                }
                """#.utf8))
        try config.validateModelConfiguration()
        return SharpAIFlintModel(config)
    }

    func testParameterKeysMatchCheckpointNames() throws {
        let model = try makeModel()
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("model.layers.1.self_attn.softmax_offset"))
        XCTAssertTrue(keys.contains("model.layers.0.mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.gate.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.shared_expert.down_proj.weight"))
        XCTAssertTrue(keys.contains("lm_head.weight"))
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = try makeModel()
        // Six tokens exceed the sliding window of 4, so the windowed mask is exercised.
        let inputs = MLXArray([1, 2, 3, 4, 5, 6] as [Int32]).reshaped(1, 6)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 6, 64])
    }

    func testCachedDecodeStepAdvancesEveryLayer() throws {
        let model = try makeModel()
        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 3)

        let prompt = MLXArray([1, 2, 3, 4, 5, 6] as [Int32]).reshaped(1, 6)
        eval(model(prompt, cache: cache))
        for (i, layerCache) in cache.enumerated() {
            XCTAssertEqual(layerCache.offset, 6, "layer \(i) cache offset after prefill")
        }

        let next = MLXArray([7] as [Int32]).reshaped(1, 1)
        let logits = model(next, cache: cache)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 1, 64])
        for (i, layerCache) in cache.enumerated() {
            XCTAssertEqual(layerCache.offset, 7, "layer \(i) cache offset after decode")
        }
    }

    func testQuantizedFullAttentionCacheDoesNotTrap() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIFlintConfiguration.self,
            from: Data(
                #"""
                {
                    "model_type": "flint",
                    "vocab_size": 64,
                    "hidden_size": 128,
                    "intermediate_size": 48,
                    "num_hidden_layers": 3,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 64,
                    "use_sliding_window": true,
                    "sliding_window": 4,
                    "layer_types": ["full_attention", "sliding_attention", "full_attention"],
                    "no_rope_layer_indices": [2],
                    "moe_intermediate_size": 16,
                    "shared_expert_intermediate_size": 16,
                    "num_experts_per_tok": 2,
                    "num_experts": 4,
                    "mlp_only_layers": [0]
                }
                """#.utf8))
        try config.validateModelConfiguration()
        let model = SharpAIFlintModel(config)
        var cache = try model.newCache(parameters: nil)

        let prompt = MLXArray([1, 2, 3, 4, 5, 6] as [Int32]).reshaped(1, 6)
        eval(model(prompt, cache: cache))
        // Swap the full-attention caches for quantized ones, as kv_bits does after prefill.
        for i in [0, 2] {
            guard let simple = cache[i] as? KVCacheSimple else {
                return XCTFail("layer \(i) is not a simple cache")
            }
            cache[i] = try simple.toQuantized(groupSize: 64, bits: 8)
        }

        let logits = model(MLXArray([7] as [Int32]).reshaped(1, 1), cache: cache)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 1, 64])
        XCTAssertEqual(cache[0].offset, 7)
    }
}
