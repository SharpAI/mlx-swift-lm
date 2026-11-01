import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAISarvamMoeTests: XCTestCase {

    /// Tiny config. First layer is dense, second is MoE.
    private func makeConfig(
        hiddenLayers: Int = 2,
        tieWordEmbeddings: Bool = false,
        scoreFunction: String = "sigmoid",
        numExpertsPerTok: Int = 2
    ) throws -> SharpAISarvamMoeConfiguration {
        let json = """
            {
                "model_type": "sarvam_moe",
                "architectures": ["SarvamMoEForCausalLM"],
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_hidden_layers": \(hiddenLayers),
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "use_qk_norm": true,
                "use_qkv_bias": false,
                "use_bias": false,
                "tie_word_embeddings": \(tieWordEmbeddings),
                "num_experts": 4,
                "num_experts_per_tok": \(numExpertsPerTok),
                "moe_intermediate_size": 8,
                "num_shared_experts": 1,
                "moe_shared_expert_intermediate_size": 8,
                "first_k_dense_replace": 1,
                "n_group": 2,
                "topk_group": 1,
                "routed_scaling_factor": 2.5,
                "norm_topk_prob": true,
                "score_function": "\(scoreFunction)",
                "moe_router_enable_expert_bias": true
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAISarvamMoeConfiguration.self, from: Data(json.utf8))
    }

    // MARK: Configuration

    /// Field values copied from the sarvam-30b conformance fixture.
    func testConfigurationDecodesSarvam30BFixture() throws {
        let json = """
            {
              "architectures": ["SarvamMoEForCausalLM"],
              "first_k_dense_replace": 1,
              "head_dim": 64,
              "hidden_act": "silu",
              "hidden_size": 4096,
              "intermediate_size": 8192,
              "max_position_embeddings": 131072,
              "model_type": "sarvam_moe",
              "moe_intermediate_size": 1024,
              "moe_router_enable_expert_bias": true,
              "moe_shared_expert_intermediate_size": 1024,
              "n_group": 1,
              "norm_topk_prob": true,
              "num_attention_heads": 64,
              "num_experts": 128,
              "num_experts_per_tok": 6,
              "num_hidden_layers": 19,
              "num_key_value_heads": 4,
              "num_shared_experts": 1,
              "rms_norm_eps": 1e-06,
              "rope_scaling": null,
              "rope_theta": 8000000,
              "routed_scaling_factor": 2.5,
              "score_function": "sigmoid",
              "tie_word_embeddings": false,
              "topk_group": 1,
              "use_bias": false,
              "use_qk_norm": true,
              "use_qkv_bias": false,
              "vocab_size": 262144
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAISarvamMoeConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(config.vocabularySize, 262144)
        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.hiddenLayers, 19)
        XCTAssertEqual(config.attentionHeads, 64)
        XCTAssertEqual(config.kvHeads, 4)
        XCTAssertEqual(config.headDim, 64)
        XCTAssertEqual(config.numExperts, 128)
        XCTAssertEqual(config.numExpertsPerTok, 6)
        XCTAssertEqual(config.moeSharedExpertIntermediateSize, 1024)
        XCTAssertEqual(config.routedScalingFactor, 2.5)
        XCTAssertEqual(config.ropeTheta, 8_000_000)
        XCTAssertEqual(config.partialRotaryFactor, 1.0)
        XCTAssertNil(config.ropeScaling)
        XCTAssertTrue(config.useQKNorm)
        XCTAssertFalse(config.useQKVBias)
        XCTAssertEqual(config.scoreFunction, "sigmoid")
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testConfigurationDecodesInlineJSON() throws {
        let config = try makeConfig()
        XCTAssertEqual(config.vocabularySize, 32)
        XCTAssertEqual(config.kvHeads, 1)
        XCTAssertEqual(config.headDim, 8)
        XCTAssertEqual(config.ropeDim, 8)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsSoftmaxScoreFunction() throws {
        let config = try makeConfig(scoreFunction: "softmax")
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    // MARK: Registration

    func testOverlayRegistersSarvamMoeKey() throws {
        XCTAssertNotNil(SharpAISarvamMoeOverlay.creators["sarvam_moe"])
    }

    // MARK: Sanitize

    func testSanitizeStacksRoutedExpertsAndDropsExtraLayers() throws {
        let model = SharpAISarvamMoeModel(try makeConfig())
        var weights: [String: MLXArray] = [:]
        for e in 0 ..< 4 {
            weights["model.layers.1.mlp.experts.\(e).gate_proj.weight"] = MLXArray.zeros([8, 16])
        }
        // Layer 2 is past num_hidden_layers (MTP-style extra layer) and must be dropped.
        weights["model.layers.2.mlp.experts.0.gate_proj.weight"] = MLXArray.zeros([8, 16])
        weights["model.layers.0.attention.rotary_emb.inv_freq"] = MLXArray.zeros([2])
        weights["model.word_embeddings.weight"] = MLXArray.zeros([32, 16])
        weights["lm_head.weight"] = MLXArray.zeros([32, 16])

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 8, 16])
        XCTAssertNil(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"])
        XCTAssertNil(sanitized["model.layers.2.mlp.experts.0.gate_proj.weight"])
        XCTAssertNil(sanitized["model.layers.2.mlp.switch_mlp.gate_proj.weight"])
        XCTAssertNil(sanitized["model.layers.0.attention.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["model.word_embeddings.weight"])
        XCTAssertNotNil(sanitized["lm_head.weight"])
    }

    func testSanitizeKeepsDenseLayerWeightsUnstacked() throws {
        let model = SharpAISarvamMoeModel(try makeConfig())
        var weights: [String: MLXArray] = [:]
        weights["model.layers.0.mlp.gate_proj.weight"] = MLXArray.zeros([24, 16])
        weights["model.layers.0.mlp.experts.0.gate_proj.weight"] = MLXArray.zeros([8, 16])

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["model.layers.0.mlp.gate_proj.weight"])
        XCTAssertNil(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
    }

    func testSanitizeDropsLMHeadWhenEmbeddingsTied() throws {
        let model = SharpAISarvamMoeModel(try makeConfig(tieWordEmbeddings: true))
        let weights: [String: MLXArray] = [
            "model.word_embeddings.weight": MLXArray.zeros([32, 16]),
            "lm_head.weight": MLXArray.zeros([32, 16]),
        ]
        let sanitized = model.sanitize(weights: weights)
        XCTAssertNil(sanitized["lm_head.weight"])
        XCTAssertNotNil(sanitized["model.word_embeddings.weight"])
    }

    // MARK: Forward

    func testForwardPassProducesLogitsShape() throws {
        let model = SharpAISarvamMoeModel(try makeConfig())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testForwardPassUpdatesCacheExactlyOnce() throws {
        let model = SharpAISarvamMoeModel(try makeConfig())
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

    // MARK: Router

    /// modeling_sarvam_moe.py renormalizes routing weights only when top_k > 1.
    func testTopOneRoutingKeepsRawScoreUnnormalized() throws {
        let gate = SharpAISarvamMoeGate(try makeConfig(numExpertsPerTok: 1))
        // Zero weights and bias make every sigmoid score 0.5, so the output is 0.5 * 2.5.
        let (indices, weights) = gate(MLXArray.ones([1, 2, 16]))
        eval(indices, weights)
        XCTAssertEqual(indices.shape, [1, 2, 1])
        for w in weights.asArray(Float.self) {
            XCTAssertEqual(w, 1.25, accuracy: 1e-5)
        }
    }

    func testTopKRoutingWeightsSumToScalingFactor() throws {
        let gate = SharpAISarvamMoeGate(try makeConfig(numExpertsPerTok: 2))
        let (_, weights) = gate(MLXArray.ones([1, 2, 16]))
        eval(weights)
        for s in weights.sum(axis: -1).asArray(Float.self) {
            XCTAssertEqual(s, 2.5, accuracy: 1e-5)
        }
    }

    // MARK: Checkpoint key names

    /// Tensor names and shapes at the tiny test config's dimensions. The names are the ones in
    /// sarvamai/sarvam-30b's model.safetensors.index.json (weight_map, 7122 keys, fetched
    /// 2026-10-30), with the layer and expert indices filled in. Layer 0 is dense; later layers
    /// are MoE with router `mlp.gate.{weight,expert_bias}` and per-expert projections.
    private func realCheckpointShapes(
        layers: Int, firstDense: Int, experts: Int
    ) -> [String: [Int]] {
        // hidden 16, heads 2, kv 1, head_dim 8, intermediate 24, moe 8, vocab 32.
        var shapes: [String: [Int]] = [
            "model.word_embeddings.weight": [32, 16],
            "model.norm.weight": [16],
            "lm_head.weight": [32, 16],
        ]
        for layer in 0 ..< layers {
            let p = "model.layers.\(layer)"
            shapes["\(p).attention.query_key_value.weight"] = [(2 + 2 * 1) * 8, 16]
            shapes["\(p).attention.dense.weight"] = [16, 2 * 8]
            shapes["\(p).attention.query_layernorm.weight"] = [8]
            shapes["\(p).attention.key_layernorm.weight"] = [8]
            shapes["\(p).input_layernorm.weight"] = [16]
            shapes["\(p).post_attention_layernorm.weight"] = [16]
            if layer < firstDense {
                shapes["\(p).mlp.gate_proj.weight"] = [24, 16]
                shapes["\(p).mlp.up_proj.weight"] = [24, 16]
                shapes["\(p).mlp.down_proj.weight"] = [16, 24]
            } else {
                shapes["\(p).mlp.gate.weight"] = [experts, 16]
                shapes["\(p).mlp.gate.expert_bias"] = [experts]
                for e in 0 ..< experts {
                    shapes["\(p).mlp.experts.\(e).gate_proj.weight"] = [8, 16]
                    shapes["\(p).mlp.experts.\(e).up_proj.weight"] = [8, 16]
                    shapes["\(p).mlp.experts.\(e).down_proj.weight"] = [16, 8]
                }
                shapes["\(p).mlp.shared_experts.gate_proj.weight"] = [8, 16]
                shapes["\(p).mlp.shared_experts.up_proj.weight"] = [8, 16]
                shapes["\(p).mlp.shared_experts.down_proj.weight"] = [16, 8]
            }
        }
        return shapes
    }

    /// Checkpoint names from the real sarvam-30b layout must sanitize onto the module tree
    /// exactly, and load through the same `verify: [.all]` path as `loadWeights`.
    func testRealCheckpointKeyNamesMatchSarvam30BIndex() throws {
        let config = try makeConfig()
        let model = SharpAISarvamMoeModel(config)

        var checkpoint = [String: MLXArray]()
        for (key, shape) in realCheckpointShapes(
            layers: config.hiddenLayers, firstDense: config.firstKDenseReplace,
            experts: config.numExperts)
        {
            checkpoint[key] = MLXArray.zeros(shape)
        }

        let sanitized = model.sanitize(weights: checkpoint)

        XCTAssertEqual(Set(sanitized.keys), Set(model.parameters().flattened().map(\.0)))
        // Throws on a missing key or a shape mismatch.
        _ = try model.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: [.all])
    }
}
