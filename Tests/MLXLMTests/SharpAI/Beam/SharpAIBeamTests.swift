import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIBeamTests: XCTestCase {

    /// Scaled-down shape of the released Beam config. Layer 1 is NoPE, so it uses SSMax.
    private let releaseShapedJSON = """
        {
            "architectures": ["BeamForCausalLM"],
            "model_type": "beam",
            "attention_bias": false,
            "final_logit_softcapping": 30.0,
            "head_dim": 16,
            "hidden_size": 64,
            "intermediate_size": 128,
            "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
            "max_position_embeddings": 1048576,
            "mlp_layer_types": ["dense", "sparse", "sparse"],
            "moe_intermediate_size": 32,
            "n_group": 8,
            "norm_topk_prob": true,
            "num_attention_heads": 4,
            "num_experts": 16,
            "num_experts_per_tok": 4,
            "num_hidden_layers": 3,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": null
            },
            "rope_theta": null,
            "routed_scaling_factor": 2.5,
            "sliding_window": 512,
            "tie_word_embeddings": false,
            "topk_group": 4,
            "vocab_size": 256
        }
        """

    /// Tiny model for the forward pass: one sliding dense layer, one full sparse layer.
    private let tinyJSON = """
        {
            "model_type": "beam",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "sliding_window": 4,
            "moe_intermediate_size": 8,
            "num_experts_per_tok": 2,
            "num_experts": 4,
            "n_group": 2,
            "topk_group": 1,
            "routed_scaling_factor": 1.0,
            "norm_topk_prob": true,
            "layer_types": ["sliding_attention", "full_attention"],
            "mlp_layer_types": ["dense", "sparse"],
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": null
            },
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": false
        }
        """

    private func decodeConfig(_ json: String) throws -> SharpAIBeamConfiguration {
        try JSONDecoder.json5().decode(SharpAIBeamConfiguration.self, from: Data(json.utf8))
    }

    func testReleaseShapedConfigDecodesAndValidates() throws {
        let config = try decodeConfig(releaseShapedJSON)
        XCTAssertEqual(config.layerTypes, ["sliding_attention", "sliding_attention", "full_attention"])
        XCTAssertEqual(config.mlpLayerTypes, ["dense", "sparse", "sparse"])
        XCTAssertEqual(config.routedScalingFactor, 2.5)
        XCTAssertEqual(config.finalLogitSoftcapping, 30.0)
        XCTAssertEqual(config.ropeTheta, 10000.0)
        XCTAssertEqual(config.nGroup, 8)
        XCTAssertEqual(config.topkGroup, 4)
        XCTAssertEqual(config.headDim, 16)
        try config.validateModelConfiguration()
    }

    func testFlatRopeParametersAndDefaults() throws {
        let json = """
            {
                "model_type": "beam",
                "num_hidden_layers": 2,
                "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}
            }
            """
        let config = try decodeConfig(json)
        XCTAssertEqual(config.ropeTheta, 500000.0)
        XCTAssertEqual(config.layerTypes, ["sliding_attention", "sliding_attention"])
        XCTAssertEqual(config.mlpLayerTypes, ["sparse", "sparse"])
        XCTAssertFalse(config.tieWordEmbeddings)
    }

    func testLayerKeyedRopeParametersPreferSlidingTheta() throws {
        let json = """
            {
                "model_type": "beam",
                "num_hidden_layers": 2,
                "rope_parameters": {
                    "a_extra": {"rope_theta": 300.0},
                    "full_attention": {"rope_theta": 500000.0},
                    "sliding_attention": {"rope_theta": 12345.0},
                    "z_extra": {"rope_theta": 900.0}
                }
            }
            """
        XCTAssertEqual(try decodeConfig(json).ropeTheta, 12345.0)
    }

    func testLayerKeyedRopeParametersFallBackToSortedFirstTheta() throws {
        let json = """
            {
                "model_type": "beam",
                "num_hidden_layers": 2,
                "rope_parameters": {
                    "z_extra": {"rope_theta": 900.0},
                    "a_extra": {"rope_theta": 300.0},
                    "full_attention": {"rope_theta": 500000.0}
                }
            }
            """
        XCTAssertEqual(try decodeConfig(json).ropeTheta, 300.0)
    }

    func testInvalidLayerSchedulesFailValidation() throws {
        var config = try decodeConfig(tinyJSON)
        config.layerTypes = ["sliding_attention"]
        XCTAssertThrowsError(try config.validateModelConfiguration())

        config = try decodeConfig(tinyJSON)
        config.mlpLayerTypes = ["dense", "shared"]
        XCTAssertThrowsError(try config.validateModelConfiguration())

        config = try decodeConfig(tinyJSON)
        config.tieWordEmbeddings = true
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testDepthScaledCoefficients() {
        // Layer 0: attention is add 2, MLP is add 3.
        let attention = sharpAIBeamDepthScales(layerIndex: 0, isAttention: true)
        XCTAssertEqual(attention.x, Float(1 / 2.0.squareRoot()), accuracy: 1e-6)
        XCTAssertEqual(attention.r, Float(1 / 2.0.squareRoot()), accuracy: 1e-6)

        let mlp = sharpAIBeamDepthScales(layerIndex: 0, isAttention: false)
        XCTAssertEqual(mlp.x, Float(1 / 3.0.squareRoot()), accuracy: 1e-6)
        XCTAssertEqual(mlp.r, Float(2.0.squareRoot() / 3.0.squareRoot()), accuracy: 1e-6)

        // Layer 2 (1-based 3): attention is add 6, so x^2 + r^2 = 1.
        let deep = sharpAIBeamDepthScales(layerIndex: 2, isAttention: true)
        XCTAssertEqual(deep.x * deep.x + deep.r * deep.r, 1, accuracy: 1e-6)
        XCTAssertEqual(deep.x, Float(1 / 6.0.squareRoot()), accuracy: 1e-6)
    }

    func testSSMaxScalesQueriesByPosition() {
        // Position 0 uses (1 + 1 / 2048) ** 0.1. Offset 2047 gives (1 + 2048 / 2048) ** 0.1 = 2 ** 0.1.
        let queries = MLXArray.ones([1, 1, 2, 4])
        let scaled = sharpAIBeamSSMaxScaled(queries, offset: 0)
        eval(scaled)
        let expectedFirst = Float(pow(1 + 1.0 / 2048.0, 0.1))
        XCTAssertEqual(scaled[0, 0, 0, 0].item(Float.self), expectedFirst, accuracy: 1e-5)
        let expectedSecond = Float(pow(1 + 2.0 / 2048.0, 0.1))
        XCTAssertEqual(scaled[0, 0, 1, 3].item(Float.self), expectedSecond, accuracy: 1e-5)

        let late = sharpAIBeamSSMaxScaled(MLXArray.ones([1, 1, 1, 4]), offset: 2047)
        eval(late)
        XCTAssertEqual(late[0, 0, 0, 0].item(Float.self), Float(pow(2.0, 0.1)), accuracy: 1e-5)
    }

    func testNVFP4NormalizeRenamesModelOptTensors() throws {
        let packed = MLXArray(Array(repeating: UInt8(0), count: 8), [1, 8])
        let blockScales = MLXArray(Array(repeating: UInt8(0), count: 1), [1, 1])
        let globalScale = MLXArray(Float(0.5))
        let inputScale = MLXArray(Float(0.25))

        let normalized = try SharpAIBeamNVFP4Checkpoint.normalize([
            "layer.weight": packed,
            "layer.weight_scale": blockScales,
            "layer.weight_scale_2": globalScale,
            "layer.input_scale": inputScale,
            "norm.weight": MLXArray.ones([4]),
        ])

        XCTAssertTrue(normalized["layer.weight"]?.dtype == .uint32)
        XCTAssertEqual(normalized["layer.weight"]?.shape, [1, 2])
        XCTAssertEqual(normalized["layer.scales"]?.shape, [1, 1])
        XCTAssertTrue(normalized["layer.global_scale"]?.dtype == .float32)
        XCTAssertTrue(normalized["layer.input_global_scale"]?.dtype == .float32)
        XCTAssertNil(normalized["layer.weight_scale"])
        XCTAssertNil(normalized["layer.weight_scale_2"])
        XCTAssertNil(normalized["layer.input_scale"])
        // Dense tensors pass through unchanged.
        XCTAssertEqual(normalized["norm.weight"]?.shape, [4])
    }

    func testSanitizeStacksExpertsAndDropsRotaryBuffers() throws {
        let config = try decodeConfig(tinyJSON)
        let model = SharpAIBeamForCausalLM(config)

        var weights = [String: MLXArray]()
        for expert in 0 ..< config.numExperts {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let prefix = "model.layers.1.mlp.experts.\(expert).\(projection)"
                // 4 rows of 8 bytes: 16 codes per row, packed as uint32 to [4, 2].
                weights["\(prefix).weight"] = MLXArray(
                    Array(repeating: UInt8(0), count: 32), [4, 8])
                weights["\(prefix).weight_scale"] = MLXArray(
                    Array(repeating: UInt8(0), count: 4), [4, 1])
                weights["\(prefix).weight_scale_2"] = MLXArray(Float(1.0))
                weights["\(prefix).input_scale"] = MLXArray(Float(0.25))
            }
        }
        weights["model.layers.0.self_attn.rotary_emb.inv_freq"] = MLXArray.ones([2])

        let sanitized = try model.sanitize(weights: weights)

        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [config.numExperts, 4, 2])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.down_proj.scales"]?.shape,
            [config.numExperts, 4, 1])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.up_proj.global_scale"]?.shape,
            [config.numExperts])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.input_global_scale"]?.shape,
            [config.numExperts])
        XCTAssertFalse(sanitized.keys.contains { $0.contains(".experts.") })
        XCTAssertFalse(sanitized.keys.contains { $0.contains("rotary_emb.inv_freq") })
    }

    func testForwardPassProducesLogitsOfVocabularyShape() throws {
        let config = try decodeConfig(tinyJSON)
        let model = SharpAIBeamForCausalLM(config)
        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, config.hiddenLayers)

        let tokens = 3
        let inputs = MLXArray(Array(Int32(1) ... Int32(tokens))).reshaped(1, tokens)
        let logits = model(inputs, cache: cache)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, tokens, config.vocabularySize])
        // Softcap keeps every logit inside (-30, 30).
        XCTAssertLessThanOrEqual(abs(logits).max().item(Float.self), 30.0)
    }

    func testOverlayCreatorBuildsModelFromJSON() throws {
        let creator = try XCTUnwrap(SharpAIBeamOverlay.creators[SharpAIBeamOverlay.modelType])
        let model = try creator(Data(tinyJSON.utf8))
        XCTAssertTrue(model is SharpAIBeamForCausalLM)
    }

    func testOverlayCreatorModelRunsForwardPass() throws {
        let creator = try XCTUnwrap(SharpAIBeamOverlay.creators[SharpAIBeamOverlay.modelType])
        let created = try creator(Data(tinyJSON.utf8))
        let model = try XCTUnwrap(created as? SharpAIBeamForCausalLM)

        let tokens = 4
        let inputs = MLXArray(Array(Int32(1) ... Int32(tokens))).reshaped(1, tokens)
        let logits = model(inputs, cache: try model.newCache(parameters: nil))
        eval(logits)

        XCTAssertEqual(logits.shape, [1, tokens, 32])
    }
}
