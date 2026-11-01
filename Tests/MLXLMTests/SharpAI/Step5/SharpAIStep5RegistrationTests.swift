// SharpAI fork overlay tests for step5 registration. Not part of upstream.
// Geometry and key names come from stepfun-ai/Step-5-Preview-BF16 (config.json, model.py,
// kernel.py). The three tests named *KnownGap now run; their names are kept for continuity.
// The DSA sparse path is still not ported, so sparse layers are refused, not run.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIStep5RegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "step5",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4
        }
        """

    /// Two sliding layers, MoE on layer 1, window 2. Every layer is a dense-visible layer.
    private let tinySlidingJSON = """
        {
            "model_type": "step5",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_attention_groups": 2,
            "head_dim": 4,
            "layer_types": ["sliding_attention", "sliding_attention"],
            "sliding_window": 2,
            "use_head_wise_attn_gate": true,
            "moe_layer_list": [1],
            "moe_num_experts": 4,
            "moe_top_k": 2,
            "moe_intermediate_size": 8,
            "share_expert_dim": 8,
            "swiglu_limits": [7.0, 7.0],
            "swiglu_limits_shared": [7.0, 7.0]
        }
        """

    private func loadCreator() throws -> (Data) throws -> any LanguageModel {
        guard let creator = SharpAIStep5Overlay.creators["step5"] else {
            throw XCTSkip("step5 creator missing")
        }
        return creator
    }

    func testRegistersStep5Key() {
        XCTAssertNotNil(SharpAIStep5Overlay.creators["step5"])
    }

    func testCreatorReportsModelBodyNotImplemented() {
        // The default layer types are full attention with no DSA sparse config. The
        // reference has no dense full-attention path, so the creator refuses this config.
        guard let creator = SharpAIStep5Overlay.creators["step5"] else {
            return XCTFail("step5 creator missing")
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIStep5Error.unsupported = error else {
                return XCTFail("expected unsupported, got \(error)")
            }
        }
    }

    func testCreatorRejectsInvalidConfigurationBeforeUnsupported() {
        guard let creator = SharpAIStep5Overlay.creators["step5"] else {
            return XCTFail("step5 creator missing")
        }
        let invalidJSON = validJSON.replacingOccurrences(
            of: "\"hidden_size\": 16", with: "\"hidden_size\": 18")

        XCTAssertThrowsError(try creator(Data(invalidJSON.utf8))) { error in
            guard case SharpAIStep5Error.invalidConfiguration = error else {
                return XCTFail("expected invalidConfiguration, got \(error)")
            }
        }
    }

    /// Forward pass over the dense-visible geometry: sliding attention, head-wise gate,
    /// partial RoPE, dense and MoE clamped SwiGLU, shared expert, zero-centred RMSNorm.
    /// No numeric oracle was run. The checks are shape, finiteness, and causality.
    func testForwardPassKnownGap() throws {
        let creator = try loadCreator()
        let built = try creator(Data(tinySlidingJSON.utf8))
        guard let model = built as? SharpAIStep5Model else {
            return XCTFail("expected SharpAIStep5Model, got \(type(of: built))")
        }

        let tokens = MLXArray([Int32(1), 5, 7, 3]).reshaped(1, 4)
        let logits = model(tokens, cache: nil)
        XCTAssertEqual(logits.shape, [1, 4, 32])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })

        // Two sliding layers with window 2 give a receptive field of 2 * (2 - 1) = 2 tokens.
        // Token 0 cannot reach position 3, so changing it must leave position 3 unchanged.
        let changedFirst = MLXArray([Int32(9), 5, 7, 3]).reshaped(1, 4)
        let logitsChanged = model(changedFirst, cache: nil)
        let tailDelta = abs(logits[0, 3] - logitsChanged[0, 3]).max().item(Float.self)
        XCTAssertLessThan(tailDelta, 1e-5)

        // Position 1 is inside token 0's window, so it must move. This shows the model
        // responds to its input.
        let headDelta = abs(logits[0, 1] - logitsChanged[0, 1]).max().item(Float.self)
        XCTAssertGreaterThan(headDelta, 1e-6)
    }

    /// DSA sparse attention (inference/kernel.py) is not ported, so a sparse layer is refused
    /// with an error that names DSA. This guard is what runs now. It is not a sparse-attention
    /// correctness test.
    func testSparseAttentionKnownGap() throws {
        let creator = try loadCreator()
        let sparseJSON = """
            {
                "model_type": "step5",
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "layer_types": ["full_attention"],
                "sparse_config": {"enabled": true, "apply_to_layer_types": ["full_attention"]}
            }
            """

        XCTAssertThrowsError(try creator(Data(sparseJSON.utf8))) { error in
            guard case SharpAIStep5Error.unsupported(let message) = error else {
                return XCTFail("expected unsupported, got \(error)")
            }
            XCTAssertTrue(message.contains("DSA"), "message should name DSA: \(message)")
        }
    }

    /// Parameter count from the official text_config geometry. It should match the card's
    /// "598B". Sharded loading of the real checkpoint is still not verified here.
    func testScaleKnownGap() throws {
        let layers = 92
        let layerTypes = (0 ..< layers).map { $0 % 4 == 3 ? "full_attention" : "sliding_attention" }
        let partial = (0 ..< layers).map { $0 % 4 == 3 ? "0.3333333333333333" : "1.0" }
        let limits = (0 ..< layers).map { _ in "7.0" }
        let moeList = (3 ... 90).map(String.init).joined(separator: ",")
        let json = """
            {
                "model_type": "step5",
                "vocab_size": 128896,
                "hidden_size": 4096,
                "intermediate_size": 13824,
                "num_hidden_layers": 92,
                "num_attention_heads": 64,
                "num_attention_groups": 4,
                "head_dim": 192,
                "rope_theta": 10000.0,
                "partial_rotary_factors": [\(partial.joined(separator: ","))],
                "layer_types": [\(layerTypes.map { "\"\($0)\"" }.joined(separator: ","))],
                "sliding_window": 512,
                "max_position_embeddings": 1048576,
                "use_head_wise_attn_gate": true,
                "use_moe": true,
                "moe_layer_list": [\(moeList)],
                "moe_intermediate_size": 1536,
                "share_expert_dim": 1536,
                "moe_num_experts": 352,
                "moe_top_k": 8,
                "norm_expert_weight": true,
                "moe_router_scaling_factor": 3.0,
                "swiglu_limits": [\(limits.joined(separator: ","))],
                "swiglu_limits_shared": [\(limits.joined(separator: ","))],
                "sparse_config": {"enabled": true, "apply_to_layer_types": ["full_attention"]}
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIStep5Configuration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()

        XCTAssertEqual(config.layerTypes.filter { $0 == "full_attention" }.count, 23)
        XCTAssertEqual(config.moeLayers.count, 88)
        XCTAssertEqual(approximateParameterCount(config), 598_476_052_480)
        // The 23 DSA layers still block the forward path, so the full model is refused.
        XCTAssertThrowsError(try config.validateSupportedArchitecture())
    }

    /// Parameter count from the config fields. Sparse indexer sizes come from the official
    /// sparse_config: proxy_dim 256 and 16 indexer heads.
    private func approximateParameterCount(_ c: SharpAIStep5Configuration) -> Int {
        let hidden = c.hiddenSize
        let vocab = c.vocabularySize
        let headDim = c.headDim
        let heads = c.attentionHeads
        let groups = c.kvHeads
        let experts = c.moeNumExperts
        let moeInter = c.moeIntermediateSize
        let shared = c.shareExpertDim
        let dense = c.intermediateSize
        let proxyDim = 256
        let indexerHeads = 16

        var total = 2 * vocab * hidden  // embed_tokens and untied lm_head
        for layer in 0 ..< c.hiddenLayers {
            // q, k, v, o, g, q_norm, k_norm, input and post-attention norms
            var attn =
                hidden * heads * headDim + 2 * hidden * groups * headDim
                + heads * headDim * hidden + hidden * heads
                + 2 * headDim + 2 * hidden
            if c.isSparseLayer(layer) {
                attn +=
                    hidden * indexerHeads * proxyDim + 2 * hidden * proxyDim
                    + hidden * indexerHeads + proxyDim + 2 * proxyDim
            }
            total += attn
            if c.isMoELayer(layer) {
                // gate, router_bias, stacked experts, shared expert
                total += hidden * experts + experts + 3 * experts * moeInter * hidden
                    + 3 * hidden * shared
            } else {
                total += 3 * hidden * dense
            }
        }
        return total + hidden
    }

    /// A quantized router gate holds packed words, so routing must go through the module call.
    func testMoERoutesThroughQuantizedGate() throws {
        let json = """
            {
                "vocab_size": 32,
                "hidden_size": 64,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "head_dim": 16,
                "moe_layer_list": [0],
                "moe_num_experts": 4,
                "moe_top_k": 2,
                "moe_intermediate_size": 32,
                "share_expert_dim": 32
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIStep5Configuration.self, from: Data(json.utf8))
        let moe = SharpAIStep5MoE(config, layer: 0)
        quantize(model: moe, groupSize: 64, bits: 4, filter: { path, _ in path == "gate" })
        XCTAssertTrue(moe.gate is QuantizedLinear)

        let x = MLXRandom.normal([1, 3, 64])
        let y = moe(x)
        XCTAssertEqual(y.shape, [1, 3, 64])
        XCTAssertTrue(y.asArray(Float.self).allSatisfy { $0.isFinite })
    }
}
