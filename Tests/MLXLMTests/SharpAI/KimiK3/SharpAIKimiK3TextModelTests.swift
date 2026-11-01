// SharpAI fork overlay tests for the kimi_k3 text backbone. Not part of upstream.
// Inline JSON only. Random weights, tiny shapes, no checkpoint and no network.
// The fixture is a tiny stand-in with the real layer mix: KDA layers 0-2 (dense MLP at layer 0, latent MoE after),
// one MLA layer (3), attention residuals every 2 layers. Field names follow configuration_kimi_k3.py.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

enum SharpAIKimiK3TextFixtures {
    static let vocabularySize = 16

    static let tinyTextJSON = """
        {
          "model_type": "kimi_k3",
          "architectures": ["KimiK3ForConditionalGeneration"],
          "text_config": {
            "vocab_size": 16,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "hidden_act": "situ",
            "rms_norm_eps": 1e-5,
            "moe_intermediate_size": 8,
            "moe_renormalize": true,
            "moe_router_activation_func": "sigmoid",
            "num_experts": 4,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "routed_scaling_factor": 1.0,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_expert_group": 1,
            "topk_group": 1,
            "q_lora_rank": 8,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 4,
            "v_head_dim": 8,
            "mla_use_nope": true,
            "mla_use_output_gate": true,
            "attn_res_block_size": 2,
            "latent_moe_use_norm": true,
            "routed_expert_hidden_size": 16,
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "linear_attn_config": {
              "kda_layers": [1, 2, 3],
              "full_attn_layers": [4],
              "num_heads": 2,
              "head_dim": 8,
              "short_conv_kernel_size": 4,
              "use_full_rank_gate": true,
              "gate_lower_bound": -5.0
            }
          },
          "vision_config": {"hidden_size": 16}
        }
        """
}

final class SharpAIKimiK3TextModelTests: XCTestCase {

    func testTypedTextConfigurationDecodesLayerMix() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIKimiK3Configuration.self, from: Data(SharpAIKimiK3TextFixtures.tinyTextJSON.utf8))
        let text = try config.decodedTextConfiguration()

        XCTAssertEqual(text.hiddenLayers, 4)
        XCTAssertEqual(text.linearAttnConfig?.kdaLayers, [1, 2, 3])
        // Config layer lists are 1-based: layer index 0 is KDA, index 3 is full (MLA) attention.
        XCTAssertTrue(text.isKDALayer(0))
        XCTAssertFalse(text.isKDALayer(3))
        XCTAssertEqual(text.attnResBlockSize, 2)
        XCTAssertEqual(text.routedExpertHiddenSize, 16)
        XCTAssertNoThrow(try text.validateForConstruction())
    }

    func testMLAConfigurationIsRejectedWhenIncomplete() throws {
        let json = SharpAIKimiK3TextFixtures.tinyTextJSON.replacingOccurrences(
            of: "\"q_lora_rank\": 8,", with: "")
        let config = try JSONDecoder.json5().decode(
            SharpAIKimiK3Configuration.self, from: Data(json.utf8))
        let text = try config.decodedTextConfiguration()

        XCTAssertThrowsError(try text.validateForConstruction()) { error in
            guard let typed = error as? SharpAIKimiK3Error, case .unsupported = typed else {
                XCTFail("expected unsupported, got \(error)")
                return
            }
        }
    }

    func testPrefillWithCacheMatchesCachelessForward() throws {
        let model = try makeTinyModel()
        let tokens = MLXArray([1, 5, 9, 2] as [Int32]).reshaped(1, 4)

        let plain = model(tokens, cache: nil)
        let cached = model(tokens, cache: model.makeCache())

        XCTAssertEqual(cached.shape, plain.shape)
        XCTAssertTrue(MLX.allClose(cached, plain, atol: 1e-4).item(Bool.self))
    }

    func testNewCacheHasOneEntryPerLayer() throws {
        let model = try makeTinyModel()
        let caches = try model.newCache(parameters: nil)
        XCTAssertEqual(caches.count, 4)
        XCTAssertTrue(caches[0] is MambaCache)  // KDA layer
        XCTAssertFalse(caches[3] is MambaCache)  // MLA layer keeps a KV cache
    }

    private func makeTinyModel() throws -> SharpAIKimiK3Model {
        let config = try JSONDecoder.json5().decode(
            SharpAIKimiK3Configuration.self, from: Data(SharpAIKimiK3TextFixtures.tinyTextJSON.utf8))
        return SharpAIKimiK3Model(try config.decodedTextConfiguration())
    }
}

final class SharpAIKimiK3ConvSanitizeTests: XCTestCase {

    func testSplitShortConvWeightsMergeIntoOneDepthwiseConv() {
        let base = "language_model.model.layers.1.self_attn."
        let weights: [String: MLXArray] = [
            "\(base)q_conv1d.weight": MLXArray.zeros([8, 1, 4]),
            "\(base)k_conv1d.weight": MLXArray.zeros([8, 1, 4]),
            "\(base)v_conv1d.weight": MLXArray.zeros([8, 1, 4]),
        ]

        let sanitized = SharpAIKimiK3Sanitizer.sanitize(weights)

        XCTAssertNil(sanitized["\(base)q_conv1d.weight"])
        XCTAssertNil(sanitized["\(base)k_conv1d.weight"])
        XCTAssertNil(sanitized["\(base)v_conv1d.weight"])
        // Torch [C, 1, K] per branch becomes MLX [C, K, 1]; the three branches stack on channels.
        XCTAssertEqual(sanitized["\(base)conv1d.weight"]?.shape, [24, 4, 1])
    }
}
