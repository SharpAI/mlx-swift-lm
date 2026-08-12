import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite("DeepseekV32 / glm_moe_dsa")
struct DeepseekV32Tests {

    /// GLM-5.2's real config shape: no top-level `rope_theta`, everything RoPE inside
    /// `rope_parameters`. Values are the shipped ones where they matter to decoding and
    /// shrunk where they only affect size.
    private func glmStyleConfigData(
        numHiddenLayers: Int = 4, indexTopk: Int = 2048
    ) -> Data {
        let json = """
        {
            "model_type": "glm_moe_dsa",
            "architectures": ["GlmMoeDsaForCausalLM"],
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": \(numHiddenLayers),
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "n_shared_experts": 1,
            "n_routed_experts": 4,
            "routed_scaling_factor": 2.5,
            "kv_lora_rank": 16,
            "q_lora_rank": 32,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "qk_nope_head_dim": 16,
            "norm_topk_prob": true,
            "n_group": 1,
            "topk_group": 1,
            "num_experts_per_tok": 2,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 1,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
            "attention_bias": false,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "index_topk": \(indexTopk),
            "index_n_heads": 32,
            "index_head_dim": 128,
            "indexer_rope_interleave": true,
            "num_nextn_predict_layers": 1
        }
        """
        return json.data(using: .utf8)!
    }

    @Test("glm_moe_dsa config decodes rope_theta out of rope_parameters")
    func testRopeParametersUnpacking() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData())

        #expect(cfg.ropeTheta == 8_000_000)
        // rope_type "default" means plain RoPE. Forwarding the dict as rope_scaling
        // would send the V3 stack looking for yarn fields that are not there.
        #expect(cfg.ropeScaling == nil)
        #expect(cfg.indexTopk == 2048)
        #expect(cfg.numNextnPredictLayers == 1)
    }

    @Test("glm_moe_dsa maps to a model through the factory registry")
    func testRegistryEntry() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: glmStyleConfigData(), modelType: "glm_moe_dsa")
        #expect(model is DeepseekV32Model)

        let dsv32 = try await LLMTypeRegistry.shared.createModel(
            configuration: glmStyleConfigData(), modelType: "deepseek_v3_2")
        #expect(dsv32 is DeepseekV32Model)
    }

    @Test("DeepseekV32 forward pass produces finite logits of the right shape")
    func testForwardPass() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData())
        let model = DeepseekV32Model(cfg)

        let input = MLXArray(0 ..< 6).reshaped(1, 6)
        let logits = model(input, cache: nil)

        #expect(logits.shape == [1, 6, 128])
        let sum = logits.sum().item(Float.self)
        #expect(!sum.isNaN)
        #expect(!sum.isInfinite)
    }

    @Test("DeepseekV32 sanitize drops indexer weights")
    func testSanitizeDropsIndexerWeights() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData())
        let model = DeepseekV32Model(cfg)

        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
            "model.layers.0.self_attn.indexer.wk.weight": MLXArray.zeros([128, 64]),
            "model.layers.0.self_attn.indexer.weights_proj.weight": MLXArray.zeros([32, 64]),
            "model.layers.0.self_attn.indexer.k_norm.weight": MLXArray.zeros([128]),
        ]
        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized.keys.contains("model.layers.0.self_attn.q_a_proj.weight"))
        #expect(
            !sanitized.keys.contains { $0.contains(".indexer.") },
            "indexer weights survived sanitize: \(sanitized.keys.filter { $0.contains(".indexer.") })"
        )
    }

    /// The trap this port had to avoid. DeepSeek-V3's sanitize dropped
    /// `model.layers.61` by string, because V3 has 61 layers and its multi-token
    /// prediction block sits at index 61. GLM-5.2 has 78 layers, so layer 61 is an
    /// ordinary load-bearing layer — the literal would have silently deleted it and
    /// left a model that loads and quietly computes the wrong thing.
    @Test("MTP-layer filtering follows num_hidden_layers, not a hardcoded index")
    func testLayer61IsKeptWhenItIsARealLayer() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData(numHiddenLayers: 78))
        let model = DeepseekV32Model(cfg)

        let weights: [String: MLXArray] = [
            "model.layers.61.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
            "model.layers.77.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
            "model.layers.78.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
        ]
        let sanitized = model.sanitize(weights: weights)

        #expect(
            sanitized.keys.contains("model.layers.61.self_attn.q_a_proj.weight"),
            "layer 61 is a real layer in a 78-layer model and must survive")
        #expect(sanitized.keys.contains("model.layers.77.self_attn.q_a_proj.weight"))
        #expect(
            !sanitized.keys.contains("model.layers.78.self_attn.q_a_proj.weight"),
            "layer 78 is past the main stack (MTP) and must be dropped")
    }

    /// The same filter has to keep behaving for DeepSeek-V3 itself, where the literal
    /// 61 was correct.
    @Test("DeepSeek-V3 still drops its MTP layer at index 61")
    func testDeepseekV3MTPLayerStillDropped() throws {
        let cfg = try JSONDecoder().decode(
            DeepseekV32Configuration.self, from: glmStyleConfigData(numHiddenLayers: 61))
        let model = DeepseekV32Model(cfg)

        let weights: [String: MLXArray] = [
            "model.layers.60.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
            "model.layers.61.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
        ]
        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized.keys.contains("model.layers.60.self_attn.q_a_proj.weight"))
        #expect(!sanitized.keys.contains("model.layers.61.self_attn.q_a_proj.weight"))
    }
}
