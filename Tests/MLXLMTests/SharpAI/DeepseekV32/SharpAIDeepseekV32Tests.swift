// SharpAI fork overlay tests for the deepseek_v32 creator, MTP block and sanitize.
// No network and no real weights. Shapes are tiny and weights are random.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

@Suite("SharpAI DeepseekV32 overlay")
struct SharpAIDeepseekV32Tests {

    /// Tiny deepseek_v32 config: 2 main layers (layer 0 dense, layer 1 MoE) and one MTP block at index 2.
    private func configJSON() -> String {
        """
        {
            "model_type": "deepseek_v32",
            "architectures": ["DeepseekV32MTPModel"],
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
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
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000,
            "attention_bias": false,
            "index_topk": 2048,
            "index_n_heads": 4,
            "index_head_dim": 16,
            "indexer_rope_interleave": true,
            "num_nextn_predict_layers": 1
        }
        """
    }

    private func configData() -> Data {
        Data(configJSON().utf8)
    }

    private func decodeConfig() throws -> DeepseekV32Configuration {
        try JSONDecoder().decode(DeepseekV32Configuration.self, from: configData())
    }

    @Test("deepseek_v32 config decodes with the MTP depth count")
    func testConfigDecodes() throws {
        let cfg = try decodeConfig()

        #expect(cfg.numHiddenLayers == 2)
        #expect(cfg.numNextnPredictLayers == 1)
        #expect(cfg.indexTopk == 2048)
        #expect(cfg.ropeTheta == 10_000)
    }

    @Test("the deepseek_v32 key resolves to the SharpAI creator")
    func testOverlayRegistersDeepseekV32Key() throws {
        let creator = SharpAIDeepseekV32Overlay.creators["deepseek_v32"]
        #expect(creator != nil)

        let model = try creator!(configData())
        #expect(model is SharpAIDeepseekV32Model)
    }

    @Test("ModelOpt configs are rejected at load time")
    func testModelOptIsRejected() throws {
        var object = try JSONSerialization.jsonObject(with: configData()) as! [String: Any]
        object["quantization_config"] = ["quant_method": "modelopt", "quant_algo": "NVFP4"]
        let data = try JSONSerialization.data(withJSONObject: object)

        let creator = SharpAIDeepseekV32Overlay.creators["deepseek_v32"]!
        #expect(throws: SharpAIDeepseekV32Error.self) {
            _ = try creator(data)
        }
    }

    @Test("a non-ModelOpt quantization_config passes the guard")
    func testNonModelOptPassesGuard() throws {
        var object = try JSONSerialization.jsonObject(with: configData()) as! [String: Any]
        object["quantization_config"] = ["quant_method": "fp8"]
        let data = try JSONSerialization.data(withJSONObject: object)

        try SharpAIDeepseekV32ModelOptProbe.rejectIfModelOpt(data)
    }

    @Test("forward pass gives finite logits of the right shape")
    func testForwardPass() throws {
        let model = SharpAIDeepseekV32Model(try decodeConfig(), retainMTP: false)
        let input = MLXArray(0 ..< 6).reshaped(1, 6)

        let logits = model(input, cache: model.newCache(parameters: nil))

        #expect(logits.shape == [1, 6, 128])
        let sum = logits.sum().item(Float.self)
        #expect(!sum.isNaN)
        #expect(!sum.isInfinite)
    }

    @Test("callMTP returns main logits plus one entry per MTP depth")
    func testCallMTPShapes() throws {
        let model = SharpAIDeepseekV32Model(try decodeConfig(), retainMTP: true)
        let input = MLXArray(0 ..< 6).reshaped(1, 6)

        let outputs = model.callMTP(
            input,
            cache: model.newCache(parameters: nil),
            mtpCaches: model.makeMTPCaches(parameters: nil))

        #expect(outputs.count == 2)
        #expect(outputs[0].shape == [1, 6, 128])
        #expect(outputs[1].shape == [1, 6, 128])
        #expect(outputs[1].sum().item(Float.self).isFinite)
    }

    @Test("without MTP retention, callMTP returns only the main logits")
    func testCallMTPWithoutRetention() throws {
        let model = SharpAIDeepseekV32Model(try decodeConfig(), retainMTP: false)
        let input = MLXArray(0 ..< 6).reshaped(1, 6)

        let outputs = model.callMTP(
            input,
            cache: model.newCache(parameters: nil),
            mtpCaches: model.makeMTPCaches(parameters: nil))

        #expect(outputs.count == 1)
        #expect(model.makeMTPCaches(parameters: nil).isEmpty)
    }

    @Test("sanitize keeps the MTP block and drops its embed and shared head copies")
    func testSanitizeKeepsMTPBlock() throws {
        let cfg = try decodeConfig()
        let weights: [String: MLXArray] = [
            "model.layers.2.eh_proj.weight": MLXArray.zeros([64, 128]),
            "model.layers.2.enorm.weight": MLXArray.zeros([64]),
            "model.layers.2.hnorm.weight": MLXArray.zeros([64]),
            "model.layers.2.shared_head.norm.weight": MLXArray.zeros([64]),
            "model.layers.2.shared_head.head.weight": MLXArray.zeros([128, 64]),
            "model.layers.2.embed_tokens.weight": MLXArray.zeros([128, 64]),
            "model.layers.2.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
        ]

        let kept = SharpAIDeepseekV32Sanitizer.sanitize(
            weights, configuration: cfg, retainMTP: true)

        #expect(kept.keys.contains("model.layers.2.eh_proj.weight"))
        #expect(kept.keys.contains("model.layers.2.shared_head.norm.weight"))
        #expect(kept.keys.contains("model.layers.2.self_attn.q_a_proj.weight"))
        #expect(!kept.keys.contains("model.layers.2.shared_head.head.weight"))
        #expect(!kept.keys.contains("model.layers.2.embed_tokens.weight"))
    }

    @Test("sanitize drops the MTP block when retention is off")
    func testSanitizeDropsMTPWhenNotRetained() throws {
        let cfg = try decodeConfig()
        let weights: [String: MLXArray] = [
            "model.layers.2.eh_proj.weight": MLXArray.zeros([64, 128]),
            "model.layers.1.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
        ]

        let dropped = SharpAIDeepseekV32Sanitizer.sanitize(
            weights, configuration: cfg, retainMTP: false)

        #expect(!dropped.keys.contains("model.layers.2.eh_proj.weight"))
        #expect(dropped.keys.contains("model.layers.1.self_attn.q_a_proj.weight"))
    }

    @Test("sanitize stacks per-expert weights into switch_mlp and removes the originals")
    func testSanitizeStacksExperts() throws {
        let cfg = try decodeConfig()
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 4 {
            weights["model.layers.1.mlp.experts.\(expert).gate_proj.weight"] =
                MLXArray.zeros([32, 64])
        }

        let sanitized = SharpAIDeepseekV32Sanitizer.sanitize(
            weights, configuration: cfg, retainMTP: false)

        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape == [4, 32, 64])
        #expect(!sanitized.keys.contains { $0.contains("mlp.experts.") })
    }

    @Test("sanitize dequantizes weight_scale_inv and removes the scale key")
    func testSanitizeDequantizesScaleInv() throws {
        let cfg = try decodeConfig()
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.q_a_proj.weight": MLXArray.zeros([32, 64]),
            "model.layers.0.self_attn.q_a_proj.weight_scale_inv": MLXArray.ones([1, 1]),
        ]

        let sanitized = SharpAIDeepseekV32Sanitizer.sanitize(
            weights, configuration: cfg, retainMTP: false)

        #expect(sanitized["model.layers.0.self_attn.q_a_proj.weight"]?.shape == [32, 64])
        #expect(!sanitized.keys.contains { $0.contains("weight_scale_inv") })
    }
}
