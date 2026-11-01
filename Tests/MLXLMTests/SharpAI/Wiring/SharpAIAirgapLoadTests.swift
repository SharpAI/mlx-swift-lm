// SharpAI SL-50: air-gapped load check for the wired LLM overlays. No download, no network.
//
// For each case: build the model through the REAL registry (LLMTypeRegistry.shared), give every parameter a
// deterministic non-zero value from a fixed rule, write that as a local safetensors file, build a fresh model and
// load it with the real loader (`loadWeights`: sanitize, then update with verify .all), and require the same logits.
// This checks the wiring and the loader path. It does NOT show that the sanitizer's key names match a real
// checkpoint: the file uses the model's own parameter names.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

enum SharpAIAirgap {
    struct Case: Sendable, CustomStringConvertible {
        let key: String
        let json: String
        var description: String { key }
    }

    /// Deterministic, non-zero, different for every tensor name and element. Vectors (norm weights, biases)
    /// sit around 1; matrices stay small.
    static func fill(name: String, shape: [Int], dtype: DType) -> MLXArray {
        var h: UInt64 = 1_469_598_103_934_665_603
        for b in name.utf8 { h = (h ^ UInt64(b)) &* 1_099_511_628_211 }
        let phase = Float(h % 997) * 0.01
        let n = shape.reduce(1, *)
        var v = [Float](repeating: 0, count: n)
        let isVector = shape.count <= 1
        for i in 0 ..< n {
            let wave = sinf(0.7 * Float(i % 1009) + phase)
            v[i] = isVector ? 1 + 0.1 * wave : 0.08 * wave + 0.01 * Float((i % 5) + 1)
        }
        return MLXArray(v, shape).asType(dtype)
    }

    static func tokens(vocab: Int) -> MLXArray {
        MLXArray([1, 2, 3, 4, 5].map { Int32($0 % max(vocab, 2)) }, [1, 5])
    }

    static func vocabSize(_ json: String) -> Int {
        (try? JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])?["vocab_size"] as? Int ?? 32
    }

    static let cases: [Case] = [
        .init(key: "PanguEmbedded", json: #"""
{"model_type":"PanguEmbedded","hidden_size":32,"num_hidden_layers":2,"intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-05,"vocab_size":100,"bias":true}
"""#),
        .init(key: "bailing_hybrid", json: #"""
{"model_type": "bailing_hybrid", "hidden_size": 32, "intermediate_size": 48, "moe_intermediate_size": 16, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 4, "num_experts": 8, "num_experts_per_tok": 2, "num_shared_experts": 1, "n_group": 4, "topk_group": 2, "first_k_dense_replace": 1, "layer_group_size": 2, "group_norm_size": 2, "vocab_size": 64, "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "max_position_embeddings": 128, "routed_scaling_factor": 2.5, "head_dim": 8, "kv_lora_rank": 8, "qk_rope_head_dim": 4, "qk_nope_head_dim": 4, "v_head_dim": 8, "q_lora_rank": 12, "partial_rotary_factor": 0.5, "tie_word_embeddings": false}
"""#),
        .init(key: "cohere2_moe", json: #"""
{"model_type":"cohere2_moe","hidden_size":64,"num_hidden_layers":2,"intermediate_size":128,"num_attention_heads":8,"num_key_value_heads":2,"vocab_size":512}
"""#),
        .init(key: "deepseek_v32", json: #"""
{"model_type":"deepseek_v32","architectures":["DeepseekV32MTPModel"],"vocab_size":128,"hidden_size":64,"intermediate_size":128,"moe_intermediate_size":32,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":4,"n_shared_experts":1,"n_routed_experts":4,"routed_scaling_factor":2.5,"kv_lora_rank":16,"q_lora_rank":32,"qk_rope_head_dim":8,"v_head_dim":16,"qk_nope_head_dim":16,"norm_topk_prob":true,"n_group":1,"topk_group":1,"num_experts_per_tok":2,"moe_layer_freq":1,"first_k_dense_replace":1,"max_position_embeddings":4096,"rms_norm_eps":1e-05,"rope_theta":10000,"attention_bias":false,"index_topk":2048,"index_n_heads":4,"index_head_dim":16,"indexer_rope_interleave":true,"num_nextn_predict_layers":1}
"""#),
        .init(key: "domynedge", json: #"""
{"model_type":"domynedge","architectures":["DomynEdgeForCausalLM"],"hidden_size":16,"num_hidden_layers":4,"intermediate_size":32,"num_attention_heads":4,"num_key_value_heads":2,"head_sizes":[16,8,8,16],"position_layers":["nope","rope","rope","nope"],"window_sizes":[[-1,0],[4,0],[4,0],[-1,0]],"rms_norm_eps":1e-06,"vocab_size":64,"max_position_embeddings":128,"rope_theta":10000.0,"hidden_act":"silu","tie_word_embeddings":true}
"""#),
        .init(key: "dory", json: #"""
{"architectures":["DoryForCausalLM"],"model_type":"dory","vocab_size":64,"hidden_size":32,"num_hidden_layers":4,"intermediate_size":48,"num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":128,"tie_word_embeddings":false,"ngpt_alpha_init":0.05,"attention_output_gate_logit_scale_init":null,"mlp_bias":false,"attention_bias":false,"n_input_layers":1,"n_recurrent_layers":2,"n_output_layers":1,"n_recurrent_loops":2,"head_dim":16,"hidden_act":"silu","initializer_range":0.0177,"dtype":"bfloat16","sliding_window":8,"rope_parameters":{"sliding_attention":{"rope_type":"default","rope_theta":10000,"partial_rotary_factor":1.0},"full_attention":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":1.0}},"layer_types":["sliding_attention","mlp","full_attention","mlp"]}
"""#),
        .init(key: "exaone_moe", json: #"""
{"model_type":"exaone_moe","vocab_size":64,"hidden_size":16,"num_hidden_layers":2,"intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8}
"""#),
        .init(key: "flint", json: #"""
{"model_type":"flint","vocab_size":512,"hidden_size":32,"intermediate_size":64,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"rms_norm_eps":1e-06,"rope_theta":10000.0,"use_sliding_window":true,"sliding_window":4,"layer_types":["full_attention","sliding_attention","full_attention"],"no_rope_layer_indices":[2],"moe_intermediate_size":16,"shared_expert_intermediate_size":16,"num_experts_per_tok":2,"num_experts":4,"norm_topk_prob":true,"router_score_function":"sigmoid","mlp_only_layers":[0]}
"""#),
        .init(key: "granite_swa", json: #"""
{"model_type":"granite_swa","hidden_size":64,"num_hidden_layers":2,"intermediate_size":128,"num_attention_heads":8,"num_key_value_heads":2,"vocab_size":512}
"""#),
        .init(key: "granitemoe_swa", json: #"""
{"model_type":"granitemoe_swa","vocab_size":64,"hidden_size":16,"intermediate_size":8,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"num_local_experts":4,"num_experts_per_tok":2,"shared_intermediate_size":8,"max_position_embeddings":128,"rope_theta":10000.0,"layer_types":["sliding_attention","full_attention"],"layer_rope_theta":[0.0,10000.0],"sliding_window":2,"tie_word_embeddings":true,"attention_multiplier":0.25,"embedding_multiplier":2.0,"residual_multiplier":0.5,"logits_scaling":2.0}
"""#),
        .init(key: "hrm_text", json: #"""
{"model_type":"hrm_text","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":8,"num_key_value_heads":2,"intermediate_size":128,"vocab_size":512,"H_cycles":2,"L_cycles":3,"L_bp_cycles":[0]}
"""#),
        .init(key: "hy_v3", json: #"""
{"model_type": "hy_v3", "hidden_size": 64, "intermediate_size": 128, "moe_intermediate_size": 32, "expert_hidden_dim": 32, "first_k_dense_replace": 1, "head_dim": 16, "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 3, "num_experts": 4, "num_experts_per_tok": 2, "num_shared_experts": 1, "moe_router_enable_expert_bias": true, "moe_router_use_sigmoid": true, "qk_norm": true, "rms_norm_eps": 1e-05, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "route_norm": true, "router_scaling_factor": 2.8, "tie_word_embeddings": false, "vocab_size": 128, "max_position_embeddings": 256, "num_nextn_predict_layers": 1}
"""#),
        .init(key: "hy_v4", json: #"""
{"architectures":["HYV4ForCausalLM"],"model_type":"hy_v4","vocab_size":100,"hidden_size":32,"intermediate_size":48,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":4,"head_dim":4,"rms_norm_eps":1e-05,"rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},"max_position_embeddings":128,"attention_bias":false,"hidden_act":"silu","tie_word_embeddings":false,"q_lora_rank":16,"kv_lora_rank":8,"qk_nope_head_dim":8,"qk_rope_head_dim":4,"v_head_dim":8,"gated_mla":true,"learnable_sink":true,"index_head_dim":8,"index_n_heads":2,"index_topk":4,"indexer_types":["full","shared","full"],"layer_types":["deepseek_sparse_attention","deepseek_sparse_attention","deepseek_sparse_attention"],"n_routed_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":16,"n_shared_experts":1,"n_group":1,"topk_group":1,"routed_scaling_factor":2.827,"norm_topk_prob":true,"gating_type":"elementwise","mlp_layer_types":["dense","sparse","sparse"],"swiglu_limit":10.0,"enable_ihc":true,"hc_mult":2,"hc_magnitude":2.0,"hc_eps":1e-06,"num_nextn_predict_layers":1}
"""#),
        .init(key: "hyperclovax", json: #"""
{"model_type": "hyperclovax", "attention_bias": false, "attention_multiplier": 0.125, "embedding_multiplier": 10.0, "head_dim": 16, "hidden_size": 64, "intermediate_size": 128, "logits_scaling": 0.125, "max_position_embeddings": 256, "mlp_bias": false, "num_attention_heads": 4, "num_hidden_layers": 2, "num_key_value_heads": 2, "residual_multiplier": 1.0, "rms_norm_eps": 1e-05, "rope_scaling": null, "rope_theta": 10000.0, "tie_word_embeddings": false, "use_post_norm": true, "vocab_size": 128}
"""#),
        .init(key: "iquestcoder", json: #"""
{"model_type":"iquestcoder","hidden_size":32,"num_hidden_layers":2,"intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-05,"vocab_size":100}
"""#),
        .init(key: "iquestloopcoder", json: #"""
{"model_type":"iquestloopcoder","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":8,"num_key_value_heads":2,"intermediate_size":128,"vocab_size":512}
"""#),
        .init(key: "jais2", json: #"""
{"model_type":"jais2","vocab_size":64,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":4}
"""#),
        .init(key: "kolibri1", json: #"""
{"architectures":["Kolibri1ForCausalLM"],"model_type":"kolibri1","hidden_size":16,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,"hidden_act":"silu","max_position_embeddings":64,"rms_norm_eps":1e-06,"vocab_size":32,"rope_theta":10000.0,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":8,"shared_expert_intermediate_size":8,"norm_topk_prob":true,"attention_bias":false,"tie_word_embeddings":false,"use_sliding_window":true,"sliding_window":3,"layer_types":["sliding_attention","full_attention"],"eos_token_id":1,"pad_token_id":0}
"""#),
        .init(key: "laguna", json: #"""
{"model_type":"laguna","vocab_size":100,"hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"layer_types":["full_attention","sliding_attention"],"sliding_window":4,"rope_theta":5000.0,"partial_rotary_factor":0.5,"mlp_layer_types":["dense","dense"]}
"""#),
        .init(key: "minimax_m3_text", json: #"""
{"architectures":["MiniMaxM3SparseForCausalLM"],"model_type":"minimax_m3_text","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":8,"num_key_value_heads":2,"vocab_size":1000}
"""#),
        .init(key: "muse_glimmer_text", json: #"""
{"model_type":"muse_glimmer_text","architectures":["MuseGlimmerForCausalLM"],"vocab_size":40,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"max_position_embeddings":128,"rms_norm_eps":1e-05,"post_norm_eps":1e-08,"attention_bias":false,"sliding_window":4,"qk_scale_factor":3.87,"output_multiplier":0.5,"final_logit_softcapping":20.0,"tie_word_embeddings":false,"rope_parameters":{"rope_theta":500000.0,"rope_type":"default"}}
"""#),
        .init(key: "nemotron_h_mtp", json: #"""
{"model_type":"nemotron_h_mtp","vocab_size":100,"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"mamba_num_heads":4,"mamba_head_dim":16,"ssm_state_size":16,"conv_kernel":4,"n_groups":2,"intermediate_size":128,"moe_intermediate_size":64,"moe_shared_expert_intermediate_size":64,"n_routed_experts":4,"num_experts_per_tok":2,"hybrid_override_pattern":"M*","layer_norm_epsilon":1e-05,"n_group":2,"topk_group":1}
"""#),
        .init(key: "olmo_hybrid", json: #"""
{"model_type": "olmo_hybrid", "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 4, "intermediate_size": 128, "vocab_size": 128, "max_position_embeddings": 256, "rms_norm_eps": 1e-06, "rope_parameters": null, "linear_num_key_heads": 4, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "linear_allow_neg_eigval": true}
"""#),
        .init(key: "pangu_ultra_moe", json: #"""
{"model_type":"pangu_ultra_moe","architectures":["PanguUltraMoEForCausalLM"],"hidden_size":32,"intermediate_size":48,"moe_intermediate_size":8,"num_hidden_layers":2,"num_dense_layers":1,"num_attention_heads":4,"num_key_value_heads":4,"vocab_size":50,"num_routed_experts":4,"num_shared_experts":1,"num_experts_per_tok":2,"routed_scaling_factor":2.5,"attention_q_lora_dim":16,"attention_kv_lora_dim":8,"attention_qk_dim":8,"attention_qk_rope_dim":4,"attention_v_dim":8,"sandwich_norm":true,"rope_theta":10000,"max_position_embeddings":64}
"""#),
        .init(key: "param2moe", json: #"""
{"model_type": "param2moe", "first_k_dense_replace": 1, "head_dim": 16, "hidden_act": "silu", "hidden_size": 64, "intermediate_size": 128, "max_position_embeddings": 256, "moe_intermediate_size": 32, "moe_router_enable_expert_bias": true, "moe_shared_expert_intermediate_size": 32, "mtp_loss_scaling_factor": 0, "n_group": 1, "norm_topk_prob": true, "num_attention_heads": 4, "num_experts": 4, "num_experts_per_tok": 2, "num_hidden_layers": 3, "num_key_value_heads": 2, "num_nextn_predict_layers": 0, "num_shared_experts": 1, "partial_rotary_factor": 1.0, "rms_norm_eps": 1e-06, "rope_parameters": {"partial_rotary_factor": 1.0, "rope_theta": 10000, "rope_type": "default"}, "rope_theta": 10000, "routed_scaling_factor": 2.5, "router_dtype": "fp32", "score_function": "sigmoid", "tie_word_embeddings": true, "topk_group": 1, "use_bias": false, "use_qk_norm": true, "use_qkv_bias": false, "use_rmsnorm": true, "vocab_size": 128}
"""#),
        .init(key: "plamo3", json: #"""
{"architectures":["Plamo3ForCausalLM"],"model_type":"plamo3","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":8,"num_key_value_heads":2,"vocab_size":1000}
"""#),
        .init(key: "qwen3_5_moe_text", json: #"""
{"model_type": "qwen3_5_moe_text", "architectures": ["Qwen3_5MoeForCausalLM"], "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "linear_num_value_heads": 4, "linear_num_key_heads": 2, "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "vocab_size": 100, "full_attention_interval": 2, "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32}
"""#),
        .init(key: "sarvam_moe", json: #"""
{"model_type": "sarvam_moe", "first_k_dense_replace": 1, "head_dim": 16, "hidden_act": "silu", "hidden_size": 64, "intermediate_size": 128, "max_position_embeddings": 256, "moe_intermediate_size": 32, "moe_router_enable_expert_bias": true, "moe_shared_expert_intermediate_size": 32, "n_group": 1, "norm_topk_prob": true, "num_attention_heads": 4, "num_experts": 4, "num_experts_per_tok": 2, "num_hidden_layers": 3, "num_key_value_heads": 2, "num_shared_experts": 1, "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 10000, "routed_scaling_factor": 2.5, "score_function": "sigmoid", "tie_word_embeddings": false, "topk_group": 1, "use_bias": false, "use_qk_norm": true, "use_qkv_bias": false, "vocab_size": 128}
"""#),
        .init(key: "telechat3", json: #"""
{"architectures":["TeleChat3ForCausalLM"],"model_type":"telechat3","hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":32,"rms_norm_eps":1e-05,"rope_scaling":{"type":"telechat3-yarn","factor":4.0,"original_max_position_embeddings":16}}
"""#),
    ]
}

@Suite("SharpAI wired LLM overlays load air-gapped synthetic weights (SL-50)", .serialized)
struct SharpAIAirgapLoadTests {

    @Test("synthetic checkpoint loads through the real loader and reproduces the logits", arguments: SharpAIAirgap.cases)
    func loadsSyntheticCheckpoint(_ c: SharpAIAirgap.Case) async throws {
        // SHARPAI_AIRGAP_ONLY=<model_type> runs one case, so a trap in one model is attributed to it
        // (a trap ends the whole test process and Swift Testing cannot catch it).
        if let only = ProcessInfo.processInfo.environment["SHARPAI_AIRGAP_ONLY"], only != c.key { return }
        #expect(SharpAIModelOverlays.llmWiredKeys.contains(c.key))
        let data = Data(c.json.utf8)
        let reference = try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: c.key)

        var arrays: [String: MLXArray] = [:]
        for (name, p) in reference.parameters().flattened() {
            arrays[name] = SharpAIAirgap.fill(name: name, shape: p.shape, dtype: p.dtype)
        }
        #expect(!arrays.isEmpty)
        reference.update(parameters: ModuleParameters.unflattened(arrays))
        eval(reference)

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sharpai-airgap-\(c.key)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try data.write(to: dir.appendingPathComponent("config.json"))
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))

        let loaded = try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: c.key)
        try await loadWeights(modelDirectory: dir, model: loaded)
        eval(loaded)

        let input = SharpAIAirgap.tokens(vocab: SharpAIAirgap.vocabSize(c.json))
        let a = reference(input, cache: nil)
        let b = loaded(input, cache: nil)
        eval(a, b)
        #expect(a.shape == b.shape)
        let finite = all(isFinite(a)).item(Bool.self)
        #expect(finite, "\(c.key): reference logits not finite")
        #expect(abs(a).max().item(Float.self) > 0, "\(c.key): logits are all zero")
        #expect(allClose(a, b, rtol: 1e-3, atol: 1e-4).item(Bool.self), "\(c.key): loaded logits differ from the reference")
    }
}
