// SharpAI SL-51 PR1: air-gapped load check for the VLM-registry overlays. No download, no network.
//
// Same method as `SharpAIAirgapLoadTests` (SL-50): build the model through the REAL registry
// (`VLMTypeRegistry.shared`), give every parameter a deterministic non-zero value, write a local safetensors file,
// load it into a fresh model with the real loader (`loadWeights`: sanitize, then update with verify .all) and
// require the same logits. The file uses the model's own parameter names, so this proves the wiring and the loader
// path, not that the sanitizer matches a real checkpoint.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

@Suite("SharpAI wired VLM-registry overlays load air-gapped synthetic weights (SL-51)", .serialized)
struct SharpAIVLMAirgapLoadTests {

    static let cases: [SharpAIAirgap.Case] = [
        .init(key: "cosmos3_omni", json: #"""
{"model_type":"cosmos3_omni","architectures":["Cosmos3OmniForConditionalGeneration"],"tie_word_embeddings":false,"image_token_id":60,"video_token_id":61,"vision_start_token_id":62,"vision_end_token_id":63,"model":{"config":{"sound_gen":true,"action_gen":true,"latent_channel":2,"latent_patch_size":2,"max_action_dim":4,"sound_dim":4,"num_embodiment_domains":3}},"text_config":{"model_type":"qwen3_vl_text","hidden_size":32,"intermediate_size":32,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"max_position_embeddings":512,"rms_norm_eps":1e-06,"vocab_size":64,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[2,1,1],"rope_theta":10000,"rope_type":"default"}},"vision_config":{"deepstack_visual_indexes":[0,1],"depth":2,"hidden_size":32,"in_channels":3,"intermediate_size":32,"model_type":"qwen3_vl_vision","num_heads":4,"num_position_embeddings":16,"out_hidden_size":32,"patch_size":16,"spatial_merge_size":1,"temporal_patch_size":2}}
"""#),
        .init(key: "interns1_pro", json: #"""
{"model_type":"interns1_pro","tie_word_embeddings":false,"text_config":{"hidden_size":8,"num_hidden_layers":1,"intermediate_size":16,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"vocab_size":16}}
"""#),
        .init(key: "interns2_mobius", json: #"""
{"model_type":"interns2_mobius","text_config":{"model_type":"interns2_mobius_text","vocab_size":64,"hidden_size":32,"num_hidden_layers":2,"layer_types":["linear_attention","full_attention"],"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"rms_norm_eps":1e-06,"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":8,"shared_expert_intermediate_size":8,"num_blocks":2,"rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
"""#),
        .init(key: "kimi_k3", json: #"""
{"model_type":"kimi_k3","architectures":["KimiK3ForConditionalGeneration"],"text_config":{"vocab_size":16,"hidden_size":32,"intermediate_size":48,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"hidden_act":"situ","rms_norm_eps":1e-05,"moe_intermediate_size":8,"moe_renormalize":true,"moe_router_activation_func":"sigmoid","num_experts":4,"num_experts_per_token":2,"num_shared_experts":1,"routed_scaling_factor":1.0,"first_k_dense_replace":1,"moe_layer_freq":1,"num_expert_group":1,"topk_group":1,"q_lora_rank":8,"kv_lora_rank":8,"qk_nope_head_dim":8,"qk_rope_head_dim":4,"v_head_dim":8,"mla_use_nope":true,"mla_use_output_gate":true,"attn_res_block_size":2,"latent_moe_use_norm":true,"routed_expert_hidden_size":16,"activation_situ_beta":4.0,"activation_situ_linear_beta":25.0,"linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],"num_heads":2,"head_dim":8,"short_conv_kernel_size":4,"use_full_rank_gate":true,"gate_lower_bound":-5.0}}}
"""#),
        .init(key: "opencua", json: SharpAIOpencuaTestFixtures.tinyJSON),
        .init(key: "openpangu_vl", json: SharpAIOpenpanguVlTextOnlyTests.json),
        .init(key: "qwen3_omni_moe", json: SharpAIQwen3OmniMoeTestConfig.tiny()),
        .init(key: "qwen4_exp", json: #"""
{"model_type":"qwen4_exp","text_config":{"eos_token_id":2,"hc_count":2,"hc_lowrank":4,"head_dim":8,"heads_per_ngram":2,"hidden_size":32,"indexer_budget":8,"indexer_compress_ratio":2,"indexer_head_dim":8,"indexer_kv_heads":1,"indexer_n_heads":2,"full_attention_interval":4,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],"linear_conv_kernel_dim":4,"linear_key_head_dim":8,"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_value_head_dim":8,"make_ngram_vocab_size_divisible_by":8,"moe_intermediate_size":16,"ngram_size":3,"ngram_vocab_size_base":50,"num_attention_heads":4,"num_experts":4,"num_experts_per_tok":2,"num_hidden_layers":4,"num_key_value_heads":2,"output_gate_type":"sigmoid","ple_conv_kernel_size":4,"ple_embed_dim":32,"ple_layer_ids":[2],"rms_norm_eps":1e-06,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[1,1,2],"partial_rotary_factor":0.5,"rope_theta":10000.0,"rope_type":"default"},"shared_expert_intermediate_size":16,"vocab_size":64}}
"""#),
        .init(key: "step1", json: #"""
{"model_type":"step1","vocab_size":64,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,"num_attention_heads":4}
"""#),
        .init(key: "intern_s2_preview", json: #"""
{"model_type":"intern_s2_preview","vocab_size":64,"image_token_id":60,"video_token_id":61,"vision_start_token_id":62,"vision_end_token_id":63,"text_config":{"model_type":"qwen3_5_moe_text","hidden_size":64,"num_hidden_layers":2,"intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":64,"full_attention_interval":2,"tie_word_embeddings":false,"linear_num_value_heads":2,"linear_num_key_heads":1,"linear_key_head_dim":32,"linear_value_head_dim":32,"linear_conv_kernel_dim":4,"num_experts":0,"num_experts_per_tok":0,"max_position_embeddings":4096,"rope_parameters":{"mrope_section":[8,4,4],"rope_theta":100000,"partial_rotary_factor":1.0,"type":"default"}},"vision_config":{"model_type":"qwen3_vl","depth":1,"hidden_size":32,"intermediate_size":64,"out_hidden_size":64,"num_heads":2,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2,"num_position_embeddings":64}}
"""#),
        .init(key: "eagle_2_5_vl", json: SharpAIEagle25VlTestFixtures.tinyJSON),
        // answer_id -1: the model sets the answer token's logit to -inf by design, which the finite check would reject.
        .init(key: "moondream3", json: SharpAIMoondream3TestFixtures.tinyJSON.replacingOccurrences(of: "\"answer_id\": 3", with: "\"answer_id\": -1")),
        .init(key: "phi4-siglip", json: SharpAIPhi4SiglipTestFixtures.tinyJSON),
    ]

    /// `vocab_size` at the top level or under `text_config`.
    static func vocabSize(_ json: String) -> Int {
        let root = (try? JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]) ?? [:]
        let text = root["text_config"] as? [String: Any]
        return root["vocab_size"] as? Int ?? text?["vocab_size"] as? Int ?? 32
    }

    /// Text-only forward through `prepare`, the path `generate` takes, so VLM-style models that do not implement
    /// `callAsFunction(_:cache:)` are exercised too. Fresh caches for every call.
    static func logits(_ model: any LanguageModel, _ tokens: MLXArray) throws -> MLXArray {
        let cache = try model.newCache(parameters: nil)
        let result = try model.prepare(
            LMInput(text: .init(tokens: tokens)), cache: cache, state: nil, prefill: PrefillParameters())
        switch result {
        case .logits(let output): return output.logits
        case .tokens(let text): return try model(text, cache: cache, state: nil).logits
        }
    }

    @Test("every wired VLM key has an air-gap case")
    func coversEveryWiredKey() {
        let covered = Set(Self.cases.map(\.key))
        #expect(covered == SharpAIVLMModelOverlays.vlmWiredKeys,
                "missing: \(SharpAIVLMModelOverlays.vlmWiredKeys.subtracting(covered).sorted()), extra: \(covered.subtracting(SharpAIVLMModelOverlays.vlmWiredKeys).sorted())")
    }

    @Test("synthetic checkpoint loads through the real loader and reproduces the logits", arguments: SharpAIVLMAirgapLoadTests.cases)
    func loadsSyntheticCheckpoint(_ c: SharpAIAirgap.Case) async throws {
        // SHARPAI_AIRGAP_ONLY=<model_type> runs one case, so a trap in one model is attributed to it.
        if let only = ProcessInfo.processInfo.environment["SHARPAI_AIRGAP_ONLY"], only != c.key { return }
        #expect(SharpAIVLMModelOverlays.vlmWiredKeys.contains(c.key))
        let data = Data(c.json.utf8)
        let reference = try await VLMTypeRegistry.shared.createModel(configuration: data, modelType: c.key)

        var arrays: [String: MLXArray] = [:]
        for (name, p) in reference.parameters().flattened() {
            arrays[name] = SharpAIAirgap.fill(name: name, shape: p.shape, dtype: p.dtype)
        }
        #expect(!arrays.isEmpty)
        reference.update(parameters: ModuleParameters.unflattened(arrays))
        eval(reference)

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sharpai-airgap-vlm-\(c.key)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try data.write(to: dir.appendingPathComponent("config.json"))
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))

        let loaded = try await VLMTypeRegistry.shared.createModel(configuration: data, modelType: c.key)
        try await loadWeights(modelDirectory: dir, model: loaded)
        eval(loaded)

        let input = SharpAIAirgap.tokens(vocab: Self.vocabSize(c.json))
        let a = try Self.logits(reference, input)
        let b = try Self.logits(loaded, input)
        eval(a, b)
        #expect(a.shape == b.shape)
        #expect(all(isFinite(a)).item(Bool.self), "\(c.key): reference logits not finite")
        #expect(abs(a).max().item(Float.self) > 0, "\(c.key): logits are all zero")
        #expect(allClose(a, b, rtol: 1e-3, atol: 1e-4).item(Bool.self), "\(c.key): loaded logits differ from the reference")
    }
}
