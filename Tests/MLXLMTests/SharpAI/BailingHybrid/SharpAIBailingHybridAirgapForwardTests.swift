// SharpAI fork overlay: air-gapped forward test for bailing_hybrid. Not part of upstream.
// Builds through the creator, saves a deterministic synthetic checkpoint under the model's own
// parameter names, reloads it through loadWeights, and checks finite logits with the hybrid cache.

import Foundation
import MLX
@testable import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class SharpAIBailingHybridAirgapForwardTests: XCTestCase {

    /// 2 layers, layer_group_size 2: layer 0 is lightning (dense MLP), layer 1 is MLA (MoE, 4 experts).
    private let json = """
        {"model_type":"bailing_hybrid","vocab_size":32,"hidden_size":32,"intermediate_size":48,
         "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":4,"head_dim":8,
         "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":false,
         "kv_lora_rank":8,"qk_nope_head_dim":4,"qk_rope_head_dim":4,"v_head_dim":4,
         "layer_group_size":2,"group_norm_size":2,"partial_rotary_factor":0.5,
         "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":8,
         "num_shared_experts":1,"moe_shared_expert_intermediate_size":8,"first_k_dense_replace":1,
         "n_group":2,"topk_group":1,"routed_scaling_factor":2.5,"moe_router_enable_expert_bias":true}
        """

    private func build() throws -> any LanguageModel {
        let creator = try XCTUnwrap(SharpAIBailingHybridOverlay.creators["bailing_hybrid"])
        return try creator(Data(json.utf8))
    }

    func testForwardWithHybridCacheIsFinite() throws {
        let model = try build()
        var arrays: [String: MLXArray] = [:]
        for (name, p) in model.parameters().flattened() {
            arrays[name] = SharpAIAirgap.fill(name: name, shape: p.shape, dtype: p.dtype)
        }
        XCTAssertFalse(arrays.isEmpty)
        XCTAssertFalse(arrays.keys.contains { $0.contains("slope") })
        model.update(parameters: ModuleParameters.unflattened(arrays))
        eval(model)

        let llm = try XCTUnwrap(model as? LLMModel)
        let cache = try llm.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 2)
        XCTAssertTrue(cache[0] is ArraysCache)
        XCTAssertFalse(cache[1] is ArraysCache)

        let tokens = MLXArray([1, 2, 3, 4, 5].map { Int32($0) }, [1, 5])
        let logits = llm(tokens, cache: cache)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 5, 32])
        XCTAssertTrue(all(isFinite(logits)).item(Bool.self))
        XCTAssertGreaterThan(abs(logits).max().item(Float.self), 0)

        // One more decode step on the filled cache.
        let step = llm(MLXArray([Int32(6)], [1, 1]), cache: cache)
        eval(step)
        XCTAssertEqual(step.shape, [1, 1, 32])
        XCTAssertTrue(all(isFinite(step)).item(Bool.self))
    }

    func testSavedCheckpointReloadsToSameLogits() async throws {
        let reference = try build()
        var arrays: [String: MLXArray] = [:]
        for (name, p) in reference.parameters().flattened() {
            arrays[name] = SharpAIAirgap.fill(name: name, shape: p.shape, dtype: p.dtype)
        }
        reference.update(parameters: ModuleParameters.unflattened(arrays))
        eval(reference)

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sharpai-airgap-bailing-hybrid-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))

        let loaded = try build()
        try await loadWeights(modelDirectory: dir, model: loaded)
        eval(loaded)

        let tokens = MLXArray([1, 2, 3, 4, 5].map { Int32($0) }, [1, 5])
        let a = (reference as! LLMModel)(tokens, cache: nil)
        let b = (loaded as! LLMModel)(tokens, cache: nil)
        eval(a, b)
        XCTAssertEqual(a.shape, b.shape)
        XCTAssertTrue(all(isFinite(a)).item(Bool.self))
        XCTAssertTrue(allClose(a, b, rtol: 1e-3, atol: 1e-4).item(Bool.self))
    }
}
