// SharpAI fork overlay tests for step5: air-gapped load. Not part of upstream.
// Builds the model through SharpAIStep5Overlay.creators (not yet wired into the registry),
// fills every parameter deterministically with SharpAIAirgap.fill, saves a safetensors file
// under the model's own parameter names, loads it into a fresh model with the real loader,
// and requires finite, identical logits. No download, no network.
// This does not prove the sanitizer matches a real checkpoint's key names.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIStep5AirgapTests: XCTestCase {

    /// All sliding layers, MoE on layer 1 with 4 experts, dense on layer 0.
    private let json = """
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
            "sliding_window": 3,
            "use_head_wise_attn_gate": true,
            "moe_layer_list": [1],
            "moe_num_experts": 4,
            "moe_top_k": 2,
            "moe_intermediate_size": 8,
            "share_expert_dim": 8,
            "swiglu_limits": [7.0, 7.0, 7.0],
            "swiglu_limits_shared": [7.0, 7.0, 7.0]
        }
        """

    func testSyntheticCheckpointLoadsAndReproducesFiniteLogits() throws {
        let data = Data(json.utf8)
        let creator = try XCTUnwrap(SharpAIStep5Overlay.creators["step5"])
        let reference = try creator(data)

        var arrays: [String: MLXArray] = [:]
        for (name, p) in reference.parameters().flattened() {
            arrays[name] = SharpAIAirgap.fill(name: name, shape: p.shape, dtype: p.dtype)
        }
        XCTAssertFalse(arrays.isEmpty)
        XCTAssertTrue(arrays.keys.contains("model.layers.1.moe.switch_mlp.gate_proj.weight"))
        XCTAssertTrue(arrays.keys.contains("model.layers.1.moe.gate.weight"))
        reference.update(parameters: ModuleParameters.unflattened(arrays))
        eval(reference)

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sharpai-airgap-step5-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try data.write(to: dir.appendingPathComponent("config.json"))
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))

        let loaded = try creator(data)
        try loadWeights(modelDirectory: dir, model: loaded)
        eval(loaded)

        let input = SharpAIAirgap.tokens(vocab: 32)
        let a = reference(input, cache: nil)
        let b = loaded(input, cache: nil)
        eval(a, b)
        XCTAssertEqual(a.shape, [1, 5, 32])
        XCTAssertEqual(a.shape, b.shape)
        XCTAssertTrue(all(isFinite(a)).item(Bool.self), "reference logits not finite")
        XCTAssertGreaterThan(abs(a).max().item(Float.self), 0)
        XCTAssertTrue(allClose(a, b, rtol: 1e-3, atol: 1e-4).item(Bool.self))
    }

    /// A full-attention layer (DSA) is refused at creation with a clear message.
    func testDSALayerConfigThrowsClearError() throws {
        let dsa = json.replacingOccurrences(
            of: "\"layer_types\": [\"sliding_attention\", \"sliding_attention\"],",
            with: """
                "layer_types": ["sliding_attention", "full_attention"],
                "sparse_config": {"enabled": true, "apply_to_layer_types": ["full_attention"]},
                """)
        let creator = try XCTUnwrap(SharpAIStep5Overlay.creators["step5"])
        XCTAssertThrowsError(try creator(Data(dsa.utf8))) { error in
            guard case SharpAIStep5Error.unsupported(let message) = error else {
                return XCTFail("expected unsupported, got \(error)")
            }
            XCTAssertTrue(message.contains("layer 1"), message)
            XCTAssertTrue(message.contains("DSA"), message)
        }
    }
}
