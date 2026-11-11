// SharpAI fork overlay tests for jina_embeddings_v5 adapter loading and embed. Not part of upstream.
// Random and synthetic weights only, no network.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIJinaEmbeddingsV5LoadTests: XCTestCase {

    private let tinyJSON = """
        {
          "model_type": "jina_embeddings_v5",
          "hidden_size": 8,
          "num_hidden_layers": 1,
          "intermediate_size": 16,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 4,
          "rms_norm_eps": 1e-06,
          "vocab_size": 32,
          "tie_word_embeddings": true
        }
        """

    private func tinyModel() throws -> SharpAIJinaEmbeddingsV5Model {
        let config = try JSONDecoder.json5().decode(
            SharpAIJinaEmbeddingsV5Configuration.self, from: Data(tinyJSON.utf8))
        return SharpAIJinaEmbeddingsV5Model(config)
    }

    /// A temporary model directory with `adapters/retrieval/` written as the checkpoint ships it.
    /// The adapter adds `B @ A` to `layers.0.self_attn.q_proj` with scaling 2 (alpha 2, r 1).
    private func makeModelDirectory(adapterKeyPrefix: String = "base_model.model.") throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(component: "jina-v5-\(UUID().uuidString)")
        let adapter = root.appending(component: "adapters").appending(component: "retrieval")
        try FileManager.default.createDirectory(at: adapter, withIntermediateDirectories: true)
        try Data(#"{ "r": 1, "lora_alpha": 2 }"#.utf8)
            .write(to: adapter.appending(component: "adapter_config.json"))

        // A: [r, in] = [1, 8] selects input column 0. B: [out, r] = [8, 1] writes row 0.
        let a = MLXArray((0 ..< 8).map { $0 == 0 ? Float(1) : Float(0) }).reshaped(1, 8)
        let b = MLXArray((0 ..< 8).map { $0 == 0 ? Float(1) : Float(0) }).reshaped(8, 1)
        try MLX.save(
            arrays: [
                "\(adapterKeyPrefix)layers.0.self_attn.q_proj.lora_A.weight": a,
                "\(adapterKeyPrefix)layers.0.self_attn.q_proj.lora_B.weight": b,
            ],
            url: adapter.appending(component: "adapter_model.safetensors"))
        return root
    }

    private func baseCheckpoint(directory: URL?) -> ModelCheckpoint {
        var checkpoint = ModelCheckpoint(weights: [
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
            "model.norm.weight": MLXArray.ones([8]),
        ])
        checkpoint.modelDirectory = directory
        return checkpoint
    }

    func testPrepareCheckpointMergesShippedAdapter() throws {
        let root = try makeModelDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let prepared = try tinyModel().prepareCheckpoint(baseCheckpoint(directory: root))

        // The merge adds scaling * B @ A = 2 * (row 0 gets 1 at column 0), so element [0, 0] is 2.
        let q = prepared.weights["model.layers.0.self_attn.q_proj.weight"]!
        XCTAssertEqual(q[0, 0].item(Float.self), 2)
        XCTAssertEqual(q[1, 0].item(Float.self), 0)
        // The LoRA tensors are dropped by sanitize, after the merge has read them.
        XCTAssertNil(prepared.weights["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"])
        XCTAssertEqual(prepared.weights["model.norm.weight"]!.asArray(Float.self), [1, 1, 1, 1, 1, 1, 1, 1])
    }

    func testPrepareCheckpointWithoutAdaptersLeavesWeightsAlone() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(component: "jina-v5-noadapter-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let prepared = try tinyModel().prepareCheckpoint(baseCheckpoint(directory: root))

        XCTAssertEqual(
            prepared.weights["model.layers.0.self_attn.q_proj.weight"]![0, 0].item(Float.self), 0)
    }

    func testPrepareCheckpointWithoutModelDirectoryIsUnchanged() throws {
        let prepared = try tinyModel().prepareCheckpoint(baseCheckpoint(directory: nil))
        XCTAssertNotNil(prepared.weights["model.norm.weight"])
    }

    func testAdapterMatchingNoWeightThrows() throws {
        let root = try makeModelDirectory(adapterKeyPrefix: "base_model.model.unrelated.")
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(
            try SharpAIJinaEmbeddingsV5Adapter.applyTaskAdapter(
                root: root, task: "retrieval", to: ["model.norm.weight": MLXArray.ones([8])])
        ) { error in
            XCTAssertEqual(
                error as? SharpAIJinaEmbeddingsV5Error,
                .noMatchingAdapterPairs(task: "retrieval"))
        }
    }

    func testEmbedIsLastTokenAndUnitNorm() throws {
        let model = try tinyModel()
        let tokens = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let embedding = model.embed(tokens)
        XCTAssertEqual(embedding.shape, [1, 8])

        let values = embedding.asArray(Float.self)
        let norm = values.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        XCTAssertEqual(norm, 1, accuracy: 1e-4)

        // The last-token hidden state, normalized, is what embed returns.
        let hidden = model.model(tokens, cache: nil)[0..., -1, 0...]
        let expected = hidden / MLX.sqrt(MLX.sum(hidden * hidden, axis: -1, keepDims: true))
        let difference = MLX.abs(embedding - expected).max().item(Float.self)
        XCTAssertLessThan(difference, 1e-5)
    }
}
