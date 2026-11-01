// SharpAI fork overlay tests for exaone_moe: clamp helper, tiny forward passes and the
// MTP predictor. Weights are random and tiny. No checkpoint is loaded. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIExaoneMoeModelTests: XCTestCase {

    /// Sliding layer 0 (dense, clamped), full layer 1 (sparse, clamped routed experts).
    private let tinyJSON = """
        {
            "vocab_size": 64,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 4,
            "mlp_layer_types": ["dense", "sparse"],
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "num_shared_experts": 1,
            "n_group": 2,
            "topk_group": 1,
            "swiglu_limits": [2.0, 2.0],
            "num_nextn_predict_layers": 1,
            "mtp_layer_types": ["full_attention"]
        }
        """

    private func makeConfig() throws -> SharpAIExaoneMoeConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIExaoneMoeConfiguration.self, from: Data(tinyJSON.utf8))
    }

    func testClampedSwiGLUMatchesFormula() {
        // silu(min(gate, 2)) * clip(up, -2, 2)
        let gate = MLXArray([Float(5), -5, 0.5])
        let up = MLXArray([Float(10), -10, 2])

        let out = sharpAIExaoneMoeClampedSwiGLU(gate: gate, up: up, limit: 2).asArray(Float.self)

        XCTAssertEqual(out.count, 3)
        XCTAssertEqual(out[0], 3.5231884, accuracy: 1e-4)
        XCTAssertEqual(out[1], 0.0669288, accuracy: 1e-4)
        XCTAssertEqual(out[2], 0.6224593, accuracy: 1e-4)
    }

    func testTinyForwardPassShape() throws {
        let model = SharpAIExaoneMoeModel(try makeConfig())
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5)

        // 5 tokens exceed the window of 4, so the sliding layer gets a windowed mask.
        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 5, 64])
    }

    func testTinyForwardPassWithCache() throws {
        let model = SharpAIExaoneMoeModel(try makeConfig())
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5)

        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 2)

        let logits = model(tokens, cache: cache)

        XCTAssertEqual(logits.shape, [1, 5, 64])
    }

    func testMTPPredictorStepShape() throws {
        let config = try makeConfig()
        let predictor = SharpAIExaoneMoeMTPPredictor(config)
        let embedding = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5)
        let hidden = MLXArray.zeros([1, 5, config.hiddenSize])

        let out = predictor(tokens, hidden: hidden, embedding: embedding)

        XCTAssertEqual(out.shape, [1, 5, config.hiddenSize])
    }

    func testMTPExtractWeightsKeepsOnlyMTPKeysWithoutPrefix() {
        let weights: [String: MLXArray] = [
            "mtp.fc.weight": MLXArray.zeros([2]),
            "mtp.layers.0.self_attn.q_proj.weight": MLXArray.zeros([2]),
            "model.embed_tokens.weight": MLXArray.zeros([2]),
        ]

        let result = SharpAIExaoneMoeMTPPredictor.extractWeights(weights)

        XCTAssertEqual(Set(result.keys), ["fc.weight", "layers.0.self_attn.q_proj.weight"])
    }
}
