// SharpAI fork overlay tests for the EAGLE Cohere draft head. Not part of upstream.
// Offline only: tiny shapes, default (random) weights, no checkpoint. UNVERIFIED: the
// fc input order (embedding first, then hidden state) follows vLLM cohere_eagle.py.
// Written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class SharpAIUnknownEagleCohereModelTests: XCTestCase {

    private func tinyConfiguration(tied: Bool = true) throws -> SharpAIEagleCohereConfiguration {
        let json = """
            {
                "architectures": ["EagleCohereForCausalLM"],
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "layer_norm_eps": 1e-05,
                "vocab_size": 100,
                "logit_scale": 1.0,
                "tie_word_embeddings": \(tied ? "true" : "false")
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIEagleCohereConfiguration.self, from: Data(json.utf8))
    }

    func testForwardProducesHiddenShape() throws {
        let model = SharpAIEagleCohereDraftModel(try tinyConfiguration().base)
        let ids = MLXArray([0, 1, 2, 3] as [Int32]).reshaped(1, 4)
        let hidden = MLXArray.zeros([1, 4, 32])

        let out = model(ids, hiddenStates: hidden, cache: nil)
        eval(out)

        XCTAssertEqual(out.shape, [1, 4, 32])
    }

    func testLogitsUseDraftEmbeddingWhenNoHeadGiven() throws {
        let model = SharpAIEagleCohereDraftModel(try tinyConfiguration().base)
        let hidden = MLXArray.zeros([1, 4, 32])

        let logits = model.logits(hidden)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 4, 100])
    }

    func testLogitsUseProvidedTargetHead() throws {
        let model = SharpAIEagleCohereDraftModel(try tinyConfiguration().base)
        let head = Embedding(embeddingCount: 100, dimensions: 32)
        let hidden = MLXArray.zeros([1, 2, 32])

        let logits = model.logits(hidden, head: head)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 2, 100])
    }

    func testParameterNamesMatchDraftLayout() throws {
        let model = SharpAIEagleCohereDraftModel(try tinyConfiguration().base)
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("fc.weight"))
        XCTAssertTrue(keys.contains("fc.bias"))
        XCTAssertTrue(keys.contains("embed_tokens.weight"))
        XCTAssertTrue(keys.contains("norm.weight"))
        XCTAssertTrue(keys.contains("layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(keys.contains("layers.1.mlp.down_proj.weight"))
    }
}
