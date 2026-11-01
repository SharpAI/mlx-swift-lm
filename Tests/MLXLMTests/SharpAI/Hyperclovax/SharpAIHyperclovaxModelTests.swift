// SharpAI fork overlay tests for the hyperclovax model structure, forward pass and creator.
// Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIHyperclovaxModelTests: XCTestCase {

    private func tinyConfiguration(postNorm: Bool = true, tied: Bool = false,
        logitsScaling: Double = 2.0)
        throws -> SharpAIHyperclovaxConfiguration
    {
        let json: [String: Any] = [
            "model_type": "hyperclovax",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "tie_word_embeddings": tied,
            "attention_multiplier": 0.25,
            "embedding_multiplier": 2.0,
            "residual_multiplier": 0.5,
            "logits_scaling": logitsScaling,
            "use_post_norm": postNorm,
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder.json5().decode(SharpAIHyperclovaxConfiguration.self, from: data)
    }

    func testParameterKeysFollowCheckpointNames() throws {
        let model = SharpAIHyperclovaxModel(try tinyConfiguration())
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("model.embed_tokens.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.norm.weight"))
        XCTAssertTrue(keys.contains("lm_head.weight"))
        // UNVERIFIED: pins the guessed post norm names from the model file, not the checkpoint.
        // Confirm against the real weight index before trusting this test.
        XCTAssertTrue(keys.contains("model.layers.0.post_attn_layernorm.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.post_mlp_layernorm.weight"))
    }

    func testPostNormKeysAbsentWhenDisabled() throws {
        let model = SharpAIHyperclovaxModel(try tinyConfiguration(postNorm: false))
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertFalse(keys.contains(where: { $0.contains("post_attn_layernorm") }))
        XCTAssertFalse(keys.contains(where: { $0.contains("post_mlp_layernorm") }))
        XCTAssertNil(model.model.layers[0].postAttnLayerNorm)
    }

    func testTiedConfigurationOmitsLMHead() throws {
        let model = SharpAIHyperclovaxModel(try tinyConfiguration(tied: true))
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertFalse(keys.contains(where: { $0.hasPrefix("lm_head") }))
    }

    func testAttentionUsesConfiguredMultiplierAsScale() throws {
        let model = SharpAIHyperclovaxModel(try tinyConfiguration())

        XCTAssertEqual(model.model.layers[0].attention.scale, 0.25)
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = SharpAIHyperclovaxModel(try tinyConfiguration())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 64])
    }

    func testLogitsAreMultipliedByLogitsScaling() throws {
        let scaled = SharpAIHyperclovaxModel(try tinyConfiguration(logitsScaling: 0.125))
        let reference = SharpAIHyperclovaxModel(try tinyConfiguration(logitsScaling: 1.0))
        reference.update(parameters: scaled.parameters())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let scaledLogits = scaled(inputs, cache: nil)
        let referenceLogits = reference(inputs, cache: nil)
        eval(scaledLogits, referenceLogits)

        XCTAssertTrue(
            allClose(scaledLogits, referenceLogits * MLXArray(Float(0.125)), atol: 1e-5)
                .item(Bool.self))
    }

    func testCreatorBuildsModelFromConfigData() throws {
        let json = """
            {
                "model_type": "hyperclovax",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "vocab_size": 64
            }
            """
        let creator = try XCTUnwrap(SharpAIHyperclovaxOverlay.creators["hyperclovax"])

        let model = try creator(Data(json.utf8))

        XCTAssertTrue(model is SharpAIHyperclovaxModel)
    }
}
