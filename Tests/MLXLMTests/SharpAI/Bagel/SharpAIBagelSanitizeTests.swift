// SharpAI fork overlay tests for bagel key mapping. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
@testable import MLXVLM
import XCTest

// UNVERIFIED: key names and the 4-D patch-embedding shape are the
// writer's assumptions, not checked against a Bagel checkpoint. These tests only confirm that
// sanitize matches those assumptions. They do not prove load compatibility.
final class SharpAIBagelSanitizeTests: XCTestCase {

    private func makeModel() throws -> SharpAIBagel {
        let config = try JSONDecoder.json5().decode(
            SharpAIBagelConfiguration.self,
            from: Data(SharpAIBagelTestFixtures.tinyJSON.utf8))
        return SharpAIBagel(config)
    }

    func testKeepsUnderstandingWeightsWithHFNames() throws {
        let model = try makeModel()
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([32, 16]),
            "language_model.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([16, 16]),
            "language_model.model.layers.0.self_attn.q_norm.weight": MLXArray.zeros([4]),
            "language_model.model.layers.0.mlp.gate_proj.weight": MLXArray.zeros([32, 16]),
            "language_model.model.norm.weight": MLXArray.zeros([16]),
            "language_model.lm_head.weight": MLXArray.zeros([32, 16]),
            "vit_model.vision_model.encoder.layers.0.self_attn.q_proj.weight": MLXArray.zeros([
                8, 8,
            ]),
            "vit_model.vision_model.post_layernorm.weight": MLXArray.zeros([8]),
            "connector.fc1.weight": MLXArray.zeros([16, 8]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.self_attn.q_proj.weight"]?.shape, [16, 16])
    }

    func testDropsGenerationPathAndPositionKeys() throws {
        let model = try makeModel()
        let weights: [String: MLXArray] = [
            "language_model.model.layers.0.self_attn.q_proj_moe_gen.weight": MLXArray.zeros([
                16, 16,
            ]),
            "language_model.model.layers.0.mlp_moe_gen.gate_proj.weight": MLXArray.zeros([32, 16]),
            "language_model.model.norm_moe_gen.weight": MLXArray.zeros([16]),
            "vae2llm.weight": MLXArray.zeros([16, 4]),
            "llm2vae.weight": MLXArray.zeros([4, 16]),
            "time_embedder.mlp.0.weight": MLXArray.zeros([16, 16]),
            "latent_pos_embed.pos_embed": MLXArray.zeros([4, 16]),
            "vit_model.vision_model.embeddings.position_ids": MLXArray.zeros([1, 4]),
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "some_unknown.weight": MLXArray.zeros([2]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertTrue(sanitized.isEmpty, "unexpected keys: \(Array(sanitized.keys))")
    }

    func testKeepsVitPosEmbed() throws {
        let model = try makeModel()
        let weights: [String: MLXArray] = [
            "vit_pos_embed.pos_embed": MLXArray.zeros([4, 16])
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(sanitized["vit_pos_embed.pos_embed"]?.shape, [4, 16])
    }

    func testVitPosEmbedIsAddedAfterConnector() throws {
        let model = try makeModel()
        let pixels = MLXArray.zeros([1, 3, 28, 28])
        let table = MLXArray(0 ..< 64).reshaped(4, 16).asType(.float32)
        model.vitPosEmbed.update(parameters: ModuleParameters.unflattened(["pos_embed": table]))
        let after = try model.imageFeatures(pixels)
        // The table must be added to the connector output, not fed through the connector.
        let nhwc = pixels.transposed(0, 2, 3, 1)
        let expected = model.connector(model.vitModel(nhwc)) + table.reshaped(1, 4, 16)

        let actual = after.asArray(Float.self)
        let want = expected.asArray(Float.self)
        XCTAssertEqual(actual.count, want.count)
        for (a, b) in zip(actual, want) {
            XCTAssertEqual(a, b, accuracy: 1e-3)
        }
    }

    func testTransposesPyTorchPatchEmbeddingToMLXLayout() throws {
        let model = try makeModel()
        // PyTorch layout (out=8, in=3, kH=14, kW=14).
        let weights: [String: MLXArray] = [
            "vit_model.vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([
                8, 3, 14, 14,
            ])
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(
            sanitized["vit_model.vision_model.embeddings.patch_embedding.weight"]?.shape,
            [8, 14, 14, 3])
    }

    func testDropsLMHeadWhenEmbeddingsAreTied() throws {
        let json = SharpAIBagelTestFixtures.tinyJSON.replacingOccurrences(
            of: "\"tie_word_embeddings\": false", with: "\"tie_word_embeddings\": true")
        let config = try JSONDecoder.json5().decode(
            SharpAIBagelConfiguration.self, from: Data(json.utf8))
        let model = SharpAIBagel(config)
        let weights: [String: MLXArray] = [
            "language_model.lm_head.weight": MLXArray.zeros([32, 16])
        ]

        XCTAssertTrue(model.sanitize(weights: weights).isEmpty)
    }
}
