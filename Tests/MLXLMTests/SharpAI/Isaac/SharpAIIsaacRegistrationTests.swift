// SharpAI fork overlay tests for the isaac creator. Not part of upstream.
// Offline only: a tiny random-init model, no checkpoint is read.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIIsaacRegistrationTests: XCTestCase {

    private let tinyJSON = """
        {
          "model_type": "isaac",
          "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "patch_size": 2,
            "image_size": 4
          },
          "text_config": { "hidden_size": 8, "num_hidden_layers": 2, "vocab_size": 32 }
        }
        """

    /// Builds the tiny model through the registered creator, the same path as a real load.
    private func tinyModel() throws -> SharpAIIsaacModel {
        let creator = try XCTUnwrap(SharpAIIsaacOverlay.creators["isaac"])
        let built = try creator(Data(tinyJSON.utf8))
        return try XCTUnwrap(built as? SharpAIIsaacModel)
    }

    /// Checkpoint names for the tiny config, in the layout of the real Isaac-0.1 safetensors
    /// header (`model.vision_tower.*`, `model.multimodal_projector.*`, `model.text_model.*`).
    private func tinyCheckpointNames() -> [String] {
        var names = [
            "model.vision_tower.embeddings.patch_embedding.weight",
            "model.vision_tower.embeddings.patch_embedding.bias",
            "model.vision_tower.embeddings.position_embedding",
            "model.vision_tower.post_layernorm.weight",
            "model.vision_tower.post_layernorm.bias",
            "model.multimodal_projector.linear_1.weight",
            "model.multimodal_projector.linear_2.weight",
            "model.text_model.embed_tokens.weight",
            "model.text_model.norm.weight",
            "lm_head.weight",
        ]
        for layer in 0 ..< 2 {
            let vision = "model.vision_tower.encoder.layers.\(layer)"
            for norm in ["layer_norm1", "layer_norm2"] {
                names += ["\(vision).\(norm).weight", "\(vision).\(norm).bias"]
            }
            for fc in ["mlp.fc1", "mlp.fc2"] {
                names += ["\(vision).\(fc).weight", "\(vision).\(fc).bias"]
            }
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                names += ["\(vision).self_attn.\(proj).weight", "\(vision).self_attn.\(proj).bias"]
            }

            let text = "model.text_model.layers.\(layer)"
            names += [
                "\(text).input_layernorm.weight",
                "\(text).post_attention_layernorm.weight",
                "\(text).self_attn.q_norm.weight",
                "\(text).self_attn.k_norm.weight",
            ]
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                names.append("\(text).self_attn.\(proj).weight")
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                names.append("\(text).mlp.\(proj).weight")
            }
        }
        return names
    }

    func testCreatorIsRegisteredUnderIsaacKey() {
        XCTAssertNotNil(SharpAIIsaacOverlay.creators["isaac"])
    }

    func testCreatorRejectsInvalidConfigBeforeConstruction() {
        guard let creator = SharpAIIsaacOverlay.creators["isaac"] else {
            return XCTFail("isaac creator missing")
        }
        let bad = tinyJSON.replacingOccurrences(of: "\"isaac\"", with: "\"colpali\"")
        XCTAssertThrowsError(try creator(Data(bad.utf8))) { error in
            XCTAssertFalse(error is SharpAIIsaacError)
        }
    }

    func testCreatorBuildsModelForValidConfig() throws {
        _ = try tinyModel()
    }

    func testEndToEndTinyLoadRunsTextForward() throws {
        let model = try tinyModel()
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let cache = try model.newCache(parameters: nil)

        let logits = model(tokens, cache: cache)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testEndToEndTinyLoadRunsImagePathToTextHiddenSize() throws {
        let model = try tinyModel()

        // Grid 2x2 of patches, pixel shuffle 2 gives one token of text hidden size 8.
        let features = model.imageFeatures(MLXArray.zeros([1, 4, 12]))

        XCTAssertEqual(features.shape, [1, 1, 8])
    }

    func testTinyCheckpointNamesCoverEveryParameter() throws {
        let model = try tinyModel()
        var weights = [String: MLXArray]()
        for name in tinyCheckpointNames() {
            weights[name] = MLXArray.zeros([1])
        }

        let sanitized = try SharpAIIsaacSanitizer.sanitize(weights)
        let parameters = Set(model.parameters().flattened().map { $0.0 })

        XCTAssertEqual(Set(sanitized.keys), parameters)
    }
}
