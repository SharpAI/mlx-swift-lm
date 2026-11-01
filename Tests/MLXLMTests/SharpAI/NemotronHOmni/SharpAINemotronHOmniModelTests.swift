// SharpAI fork overlay tests for the nemotron_h_omni creator and model. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAINemotronHOmniModelTests: XCTestCase {

    /// Tiny text stack with one latent MoE layer (moe_latent_size set).
    private let tinyJSON = """
        {
            "model_type": "nemotron_h_omni",
            "llm_config": {
                "model_type": "nemotron_h",
                "vocab_size": 100,
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "mamba_num_heads": 4,
                "mamba_head_dim": 16,
                "ssm_state_size": 16,
                "conv_kernel": 4,
                "n_groups": 2,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "moe_shared_expert_intermediate_size": 64,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "n_group": 2,
                "topk_group": 1,
                "moe_latent_size": 16,
                "hybrid_override_pattern": "M*E"
            }
        }
        """

    private func makeModel() throws -> SharpAINemotronHOmniModel {
        let config = try JSONDecoder().decode(
            SharpAINemotronHOmniConfiguration.self, from: Data(tinyJSON.utf8))
        return SharpAINemotronHOmniModel(config)
    }

    // MARK: - Registration

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAINemotronHOmniOverlay.creators["nemotron_h_omni"])
    }

    func testArchStringIsNotUsedForDispatch() {
        XCTAssertNil(
            SharpAINemotronHOmniOverlay.creators["NemotronH_Omni_Reasoning_V3"])
    }

    func testCreatorBuildsModelFromValidConfig() throws {
        guard let creator = SharpAINemotronHOmniOverlay.creators["nemotron_h_omni"] else {
            XCTFail("nemotron_h_omni creator missing")
            return
        }

        let model = try creator(Data(tinyJSON.utf8))
        XCTAssertTrue(model is SharpAINemotronHOmniModel)
    }

    func testCreatorRejectsMissingLLMConfig() {
        guard let creator = SharpAINemotronHOmniOverlay.creators["nemotron_h_omni"] else {
            XCTFail("nemotron_h_omni creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{\"model_type\": \"nemotron_h_omni\"}".utf8)))
    }

    // MARK: - Sanitize through the model

    func testModelSanitizeKeepsTextUnderLanguageModelPrefix() throws {
        let model = try makeModel()
        let weights: [String: MLXArray] = [
            "language_model.lm_head.weight": MLXArray(Float(1.0)),
            "vision_model.radio.weight": MLXArray(Float(1.0)),
        ]

        let sanitized = try model.sanitize(weights: weights)

        XCTAssertEqual(Set(sanitized.keys), ["language_model.lm_head.weight"])
    }

    func testModelSanitizeKeepsLatentMoEProjectionKeys() throws {
        let model = try makeModel()
        let key = "language_model.backbone.layers.1.mixer.fc1_latent_proj.weight"
        let weights: [String: MLXArray] = [key: MLXArray(Float(1.0))]

        let sanitized = try model.sanitize(weights: weights)

        XCTAssertEqual(Set(sanitized.keys), [key])
    }

    // MARK: - Forward

    func testTinyForwardPassWithLatentMoEHasExpectedShape() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let cache = try model.newCache(parameters: nil)

        let logits = model.callAsFunction(tokens, cache: cache)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 100])
    }

    // MARK: - Known gap

    private func imageInput() -> LMInput {
        LMInput(
            text: .init(tokens: MLXArray([Int32(1), 2, 3]).reshaped([1, 3])),
            image: .init(pixels: MLXArray.zeros([1, 3, 16, 16])))
    }

    /// Runs now: the image path must fail with the explicit not-implemented error,
    /// not silently drop the image.
    func testImageInputThrowsVisionTowerNotImplemented() throws {
        let model = try makeModel()
        let cache = try model.newCache(parameters: nil)

        XCTAssertThrowsError(
            try model.prepare(
                imageInput(), cache: cache, state: nil, prefill: PrefillParameters())
        ) { error in
            XCTAssertEqual(error as? SharpAINemotronHOmniError, .visionTowerNotImplemented)
        }
    }

    func testVideoInputThrowsVisionTowerNotImplemented() throws {
        let model = try makeModel()
        let cache = try model.newCache(parameters: nil)
        let input = LMInput(
            text: .init(tokens: MLXArray([Int32(1), 2, 3]).reshaped([1, 3])),
            video: .init(pixels: MLXArray.zeros([1, 3, 16, 16])))

        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters())
        ) { error in
            XCTAssertEqual(error as? SharpAINemotronHOmniError, .visionTowerNotImplemented)
        }
    }

    func testAudioInputThrowsVisionTowerNotImplemented() throws {
        let model = try makeModel()
        let cache = try model.newCache(parameters: nil)
        let input = LMInput(
            text: .init(tokens: MLXArray([Int32(1), 2, 3]).reshaped([1, 3])),
            audio: .init(features: MLXArray.zeros([1, 8, 4])))

        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters())
        ) { error in
            XCTAssertEqual(error as? SharpAINemotronHOmniError, .visionTowerNotImplemented)
        }
    }

    /// KNOWN GAP: prepare cannot merge image features into the text embeddings yet. The Omni
    /// vision_config is C-RADIOv4-H (version c-radio_v4-h, cls_token_per_teacher with 4 teachers,
    /// register_multiple 10, cpe_max_size 2048), per
    /// https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/raw/main/config.json.
    /// The RADIOv4 modeling code (nvidia/C-RADIOv4-H hf_model.py, cls_token.py) has not been read,
    /// so the tower is not ported. Also missing: NemotronHModel exposes no inputs_embeds entry
    /// point (its backbone is private in Libraries/MLXLLM/Models/NemotronH.swift, outside this
    /// lane's allowed paths). This flips to a hard failure once both land.
    func testImageInputPrepareReturnsEmbeddings() throws {
        let model = try makeModel()
        let cache = try model.newCache(parameters: nil)

        try XCTExpectFailure(
            "KNOWN GAP: C-RADIOv4-H vision tower (cls_token.py and hf_model.py not read) and "
                + "NemotronHModel inputs_embeds entry point (backbone private in MLXLLM, out of lane)"
        ) {
            XCTAssertNoThrow(
                try model.prepare(
                    imageInput(), cache: cache, state: nil, prefill: PrefillParameters())
            )
        }
    }
}
