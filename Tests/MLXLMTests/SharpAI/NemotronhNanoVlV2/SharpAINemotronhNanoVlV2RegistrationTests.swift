// SharpAI fork overlay tests for the NemotronH_Nano_VL_V2 creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAINemotronhNanoVlV2RegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "NemotronH_Nano_VL_V2",
            "llm_config": {
                "model_type": "nemotron_h",
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "mamba_num_heads": 8,
                "mamba_head_dim": 32,
                "ssm_state_size": 16,
                "conv_kernel": 4,
                "n_groups": 2,
                "intermediate_size": 512,
                "hybrid_override_pattern": "M*"
            }
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAINemotronhNanoVlV2Overlay.creators["NemotronH_Nano_VL_V2"])
    }

    func testValidConfigFailsUntilForwardIsImplemented() throws {
        guard let creator = SharpAINemotronhNanoVlV2Overlay.creators["NemotronH_Nano_VL_V2"] else {
            XCTFail("NemotronH_Nano_VL_V2 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard
                case SharpAINemotronhNanoVlV2Error.forwardNotImplemented(let type) = error
            else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "NemotronH_Nano_VL_V2")
        }
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAINemotronhNanoVlV2Overlay.creators["NemotronH_Nano_VL_V2"] else {
            XCTFail("NemotronH_Nano_VL_V2 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 64 }".utf8))) { error in
            if case SharpAINemotronhNanoVlV2Error.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }

    /// Tiny RADIO ViT (2 blocks, 32 wide, 8 px patches, 64 px input, 16x16 CPE grid so the
    /// interpolation path runs) feeding a tiny NemotronH text stack.
    func testTinyForwardPass() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAINemotronhNanoVlV2Configuration.self, from: Data(validJSON.utf8))
        let vision = SharpAINemotronhNanoVlV2VisionConfiguration(
            patchSize: 8, embedDim: 32, depth: 2, numHeads: 4, mlpHiddenSize: 64,
            cpeMaxSize: 128, numClsTokens: 4, registerMultiple: 16, layerNormEps: 1e-6)
        let model = SharpAINemotronhNanoVlV2Model(
            text: config.textConfiguration, vision: vision, projectorHiddenSize: 64)
        let pixels = MLXRandom.normal([1, 3, 64, 64])

        // 4 cls + 12 registers are dropped; 8x8 patches pixel-shuffle to 4x4 image tokens.
        XCTAssertEqual(vision.numSkipTokens, 16)
        let features = model.visionModel(pixels)
        eval(features)
        XCTAssertEqual(features.shape, [1, 64, 32])

        let image = model.imageFeatures(pixels)
        eval(image)
        XCTAssertEqual(image.shape, [1, 16, 256])

        let tokens = MLXArray([Int32(1), 2, 3]).reshaped([1, 3])
        let cache = try model.newCache(parameters: nil)
        let logits = model.callAsFunction(tokens, cache: cache)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, 1024])

        // KNOWN GAP: the reference (nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 modeling.py) builds
        // inputs_embeds from input_ids, scatters vit_embeds at img_context_token_id, then calls
        // language_model(inputs_embeds=...). NemotronHModel has no such entry point; its backbone
        // is private in Libraries/MLXLLM/Models/NemotronH.swift, outside this lane's allowed paths.
        try XCTExpectFailure(
            "KNOWN GAP: NemotronHModel needs a public inputs_embeds entry point (reference: "
                + "NemotronH_Nano_VL_V2.forward in modeling.py); NemotronHBackbone is private in "
                + "Libraries/MLXLLM/Models/NemotronH.swift, outside this lane's allowed paths"
        ) {
            XCTAssertNoThrow(try model.forwardWithImages(tokens, pixelValues: pixels))
        }
    }
}
