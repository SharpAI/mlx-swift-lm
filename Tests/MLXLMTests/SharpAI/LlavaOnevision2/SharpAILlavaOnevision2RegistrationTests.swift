// SharpAI fork overlay tests for the llava_onevision2 creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAILlavaOnevision2RegistrationTests: XCTestCase {

    /// The 8B checkpoint shape (config.json of lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct).
    private let validJSON = """
        {
            "model_type": "llava_onevision2",
            "architectures": ["LlavaOnevision2ForConditionalGeneration"],
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
            "text_config": {
                "model_type": "qwen3",
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rms_norm_eps": 1e-06,
                "vocab_size": 151936,
                "rope_parameters": { "rope_theta": 8000000, "rope_type": "default" }
            },
            "vision_config": {
                "model_type": "onevision_encoder",
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "image_size": 448,
                "patch_size": 14,
                "num_channels": 3,
                "spatial_merge_size": 2,
                "out_hidden_size": 4096,
                "frame_windows_size": 4,
                "tokens_per_second": 1,
                "hidden_act": "gelu",
                "layer_norm_type": "layer_norm",
                "layer_norm_eps": 1e-06,
                "text_hidden_size": 4096,
                "use_head": false,
                "rope_theta": 10000.0
            }
        }
        """

    /// Same field layout at a size that builds and runs on a test machine. Vision head_dim is 32
    /// (a multiple of 32 for the 4:6:6 RoPE split), the text head_dim is 16, and the 56x56 image
    /// gives a 4x4 patch grid, which merges to 4 tokens.
    private let tinyJSON = """
        {
            "model_type": "llava_onevision2",
            "architectures": ["LlavaOnevision2ForConditionalGeneration"],
            "image_token_id": 200,
            "video_token_id": 201,
            "vision_start_token_id": 202,
            "vision_end_token_id": 203,
            "text_config": {
                "model_type": "qwen3",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "rms_norm_eps": 1e-06,
                "vocab_size": 300,
                "rope_parameters": { "rope_theta": 10000, "rope_type": "default" }
            },
            "vision_config": {
                "model_type": "onevision_encoder",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "image_size": 56,
                "patch_size": 14,
                "num_channels": 3,
                "spatial_merge_size": 2,
                "out_hidden_size": 64,
                "frame_windows_size": 4,
                "tokens_per_second": 1,
                "hidden_act": "gelu",
                "layer_norm_type": "layer_norm",
                "layer_norm_eps": 1e-06,
                "text_hidden_size": 64,
                "use_head": false
            }
        }
        """

    private func tinyModel() throws -> SharpAILlavaOnevision2Model {
        guard let creator = SharpAILlavaOnevision2Overlay.creators["llava_onevision2"] else {
            throw ModelFactoryError.invalidConfiguration("llava_onevision2 creator missing")
        }
        let model = try creator(Data(tinyJSON.utf8))
        guard let typed = model as? SharpAILlavaOnevision2Model else {
            throw ModelFactoryError.invalidConfiguration("creator did not return the llava_onevision2 model")
        }
        return typed
    }

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAILlavaOnevision2Overlay.creators["llava_onevision2"])
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAILlavaOnevision2Overlay.creators["llava_onevision2"] else {
            XCTFail("llava_onevision2 creator missing")
            return
        }

        let model = try creator(Data(tinyJSON.utf8))

        XCTAssertTrue(model is SharpAILlavaOnevision2Model)
    }

    func testWrongModelTypeFailsValidationBeforeConstruction() throws {
        guard let creator = SharpAILlavaOnevision2Overlay.creators["llava_onevision2"] else {
            XCTFail("llava_onevision2 creator missing")
            return
        }
        let json = validJSON.replacingOccurrences(
            of: "\"model_type\": \"llava_onevision2\",", with: "\"model_type\": \"llava\",")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testIncompleteConfigFailsDecodingBeforeConstruction() throws {
        guard let creator = SharpAILlavaOnevision2Overlay.creators["llava_onevision2"] else {
            XCTFail("llava_onevision2 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 64 }".utf8))) { error in
            XCTAssertTrue(error is DecodingError)
        }
    }

    func testTextOnlyForwardProducesVocabularyLogits() throws {
        let model = try tinyModel()
        let tokens = MLXArray([Int32(5), 6, 7]).reshaped(1, 3)
        let cache = try model.newCache(parameters: nil)

        let logits = model(tokens, cache: cache)

        XCTAssertEqual(logits.shape, [1, 3, 300])
    }

    func testImageFeaturesSpliceIntoPlaceholderTokens() throws {
        let model = try tinyModel()
        // Two text tokens, four image placeholders (a 4x4 grid merges to 4 tokens), one text token.
        let tokens = MLXArray([Int32(5), 200, 200, 200, 200, 7]).reshaped(1, 6)
        let input = LMInput(
            text: .init(tokens: tokens),
            image: LMInput.ProcessedImage(
                pixels: MLXArray.zeros([16, 3, 14, 14]),
                frames: [THW(1, 4, 4)]))
        let cache = try model.newCache(parameters: nil)

        let result = try model.prepare(
            input, cache: cache, state: nil, prefill: PrefillParameters())

        guard case .logits(let output) = result else {
            XCTFail("expected logits from prepare")
            return
        }
        XCTAssertEqual(output.logits.shape, [1, 6, 300])
    }

    func testPlaceholderCountMismatchThrows() throws {
        let model = try tinyModel()
        // Three placeholders but four merged image features.
        let tokens = MLXArray([Int32(5), 200, 200, 200, 7]).reshaped(1, 5)
        let input = LMInput(
            text: .init(tokens: tokens),
            image: LMInput.ProcessedImage(
                pixels: MLXArray.zeros([16, 3, 14, 14]),
                frames: [THW(1, 4, 4)]))
        let cache = try model.newCache(parameters: nil)

        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters())
        ) { error in
            guard case SharpAILlavaOnevision2Error.imageTokenMismatch(let placeholders, let features) = error
            else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(placeholders, 3)
            XCTAssertEqual(features, 4)
        }
    }
}
