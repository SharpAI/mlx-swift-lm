// SharpAI fork overlay tests for the step3p7 configuration. Not part of upstream.
// Inline JSON copied from the step3p7 fixture (trimmed). No checkpoint, no network.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

enum SharpAIStep3p7Fixtures {
    static let fixtureJSON = """
        {
          "architectures": ["Step3p7ForConditionalGeneration"],
          "model_type": "step3p7",
          "image_token_id": 128001,
          "image_token_len": 169,
          "understand_projector_stride": 2,
          "projector_bias": false,
          "vision_select_layer": -1,
          "vision_config": {
            "model_type": "perception_encoder",
            "image_size": 728,
            "patch_size": 14,
            "width": 1536,
            "layers": 47,
            "heads": 16,
            "pool_type": "none",
            "output_dim": null,
            "use_cls_token": false,
            "ls_init_value": 0.1,
            "use_ln_post": false,
            "hidden_act": "quick_gelu"
          },
          "text_config": {
            "model_type": "step3p5",
            "hidden_size": 4096,
            "num_hidden_layers": 4,
            "vocab_size": 128896,
            "num_attention_heads": 64,
            "num_attention_groups": 8,
            "head_dim": 128,
            "moe_num_experts": 288,
            "moe_top_k": 8,
            "moe_intermediate_size": 1280,
            "share_expert_dim": 1280,
            "sliding_window": 512,
            "num_nextn_predict_layers": 3,
            "layer_types": [
              "full_attention", "sliding_attention", "sliding_attention", "sliding_attention"
            ]
          }
        }
        """
}

final class SharpAIStep3p7ConfigurationTests: XCTestCase {

    private func decode() throws -> SharpAIStep3p7Configuration {
        try JSONDecoder.json5().decode(
            SharpAIStep3p7Configuration.self, from: Data(SharpAIStep3p7Fixtures.fixtureJSON.utf8))
    }

    func testDecodesTopLevelKeys() throws {
        let config = try decode()

        XCTAssertEqual(config.modelType, "step3p7")
        XCTAssertEqual(config.architectures, ["Step3p7ForConditionalGeneration"])
        XCTAssertEqual(config.imageTokenId, 128001)
        XCTAssertEqual(config.imageTokenLen, 169)
        XCTAssertEqual(config.understandProjectorStride, 2)
        XCTAssertEqual(config.projectorBias, false)
        XCTAssertEqual(config.visionSelectLayer, -1)
    }

    func testDecodesVisionConfig() throws {
        let vision = try decode().visionConfig

        XCTAssertEqual(vision.modelType, "perception_encoder")
        XCTAssertEqual(vision.imageSize, 728)
        XCTAssertEqual(vision.patchSize, 14)
        XCTAssertEqual(vision.width, 1536)
        XCTAssertEqual(vision.layers, 47)
        XCTAssertEqual(vision.heads, 16)
        XCTAssertNil(vision.outputDim)
        XCTAssertEqual(vision.hiddenAct, "quick_gelu")
    }

    func testDecodesTextConfig() throws {
        let text = try decode().textConfig

        XCTAssertEqual(text.modelType, "step3p5")
        XCTAssertEqual(text.hiddenSize, 4096)
        XCTAssertEqual(text.hiddenLayers, 4)
        XCTAssertEqual(text.attentionHeads, 64)
        XCTAssertEqual(text.attentionGroups, 8)
        XCTAssertEqual(text.moeNumExperts, 288)
        XCTAssertEqual(text.moeTopK, 8)
        XCTAssertEqual(text.slidingWindow, 512)
        XCTAssertEqual(text.numNextnPredictLayers, 3)
        XCTAssertEqual(text.layerTypes?.count, 4)
    }

    func testValidFixtureConfigPassesValidation() throws {
        XCTAssertNoThrow(try decode().validateModelConfiguration())
    }

    func testValidationRejectsOtherArchitecture() throws {
        let json = SharpAIStep3p7Fixtures.fixtureJSON.replacingOccurrences(
            of: "Step3p7ForConditionalGeneration", with: "LlamaForCausalLM")
        let config = try JSONDecoder.json5().decode(
            SharpAIStep3p7Configuration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsLayerTypeCountMismatch() throws {
        let json = SharpAIStep3p7Fixtures.fixtureJSON.replacingOccurrences(
            of: "\"num_hidden_layers\": 4", with: "\"num_hidden_layers\": 5")
        let config = try JSONDecoder.json5().decode(
            SharpAIStep3p7Configuration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsImageSizeNotMultipleOfPatch() throws {
        let json = SharpAIStep3p7Fixtures.fixtureJSON.replacingOccurrences(
            of: "\"image_size\": 728", with: "\"image_size\": 730")
        let config = try JSONDecoder.json5().decode(
            SharpAIStep3p7Configuration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testMissingTextConfigFailsDecoding() {
        let json = #"{"model_type": "step3p7", "vision_config": {"image_size": 728, "patch_size": 14, "width": 1, "layers": 1, "heads": 1}}"#

        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIStep3p7Configuration.self, from: Data(json.utf8)))
    }
}
