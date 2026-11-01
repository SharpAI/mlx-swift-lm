// SharpAI fork overlay tests for the isaac configuration. Not part of upstream.
// Offline only: the config is decoded from an inline JSON string.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIIsaacConfigurationTests: XCTestCase {

    private let tinyJSON = """
        {
          "model_type": "isaac",
          "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_channels": 3,
            "patch_size": 2,
            "image_size": 4,
            "layer_norm_eps": 1e-6,
            "hidden_act": "gelu_pytorch_tanh"
          },
          "text_config": {
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "vocab_size": 32
          }
        }
        """

    private func decode(_ json: String) throws -> SharpAIIsaacConfiguration {
        try JSONDecoder.json5().decode(SharpAIIsaacConfiguration.self, from: Data(json.utf8))
    }

    private func replacing(_ old: String, with new: String) -> String {
        tinyJSON.replacingOccurrences(of: old, with: new)
    }

    func testDecodesVisionAndTextFields() throws {
        let config = try decode(tinyJSON)
        XCTAssertEqual(config.modelType, "isaac")
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 8)
        XCTAssertEqual(config.visionConfiguration.gridSide, 2)
        XCTAssertEqual(config.visionConfiguration.numPatches, 4)
        XCTAssertEqual(config.visionConfiguration.patchInputSize, 12)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 32)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testDefaultsFillMissingVisionFields() throws {
        let json = """
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
        let config = try decode(json)
        XCTAssertEqual(config.visionConfiguration.channels, 3)
        XCTAssertEqual(config.visionConfiguration.layerNormEps, 1e-6, accuracy: 1e-12)
        XCTAssertEqual(config.visionConfiguration.hiddenAct, "gelu_pytorch_tanh")
    }

    func testRejectsWrongModelType() throws {
        let config = try decode(replacing("\"isaac\"", with: "\"colpali\""))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsNumPatchesThatAreNotTheGrid() throws {
        let config = try decode(
            replacing("\"image_size\": 4,", with: "\"image_size\": 4, \"num_patches\": 5,"))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsUnsupportedActivation() throws {
        let config = try decode(replacing("gelu_pytorch_tanh", with: "relu"))
        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            XCTAssertEqual(
                (error as? SharpAIIsaacError)?.errorDescription,
                SharpAIIsaacError.unsupportedActivation("relu").errorDescription)
        }
    }
}
