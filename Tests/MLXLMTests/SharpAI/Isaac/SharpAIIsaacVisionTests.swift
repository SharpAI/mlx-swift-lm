// SharpAI fork overlay tests for the isaac Siglip2 vision tower. Not part of upstream.
// One tiny-shape forward pass with the module's default random init. No checkpoint is read.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIIsaacVisionTests: XCTestCase {

    private let tinyVisionJSON = """
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
            "hidden_act": "gelu_pytorch_tanh"
          },
          "text_config": { "hidden_size": 8, "num_hidden_layers": 2, "vocab_size": 32 }
        }
        """

    func testForwardKeepsPatchCountAndHiddenSize() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIIsaacConfiguration.self, from: Data(tinyVisionJSON.utf8))
        let vision = config.visionConfiguration
        let model = SharpAIIsaacVisionModel(vision)

        let patches = MLXArray.zeros([2, vision.numPatches, vision.patchInputSize])
        let output = model(patches)

        XCTAssertEqual(output.shape, [2, 4, 8])
    }

    /// Gather order from pixel_shuffle_varlen / create_pixel_shuffle_index_map: a 4x4 grid
    /// with scale 2 gives four tokens; token (a, b) takes patches (2a+i, 2b+j) in block i*2+j.
    func testPixelShuffleGathersSubPatchesInHFOrder() {
        let input = MLXArray((0 ..< 16).map { Float($0) }).reshaped(1, 16, 1)

        let output = SharpAIIsaacPixelShuffle.shuffle(input, gridSide: 4, scale: 2)

        XCTAssertEqual(output.shape, [1, 4, 4])
        XCTAssertEqual(output[0, 0].asArray(Float.self), [0, 1, 4, 5])
        XCTAssertEqual(output[0, 1].asArray(Float.self), [2, 3, 6, 7])
        XCTAssertEqual(output[0, 2].asArray(Float.self), [8, 9, 12, 13])
        XCTAssertEqual(output[0, 3].asArray(Float.self), [10, 11, 14, 15])
    }
}
