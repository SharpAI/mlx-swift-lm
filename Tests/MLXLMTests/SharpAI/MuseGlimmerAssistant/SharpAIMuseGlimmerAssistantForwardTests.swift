// SharpAI fork overlay tests for the muse_glimmer_assistant forward pass. Not part of upstream.
//
// Offline only: a tiny synthetic config, random weights, no checkpoint. These checks cover shapes,
// finiteness, weight key layout, and that the target context reaches the output.
// UNVERIFIED: no reference output is available in this repo, so numerics are not compared.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXVLM

final class SharpAIMuseGlimmerAssistantForwardTests: XCTestCase {

    /// Hidden 16, two target layers (context width 32), and one full plus one sliding layer.
    /// sliding_window is 4, which is shorter than the 4 + 6 = 10 positions, so the sliding mask
    /// excludes some keys.
    private let tinyJSON = """
        {
          "model_type": "muse_glimmer_assistant",
          "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
          "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
          "layer_types": ["full_attention", "sliding_attention"],
          "sliding_window": 4,
          "target_layer_ids": [0, 1],
          "block_size": 4, "mask_token_id": 9
        }
        """

    private func tinyModel() throws -> SharpAIMuseGlimmerAssistantModel {
        let configuration = try JSONDecoder().decode(
            SharpAIMuseGlimmerAssistantConfiguration.self, from: Data(tinyJSON.utf8))
        try configuration.validateModelConfiguration()
        return SharpAIMuseGlimmerAssistantModel(configuration)
    }

    func testTinyShapeForwardPass() throws {
        let model = try tinyModel()
        let noise = MLXRandom.normal([1, 4, 16])
        let context = MLXRandom.normal([1, 6, 32])

        let output = model(noise, contextHiddenStates: context)
        eval(output)

        XCTAssertEqual(output.shape, [1, 4, 16])
        let values = output.reshaped(-1).asArray(Float.self)
        XCTAssertFalse(values.isEmpty)
        XCTAssertTrue(values.allSatisfy { $0.isFinite })
    }

    func testParameterKeysMatchCheckpointLayout() throws {
        let model = try tinyModel()
        let keys = Set(model.parameters().flattened().map { $0.0 })

        var expected: Set<String> = [
            "encoder.fc.weight", "encoder.output_norm_enc.weight", "norm.weight",
        ]
        for layer in 0 ..< 2 {
            for name in [
                "input_layernorm.weight", "post_attention_layernorm.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            ] {
                expected.insert("layers.\(layer).\(name)")
            }
        }
        XCTAssertEqual(keys, expected)
    }

    func testTargetContextChangesBlockOutput() throws {
        let model = try tinyModel()
        let noise = MLXRandom.normal([1, 4, 16])
        let contextA = MLXRandom.normal([1, 6, 32])
        let contextB = MLXRandom.normal([1, 6, 32])

        let outputA = model(noise, contextHiddenStates: contextA).reshaped(-1).asArray(Float.self)
        let outputB = model(noise, contextHiddenStates: contextB).reshaped(-1).asArray(Float.self)

        XCTAssertEqual(outputA.count, outputB.count)
        XCTAssertTrue(zip(outputA, outputB).contains { abs($0 - $1) > 0 })
    }

    func testBFloat16WeightsKeepOutputInBFloat16() throws {
        let model = try tinyModel()
        let flat = model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }
        model.update(parameters: ModuleParameters.unflattened(flat))

        let noise = MLXRandom.normal([1, 4, 16]).asType(.bfloat16)
        let context = MLXRandom.normal([1, 6, 32]).asType(.bfloat16)

        let output = model(noise, contextHiddenStates: context)
        eval(output)

        XCTAssertEqual(output.dtype, .bfloat16)
    }
}
