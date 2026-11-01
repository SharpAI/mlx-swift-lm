// SharpAI fork overlay tests for openpangu_vl parameter layout. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import XCTest

final class SharpAIOpenpanguVlModelTests: XCTestCase {

    private func makeModel(tied: Bool, layers: Int = 1) throws -> SharpAIOpenpanguVlModel {
        let json = """
            {
                "model_type": "openpangu_vl",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": \(layers),
                "rms_norm_eps": 1e-05,
                "rope_theta": 10000.0,
                "max_position_embeddings": 128,
                "tie_word_embeddings": \(tied),
                "vocab_size": 64,
                "vision_config": {
                    "depth": 1,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_heads": 2,
                    "in_chans": 3,
                    "out_hidden_size": 32,
                    "patch_size": 14,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "window_size": 112,
                    "fullatt_block_indexes": [0]
                }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIOpenpanguVlConfiguration.self, from: Data(json.utf8))
        return try SharpAIOpenpanguVlModel(config)
    }

    private func parameterKeys(_ model: SharpAIOpenpanguVlModel) -> Set<String> {
        Set(model.parameters().flattened().map { $0.0 })
    }

    func testTiedModelHasNoLMHeadParameter() throws {
        let keys = parameterKeys(try makeModel(tied: true))

        XCTAssertFalse(keys.contains { $0.hasPrefix("lm_head") })
        XCTAssertTrue(keys.contains { $0.hasPrefix("model.embed_tokens") })
    }

    func testUntiedModelKeepsLMHeadParameter() throws {
        let keys = parameterKeys(try makeModel(tied: false))

        XCTAssertTrue(keys.contains("lm_head.weight"))
    }

    func testLoraLayersAreTheDecoderLayers() throws {
        let model = try makeModel(tied: true, layers: 3)

        XCTAssertEqual(model.loraLayers.count, 3)
    }
}
