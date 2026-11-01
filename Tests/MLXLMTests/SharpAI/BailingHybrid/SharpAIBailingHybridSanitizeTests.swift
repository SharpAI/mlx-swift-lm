// SharpAI fork overlay tests for bailing_hybrid checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIBailingHybridSanitizeTests: XCTestCase {

    private func configuration(tied: Bool, nextn: Int = 0) throws
        -> SharpAIBailingHybridConfiguration
    {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "intermediate_size": 128,
                "vocab_size": 512,
                "tie_word_embeddings": \(tied),
                "num_nextn_predict_layers": \(nextn)
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIBailingHybridConfiguration.self, from: Data(json.utf8))
    }

    private func weights(_ keys: [String]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: keys.map { ($0, MLXArray([Float(1)])) })
    }

    func testDropsLmHeadWhenEmbeddingsAreTied() throws {
        let config = try configuration(tied: true)
        let input = weights([
            "model.word_embeddings.weight",
            "lm_head.weight",
        ])

        let out = SharpAIBailingHybridSanitizer.sanitize(input, configuration: config)

        XCTAssertNotNil(out["model.word_embeddings.weight"])
        XCTAssertNil(out["lm_head.weight"])
    }

    func testKeepsLmHeadWhenEmbeddingsAreNotTied() throws {
        let config = try configuration(tied: false)
        let input = weights(["lm_head.weight"])

        let out = SharpAIBailingHybridSanitizer.sanitize(input, configuration: config)

        XCTAssertNotNil(out["lm_head.weight"])
    }

    func testDropsNextnLayerKeysPastHiddenLayers() throws {
        // UNVERIFIED: MTP layer assumed at index num_hidden_layers (here 2), one nextn layer.
        let config = try configuration(tied: false, nextn: 1)
        let input = weights([
            "model.layers.1.attention.query_key_value.weight",
            "model.layers.2.attention.query_key_value.weight",
            "model.layers.2.eh_proj.weight",
        ])

        let out = SharpAIBailingHybridSanitizer.sanitize(input, configuration: config)

        XCTAssertNotNil(out["model.layers.1.attention.query_key_value.weight"])
        XCTAssertNil(out["model.layers.2.attention.query_key_value.weight"])
        XCTAssertNil(out["model.layers.2.eh_proj.weight"])
    }

    func testKeepsLayerKeysPastNextnRange() throws {
        // With no nextn layers, an index past num_hidden_layers is kept, not dropped.
        let config = try configuration(tied: false, nextn: 0)
        let input = weights(["model.layers.2.attention.query_key_value.weight"])

        let out = SharpAIBailingHybridSanitizer.sanitize(input, configuration: config)

        XCTAssertNotNil(out["model.layers.2.attention.query_key_value.weight"])
    }

    func testMainLayerKeysKeepHFNames() throws {
        let config = try configuration(tied: false)
        let keys = [
            "model.word_embeddings.weight",
            "model.layers.0.attention.dense.weight",
            "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            "model.norm.weight",
        ]

        let out = SharpAIBailingHybridSanitizer.sanitize(weights(keys), configuration: config)

        XCTAssertEqual(Set(out.keys), Set(keys))
    }

    func testLayerIndexParsing() {
        XCTAssertEqual(
            SharpAIBailingHybridSanitizer.layerIndex(of: "model.layers.7.mlp.weight"), 7)
        XCTAssertNil(SharpAIBailingHybridSanitizer.layerIndex(of: "model.word_embeddings.weight"))
        XCTAssertNil(SharpAIBailingHybridSanitizer.layerIndex(of: "model.layers"))
    }
}
