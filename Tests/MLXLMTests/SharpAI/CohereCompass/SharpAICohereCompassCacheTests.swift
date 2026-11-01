// SharpAI fork overlay tests for cohere_compass cache layout. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAICohereCompassCacheTests: XCTestCase {

    private let textConfigJSON = """
        {
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 50,
            "layer_norm_eps": 1e-5,
            "attention_bias": false,
            "sliding_window": 8,
            "hidden_act": "silu",
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
            ],
            "rope_parameters": {
                "sliding_attention": { "rope_theta": 10000.0 },
                "full_attention": null
            }
        }
        """

    func testSlidingLayersGetRotatingCacheAndFullLayersGetSimpleCache() throws {
        let config = try JSONDecoder().decode(
            SharpAICohereCompassTextConfiguration.self, from: Data(textConfigJSON.utf8))
        let model = SharpAICohereCompassTextModel(config)

        let caches = try model.newCache(parameters: nil)

        XCTAssertEqual(caches.count, 4)
        for i in 0 ..< 3 {
            let rotating = try XCTUnwrap(caches[i] as? RotatingKVCache)
            XCTAssertEqual(rotating.maxSize, 8)
        }
        XCTAssertTrue(caches[3] is KVCacheSimple)
    }
}
