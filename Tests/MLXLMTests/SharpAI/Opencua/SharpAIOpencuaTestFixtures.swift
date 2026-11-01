import Foundation
import MLXLMCommon

@testable import MLXVLM

/// Tiny nested OpenCUA config. Shapes mirror the Qwen2.5-VL test fixture; no weights are loaded.
enum SharpAIOpencuaTestFixtures {

    static let tinyJSON = """
        {
            "model_type": "opencua",
            "media_placeholder_token_id": 6,
            "architectures": ["OpenCUAForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen2",
                "vocab_size": 128,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 4096,
                "rope_theta": 1000000.0,
                "rope_scaling": {
                    "mrope_section": [2, 1, 1],
                    "type": "mrope"
                },
                "tie_word_embeddings": true
            },
            "vision_config": {
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "out_hidden_size": 64,
                "num_heads": 4,
                "patch_size": 16,
                "spatial_patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "window_size": 64,
                "fullatt_block_indexes": [0],
                "tokens_per_second": 2
            }
        }
        """

    static func decode(_ json: String = tinyJSON) throws -> SharpAIOpencuaConfiguration {
        try JSONDecoder.json5().decode(SharpAIOpencuaConfiguration.self, from: Data(json.utf8))
    }

    /// `tinyJSON` without `text_config.rope_scaling`, the shape of the real OpenCUA-7B config.
    static func withoutRopeScaling() throws -> String {
        var root = try JSONSerialization.jsonObject(with: Data(tinyJSON.utf8)) as! [String: Any]
        var text = root["text_config"] as! [String: Any]
        text.removeValue(forKey: "rope_scaling")
        root["text_config"] = text
        let data = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }
}
