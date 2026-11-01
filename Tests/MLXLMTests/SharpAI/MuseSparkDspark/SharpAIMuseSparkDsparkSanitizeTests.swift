// SharpAI fork overlay tests for the muse_spark_dspark key sanitize. Not part of upstream.
// Offline only: inline JSON and zero-filled tensors.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMuseSparkDsparkSanitizeTests: XCTestCase {

    private func config(confidence: Bool, residualGate: Bool) throws
        -> SharpAIMuseSparkDsparkConfiguration
    {
        let json = """
            {
              "model_type": "muse_spark_dspark",
              "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
              "num_key_value_heads": 1, "head_dim": 8, "vocab_size": 50, "draft_vocab_size": 40,
              "moe": {"is_enabled": false}, "markov_rank": 4,
              "target_layer_ids": [1, 3], "target_hidden_size": 6,
              "enable_confidence_head": \(confidence),
              "use_residual_gate": \(residualGate)
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIMuseSparkDsparkConfiguration.self, from: Data(json.utf8))
    }

    private func checkpointWeights() -> [String: MLXArray] {
        [
            "layers.0.self_attn.q_proj.weight": MLXArray.zeros([2, 2]),
            "layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([1]),
            "layers.0.add_attn.gate": MLXArray.zeros([2]),
            "fc.weight": MLXArray.zeros([2, 2]),
            "hidden_norm.weight": MLXArray.zeros([2]),
            "norm.weight": MLXArray.zeros([2]),
            "markov_head.markov_w1.weight": MLXArray.zeros([2, 2]),
            "confidence_head.proj.weight": MLXArray.zeros([1, 2]),
            "lm_head.weight": MLXArray.zeros([2, 2]),
            "embed_tokens.weight": MLXArray.zeros([2, 2]),
            "mask_embedding": MLXArray.zeros([2]),
            "t2d": MLXArray.zeros([2]),
            "d2t": MLXArray.zeros([2]),
        ]
    }

    func testRenamesKeysAndDropsTargetOwnedWeights() throws {
        let cfg = try config(confidence: true, residualGate: true)

        let sanitized = SharpAIMuseSparkDsparkSanitizer.sanitize(
            checkpointWeights(), configuration: cfg)

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.add_attn.gate",
                "model.fc.weight",
                "model.hidden_norm.weight",
                "model.norm.weight",
                "model.markov_head.markov_w1.weight",
                "model.confidence_head.proj.weight",
                "lm_head.weight",
                "draft_id_to_target_id",
            ])
    }

    func testDropsConfidenceHeadAndResidualGateWhenDisabled() throws {
        let cfg = try config(confidence: false, residualGate: false)

        let sanitized = SharpAIMuseSparkDsparkSanitizer.sanitize(
            checkpointWeights(), configuration: cfg)

        XCTAssertFalse(sanitized.keys.contains { $0.contains("confidence_head") })
        XCTAssertFalse(sanitized.keys.contains { $0.contains("add_attn") })
        XCTAssertTrue(sanitized.keys.contains("lm_head.weight"))
    }

    func testKeepsAlreadyPrefixedKeys() throws {
        let cfg = try config(confidence: false, residualGate: false)
        let weights: [String: MLXArray] = ["model.fc.weight": MLXArray.zeros([2, 2])]

        let sanitized = SharpAIMuseSparkDsparkSanitizer.sanitize(weights, configuration: cfg)

        XCTAssertEqual(Set(sanitized.keys), ["model.fc.weight"])
    }
}
