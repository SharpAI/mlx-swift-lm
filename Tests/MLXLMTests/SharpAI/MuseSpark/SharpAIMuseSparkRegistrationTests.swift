// SharpAI fork overlay tests for the muse_spark creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIMuseSparkRegistrationTests: XCTestCase {

    /// `text_config` of the tiny-random muse_spark checkpoint (tiny-random/muse-spark-open).
    private let tinyTextConfigJSON = """
        {
            "attention_bias": false,
            "bos_token_id": 200000,
            "eos_token_id": 200001,
            "final_logit_softcapping": 20.0,
            "head_dim": 32,
            "hidden_act": "silu",
            "hidden_size": 8,
            "initializer_range": 0.02,
            "intermediate_size": 32,
            "layer_rope_theta": [500000.0, 0, 500000.0, 500000.0, 500000.0, 0, 500000.0, 500000.0],
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention"],
            "max_position_embeddings": 1048576,
            "mlp_layer_types": ["sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse"],
            "model_type": "muse_spark_text",
            "moe_hidden_size": 4,
            "moe_intermediate_size": 16,
            "num_attention_heads": 8,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 8,
            "num_key_value_heads": 1,
            "num_local_experts": 32,
            "output_multiplier": 0.17677669529663687,
            "pad_token_id": 200018,
            "post_norm_eps": 1e-08,
            "qk_scale_factor": 5.4730064863875,
            "residual_gate_temperature": 0.3,
            "rms_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            "routed_scaling_factor": 1.0,
            "sliding_window": 2048,
            "tie_word_embeddings": false,
            "vocab_size": 202048
        }
        """

    private let validJSON = """
        {
            "architectures": ["MuseSparkForConditionalGeneration"],
            "model_type": "muse_spark"
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMuseSparkOverlay.creators["muse_spark"])
    }

    func testValidConfigFailsUntilModelIsImplemented() throws {
        guard let creator = SharpAIMuseSparkOverlay.creators["muse_spark"] else {
            XCTFail("muse_spark creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIMuseSparkError.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testWrongModelTypeFailsBeforeConstruction() throws {
        guard let creator = SharpAIMuseSparkOverlay.creators["muse_spark"] else {
            XCTFail("muse_spark creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(#"{ "model_type": "llama" }"#.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("config validation should fail first, got: \(error)")
                return
            }
        }
    }

    // Tiny-shape forward pass with random weights. Shapes come from the tiny-random text_config
    // above; the layer math follows transformers modeling_muse_spark.py.
    func testTinyForwardPassProducesFiniteCausalLogits() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMuseSparkTextConfiguration.self, from: Data(tinyTextConfigJSON.utf8))
        let model = SharpAIMuseSparkTextForCausalLM(config)

        let logits = model(MLXArray([1, 5, 9, 2] as [Int32]).reshaped(1, 4), cache: nil)
        let logitsChangedLast = model(MLXArray([1, 5, 9, 7] as [Int32]).reshaped(1, 4), cache: nil)
        eval(logits, logitsChangedLast)

        XCTAssertEqual(logits.shape, [1, 4, 202048])
        // NaN is the only value that is not equal to itself.
        XCTAssertTrue((logits .== logits).all().item(Bool.self))
        // Causal attention and per-token MoE: positions before the last token do not change.
        XCTAssertTrue(
            allClose(
                logits[0..., ..<3, 0...], logitsChangedLast[0..., ..<3, 0...],
                rtol: 1e-4, atol: 1e-4
            ).item(Bool.self))
    }
}
