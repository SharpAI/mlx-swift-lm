import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIHyV4ConfigurationTests: XCTestCase {

    /// Short config in the shape of the offline fixture. Three layers, so the per-layer arrays have three entries.
    static func makeJSON(
        numExpertsPerTok: Int = 8, extra: String = ""
    ) -> String {
        """
        {
            "architectures": ["HYV4ForCausalLM"],
            "model_type": "hy_v4",
            "vocab_size": 120832,
            "hidden_size": 6144,
            "intermediate_size": 18432,
            "num_hidden_layers": 3,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "rms_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
            "max_position_embeddings": 1048576,
            "attention_bias": false,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "gated_mla": true,
            "learnable_sink": true,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_topk": 2048,
            "indexer_types": ["full", "shared", "shared"],
            "layer_types": ["deepseek_sparse_attention", "deepseek_sparse_attention", "deepseek_sparse_attention"],
            "n_routed_experts": 256,
            "num_experts_per_tok": \(numExpertsPerTok),
            "moe_intermediate_size": 2048,
            "n_shared_experts": 1,
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 2.827,
            "norm_topk_prob": true,
            "gating_type": "elementwise",
            "mlp_layer_types": ["dense", "sparse", "sparse"],
            "swiglu_limit": 10.0,
            "enable_ihc": true,
            "hc_mult": 4,
            "hc_magnitude": 2.0,
            "hc_eps": 1e-06,
            "num_nextn_predict_layers": 1\(extra)
        }
        """
    }

    func testConfigurationDecodesInlineJSON() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIHyV4Configuration.self, from: Data(Self.makeJSON().utf8))
        XCTAssertEqual(config.modelType, "hy_v4")
        XCTAssertEqual(config.architectures, ["HYV4ForCausalLM"])
        XCTAssertEqual(config.vocabularySize, 120832)
        XCTAssertEqual(config.hiddenSize, 6144)
        XCTAssertEqual(config.hiddenLayers, 3)
        XCTAssertEqual(config.attentionHeads, 64)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.qLoraRank, 2048)
        XCTAssertEqual(config.kvLoraRank, 512)
        XCTAssertEqual(config.qkNopeHeadDim, 192)
        XCTAssertEqual(config.qkRopeHeadDim, 64)
        XCTAssertEqual(config.vHeadDim, 256)
        XCTAssertEqual(config.gatedMla, true)
        XCTAssertEqual(config.indexTopk, 2048)
        XCTAssertEqual(config.indexerTypes, ["full", "shared", "shared"])
        XCTAssertEqual(config.numRoutedExperts, 256)
        XCTAssertEqual(config.numExpertsPerTok, 8)
        XCTAssertEqual(config.numSharedExperts, 1)
        XCTAssertEqual(config.swigluLimit, 10.0)
        XCTAssertEqual(config.enableIhc, true)
        XCTAssertEqual(config.hcMult, 4)
        XCTAssertEqual(config.numNextnPredictLayers, 1)
        XCTAssertEqual(config.ropeParameters?.ropeTheta, 10_000_000)
    }

    func testValidConfigurationPasses() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIHyV4Configuration.self, from: Data(Self.makeJSON().utf8))
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsTopKAboveExpertCount() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIHyV4Configuration.self,
            from: Data(Self.makeJSON(numExpertsPerTok: 257).utf8))
        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard let typed = error as? SharpAIHyV4Error,
                case .invalidConfiguration = typed
            else {
                return XCTFail("expected invalidConfiguration, got \(error)")
            }
        }
    }

    func testValidationRejectsIndexerTypesLengthMismatch() throws {
        let json = Self.makeJSON().replacingOccurrences(
            of: "\"indexer_types\": [\"full\", \"shared\", \"shared\"]",
            with: "\"indexer_types\": [\"full\"]")
        let config = try JSONDecoder.json5().decode(
            SharpAIHyV4Configuration.self, from: Data(json.utf8))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
