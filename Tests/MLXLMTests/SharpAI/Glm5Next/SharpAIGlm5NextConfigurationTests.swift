// SharpAI fork overlay tests for the glm5_next configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIGlm5NextConfigurationTests: XCTestCase {

    /// Field values follow the GLM-5.3-Flash fixture. Layer lists are shortened.
    private let fixtureJSON = """
        {
            "architectures": ["Glm5NextForConditionalGeneration"],
            "model_type": "glm5_next",
            "image_token_id": 154854,
            "video_token_id": 154855,
            "image_start_token_id": 154830,
            "image_end_token_id": 154831,
            "video_start_token_id": 154832,
            "video_end_token_id": 154833,
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "glm5_next_text",
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "moe_intermediate_size": 2048,
                "num_hidden_layers": 45,
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "vocab_size": 154880,
                "rms_norm_eps": 1e-05,
                "max_position_embeddings": 1048576,
                "attention_bias": false,
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 256,
                "qk_rope_head_dim": 0,
                "qk_head_dim": 256,
                "v_head_dim": 256,
                "mla_use_nope": true,
                "index_topk": 2048,
                "index_head_dim": 128,
                "index_n_heads": 32,
                "index_kpool": 4,
                "index_kpool_always_select_tail": true,
                "index_kpool_compress": true,
                "index_share_for_mtp_iteration": true,
                "indexer_rope_interleave": true,
                "indexer_types": ["full", "full"],
                "mhc": true,
                "hc_mult": 4,
                "hc_sinkhorn_iters": 20,
                "hc_eps": 1e-06,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"],
                "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
                "linear_attn_config": {
                    "num_heads": 64,
                    "head_dim": 128,
                    "short_conv_kernel_size": 4,
                    "gate_lower_bound": -5.0,
                    "kda_layers": [0, 1, 2],
                    "full_attn_layers": [3]
                },
                "first_k_dense_replace": 3,
                "num_nextn_predict_layers": 1,
                "n_routed_experts": 288,
                "n_shared_experts": 1,
                "num_experts_per_tok": 8,
                "n_group": 1,
                "topk_group": 1,
                "routed_scaling_factor": 2.5,
                "norm_topk_prob": true,
                "scoring_func": "sigmoid",
                "topk_method": "noaux_tc",
                "moe_router_dtype": "float32",
                "swiglu_limit": 10.0
            },
            "vision_config": {
                "model_type": "glm5_next_vision",
                "depth": 24,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_heads": 16,
                "image_size": 448,
                "in_channels": 3,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 4096,
                "projection_intermediate_size": 10240,
                "rms_norm_eps": 1e-05,
                "attention_bias": true,
                "hidden_act": "silu",
                "swiglu_limit": 10.0
            }
        }
        """

    func testDecodesTopLevelFieldsAndSubConfigs() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIGlm5NextConfiguration.self, from: Data(fixtureJSON.utf8))

        XCTAssertEqual(config.modelType, "glm5_next")
        XCTAssertEqual(config.architectures, ["Glm5NextForConditionalGeneration"])
        XCTAssertEqual(config.imageTokenId, 154854)
        XCTAssertEqual(config.videoTokenId, 154855)
        XCTAssertEqual(config.videoEndTokenId, 154833)
        XCTAssertEqual(config.tieWordEmbeddings, false)
        XCTAssertEqual(config.textConfig?.modelType, "glm5_next_text")
        XCTAssertEqual(config.visionConfig?.modelType, "glm5_next_vision")
    }

    func testDecodesTextConfigFields() throws {
        let text = try XCTUnwrap(
            JSONDecoder.json5().decode(SharpAIGlm5NextConfiguration.self, from: Data(fixtureJSON.utf8))
                .textConfig)

        XCTAssertEqual(text.numHiddenLayers, 45)
        XCTAssertEqual(text.nRoutedExperts, 288)
        XCTAssertEqual(text.numExpertsPerTok, 8)
        XCTAssertEqual(text.firstKDenseReplace, 3)
        XCTAssertEqual(text.numNextnPredictLayers, 1)
        XCTAssertEqual(text.routedScalingFactor, 2.5)
        XCTAssertEqual(text.scoringFunc, "sigmoid")
        XCTAssertEqual(text.topkMethod, "noaux_tc")
        XCTAssertEqual(text.swigluLimit, 10.0)
    }

    func testDecodesNoPEMLAFieldsAsShipped() throws {
        // NoPE: the rope slice is zero-width, so qk_nope_head_dim carries the full query/key width.
        let text = try XCTUnwrap(
            JSONDecoder.json5().decode(SharpAIGlm5NextConfiguration.self, from: Data(fixtureJSON.utf8))
                .textConfig)

        XCTAssertEqual(text.mlaUseNope, true)
        XCTAssertEqual(text.qkRopeHeadDim, 0)
        XCTAssertEqual(text.qkNopeHeadDim, 256)
        XCTAssertEqual(text.kvLoraRank, 512)
        XCTAssertEqual(text.qLoraRank, 1536)
        XCTAssertEqual(text.vHeadDim, 256)
    }

    func testDecodesKpoolIndexerAndMHCFields() throws {
        let text = try XCTUnwrap(
            JSONDecoder.json5().decode(SharpAIGlm5NextConfiguration.self, from: Data(fixtureJSON.utf8))
                .textConfig)

        XCTAssertEqual(text.indexTopk, 2048)
        XCTAssertEqual(text.indexHeadDim, 128)
        XCTAssertEqual(text.indexKpool, 4)
        XCTAssertEqual(text.indexKpoolCompress, true)
        XCTAssertEqual(text.indexKpoolAlwaysSelectTail, true)
        XCTAssertEqual(text.indexShareForMtpIteration, true)
        XCTAssertEqual(text.mhc, true)
        XCTAssertEqual(text.hcMult, 4)
        XCTAssertEqual(text.hcSinkhornIters, 20)
        XCTAssertEqual(text.hcEps, 1e-6)
    }

    func testDecodesLinearAttentionLayout() throws {
        let text = try XCTUnwrap(
            JSONDecoder.json5().decode(SharpAIGlm5NextConfiguration.self, from: Data(fixtureJSON.utf8))
                .textConfig)
        let linear = try XCTUnwrap(text.linearAttnConfig)

        XCTAssertEqual(linear.numHeads, 64)
        XCTAssertEqual(linear.headDim, 128)
        XCTAssertEqual(linear.shortConvKernelSize, 4)
        XCTAssertEqual(linear.gateLowerBound, -5.0)
        XCTAssertEqual(linear.kdaLayers, [0, 1, 2])
        XCTAssertEqual(linear.fullAttnLayers, [3])
        XCTAssertEqual(text.layerTypes?[3], "deepseek_sparse_attention")
    }

    func testDecodesVisionConfig() throws {
        let vision = try XCTUnwrap(
            JSONDecoder.json5().decode(SharpAIGlm5NextConfiguration.self, from: Data(fixtureJSON.utf8))
                .visionConfig)

        XCTAssertEqual(vision.depth, 24)
        XCTAssertEqual(vision.hiddenSize, 1024)
        XCTAssertEqual(vision.numHeads, 16)
        XCTAssertEqual(vision.imageSize, 448)
        XCTAssertEqual(vision.patchSize, 14)
        XCTAssertEqual(vision.temporalPatchSize, 2)
        XCTAssertEqual(vision.spatialMergeSize, 2)
        XCTAssertEqual(vision.outHiddenSize, 4096)
        XCTAssertEqual(vision.projectionIntermediateSize, 10240)
        XCTAssertEqual(vision.attentionBias, true)
    }

    func testMissingFieldsDecodeAsNil() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIGlm5NextConfiguration.self, from: Data("{}".utf8))

        XCTAssertNil(config.modelType)
        XCTAssertNil(config.textConfig)
        XCTAssertNil(config.visionConfig)
    }

    func testValidationAcceptsGlm5NextWithTextConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIGlm5NextConfiguration.self, from: Data(fixtureJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsOtherModelType() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIGlm5NextConfiguration.self,
            from: Data(#"{ "model_type": "glm5", "text_config": {} }"#.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testValidationRejectsMissingTextConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIGlm5NextConfiguration.self,
            from: Data(#"{ "model_type": "glm5_next" }"#.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
