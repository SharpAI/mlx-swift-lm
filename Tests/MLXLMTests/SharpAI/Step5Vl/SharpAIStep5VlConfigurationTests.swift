// SharpAI fork overlay tests for step5_vl config decoding. Not part of upstream.
// UNVERIFIED: fixture values are copied from the conformance fixture, trimmed to four layers.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIStep5VlConfigurationTests: XCTestCase {

    private let fixtureJSON = """
        {
            "architectures": ["Step5ForConditionalGeneration"],
            "model_type": "step5_vl",
            "image_token_len": 169,
            "patch_token_len": 81,
            "image_token_id": 128001,
            "understand_projector_stride": 2,
            "use_im_start_end": "true",
            "vision_select_layer": -1,
            "projector_bias": false,
            "vision_config": {
                "model_type": "perception_encoder",
                "image_size": 728,
                "patch_size": 14,
                "width": 1536,
                "layers": 47,
                "heads": 16,
                "pool_type": "none",
                "output_dim": null,
                "use_cls_token": false,
                "ls_init_value": 0.1,
                "use_ln_post": false,
                "hidden_act": "quick_gelu"
            },
            "text_config": {
                "architectures": ["Step5ForCausalLM"],
                "model_type": "step5",
                "hidden_size": 4096,
                "intermediate_size": 13824,
                "num_hidden_layers": 4,
                "vocab_size": 128896,
                "rms_norm_eps": 1e-05,
                "num_attention_heads": 64,
                "num_attention_groups": 4,
                "head_dim": 192,
                "sliding_window": 512,
                "rope_theta": [10000.0, 10000.0, 10000.0, 10000000.0],
                "layer_types": [
                    "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
                ],
                "partial_rotary_factors": [1.0, 1.0, 1.0, 0.3333333333333333],
                "use_head_wise_attn_gate": true,
                "num_nextn_predict_layers": 3,
                "use_moe": true,
                "moe_num_experts": 352,
                "moe_top_k": 8,
                "moe_intermediate_size": 1536,
                "share_expert_dim": 1536,
                "sparse_config": {
                    "enabled": true,
                    "proxy_dim": 256,
                    "topk": 512,
                    "region_block_size": 8,
                    "num_provider_groups": 4,
                    "attention_impl": "sparse_gqa",
                    "apply_to_layer_types": ["full_attention"]
                }
            }
        }
        """

    private let minimalJSON = """
        {
            "vision_config": {
                "image_size": 728,
                "patch_size": 14,
                "width": 64,
                "layers": 2,
                "heads": 4
            },
            "text_config": {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4
            }
        }
        """

    private func decode(_ json: String) throws -> SharpAIStep5VlConfiguration {
        try JSONDecoder.json5().decode(SharpAIStep5VlConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesFixtureTopLevelFields() throws {
        let config = try decode(fixtureJSON)

        XCTAssertEqual(config.modelType, "step5_vl")
        XCTAssertEqual(config.architectures, ["Step5ForConditionalGeneration"])
        XCTAssertEqual(config.imageTokenId, 128_001)
        XCTAssertEqual(config.imageTokenLen, 169)
        XCTAssertEqual(config.patchTokenLen, 81)
        XCTAssertEqual(config.understandProjectorStride, 2)
        XCTAssertFalse(config.projectorBias)
        XCTAssertEqual(config.visionSelectLayer, -1)
        XCTAssertTrue(config.useImStartEnd)
    }

    func testDecodesVisionConfig() throws {
        let vision = try decode(fixtureJSON).visionConfig

        XCTAssertEqual(vision.modelType, "perception_encoder")
        XCTAssertEqual(vision.imageSize, 728)
        XCTAssertEqual(vision.patchSize, 14)
        XCTAssertEqual(vision.width, 1536)
        XCTAssertEqual(vision.layers, 47)
        XCTAssertEqual(vision.heads, 16)
        XCTAssertNil(vision.outputDim)
        XCTAssertEqual(vision.useClsToken, false)
        XCTAssertEqual(vision.hiddenAct, "quick_gelu")
    }

    func testDecodesTextConfigIncludingSparseAndMoeFields() throws {
        let text = try decode(fixtureJSON).textConfig

        XCTAssertEqual(text.hiddenLayers, 4)
        XCTAssertEqual(text.attentionHeads, 64)
        XCTAssertEqual(text.attentionGroups, 4)
        XCTAssertEqual(text.headDim, 192)
        XCTAssertEqual(text.slidingWindow, 512)
        XCTAssertEqual(text.ropeTheta, [10_000, 10_000, 10_000, 10_000_000])
        XCTAssertEqual(text.layerTypes?.last, "full_attention")
        XCTAssertEqual(text.useMoe, true)
        XCTAssertEqual(text.moeNumExperts, 352)
        XCTAssertEqual(text.moeTopK, 8)
        XCTAssertEqual(text.numNextnPredictLayers, 3)
        XCTAssertEqual(text.sparseConfig?.enabled, true)
        XCTAssertEqual(text.sparseConfig?.topk, 512)
        XCTAssertEqual(text.sparseConfig?.applyToLayerTypes, ["full_attention"])
    }

    func testScalarRopeThetaIsWrappedInList() throws {
        let config = try decode(
            """
            {
                "vision_config": {"image_size": 728, "patch_size": 14, "width": 64, "layers": 2, "heads": 4},
                "text_config": {
                    "vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
                    "num_hidden_layers": 2, "num_attention_heads": 4, "rope_theta": 500000.0
                }
            }
            """)

        XCTAssertEqual(config.textConfig.ropeTheta, [500_000])
    }

    func testAppliesDefaultsWhenOptionalFieldsAbsent() throws {
        let config = try decode(minimalJSON)

        // UNVERIFIED: defaults are the narrowest guess, not confirmed against a checkpoint.
        XCTAssertEqual(config.modelType, "step5_vl")
        XCTAssertNil(config.architectures)
        XCTAssertNil(config.imageTokenId)
        XCTAssertEqual(config.understandProjectorStride, 2)
        XCTAssertFalse(config.projectorBias)
        XCTAssertEqual(config.visionSelectLayer, -1)
        XCTAssertFalse(config.useImStartEnd)
        XCTAssertEqual(config.textConfig.attentionGroups, 4)
        XCTAssertEqual(config.textConfig.headDim, 4)
        XCTAssertEqual(config.textConfig.ropeTheta, [10_000])
        XCTAssertNil(config.textConfig.sparseConfig)
        XCTAssertNil(config.textConfig.layerTypes)
    }

    func testFixtureShapePassesValidation() throws {
        XCTAssertNoThrow(try decode(fixtureJSON).validateModelConfiguration())
    }

    func testValidationRejectsWrongModelType() throws {
        let config = try decode(
            """
            {
                "model_type": "step5",
                "vision_config": {"image_size": 728, "patch_size": 14, "width": 64, "layers": 2, "heads": 4},
                "text_config": {
                    "vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
                    "num_hidden_layers": 1, "num_attention_heads": 4
                }
            }
            """)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsGroupsThatDoNotDivideHeads() throws {
        let config = try decode(
            """
            {
                "vision_config": {"image_size": 728, "patch_size": 14, "width": 64, "layers": 2, "heads": 4},
                "text_config": {
                    "vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
                    "num_hidden_layers": 1, "num_attention_heads": 6, "num_attention_groups": 4
                }
            }
            """)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsImageSizeNotMultipleOfPatch() throws {
        let config = try decode(
            """
            {
                "vision_config": {"image_size": 730, "patch_size": 14, "width": 64, "layers": 2, "heads": 4},
                "text_config": {
                    "vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
                    "num_hidden_layers": 1, "num_attention_heads": 4
                }
            }
            """)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsUnknownLayerType() throws {
        let config = try decode(
            """
            {
                "vision_config": {"image_size": 728, "patch_size": 14, "width": 64, "layers": 2, "heads": 4},
                "text_config": {
                    "vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
                    "num_hidden_layers": 1, "num_attention_heads": 4,
                    "layer_types": ["linear_attention"]
                }
            }
            """)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
