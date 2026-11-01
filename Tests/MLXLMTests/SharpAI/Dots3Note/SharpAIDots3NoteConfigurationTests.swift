// SharpAI fork overlay tests for the dots3_note configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIDots3NoteConfigurationTests: XCTestCase {

    /// Full attention at layers 0, 1 and every fourth layer from 5. Sliding elsewhere.
    private var layerTypesJSON: String {
        (0 ..< 46).map { index in
            let full = index <= 1 || (index >= 5 && (index - 5) % 4 == 0)
            return "\"\(full ? "full_attention" : "sliding_attention")\""
        }.joined(separator: ", ")
    }

    private var configJSON: String {
        """
        {
            "architectures": ["Dots3NoteForCausalLM"],
            "model_type": "dots3_note",
            "hidden_size": 5120,
            "num_hidden_layers": 46,
            "vocab_size": 152064,
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": false,
            "layer_types": [\(layerTypesJSON)],
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "q_lora_rank": 1024,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "rope_theta": 80000000.0,
            "attention_gate_type": "headwise",
            "sliding_window_size": 513,
            "swa_num_attention_heads": 64,
            "swa_num_key_value_heads": 64,
            "swa_q_lora_rank": 1024,
            "swa_kv_lora_rank": 1024,
            "swa_qk_nope_head_dim": 192,
            "swa_qk_rope_head_dim": 64,
            "swa_v_head_dim": 128,
            "swa_rope_theta": 50000.0,
            "swa_attention_gate_type": "headwise",
            "index_head_dim": 128,
            "index_n_heads": 64,
            "index_topk": 2048,
            "moe_intermediate_size": 1536,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 1,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "scoring_func": "sigmoid",
            "routed_scaling_factor": 1.0,
            "norm_topk_prob": true,
            "vision_config": {
                "embed_dim": 1536,
                "hidden_size": 5120,
                "intermediate_size": 4224,
                "moe_intermediate_size": 2112,
                "num_hidden_layers": 42,
                "num_attention_heads": 24,
                "num_channels": 3,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "rms_norm_eps": 1e-05,
                "use_bias": false,
                "use_qk_norm": true,
                "post_norm": true,
                "pre_pixel_shuffle": true,
                "pyramid_num_routed": [
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                    4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 64
                ],
                "capacity_factor": 2,
                "router_scoring_func": "sigmoid",
                "router_scale": 1.0,
                "adapter_type": "patch_merger",
                "adapter_in_dim": 1536,
                "adapter_out_dim": 5120,
                "adapter_merge_size": 2
            },
            "audio_config": {
                "encoder_type": "dots",
                "whisper_config": {
                    "d_model": 1280,
                    "encoder_attention_heads": 20,
                    "encoder_ffn_dim": 5120,
                    "encoder_layers": 32,
                    "num_mel_bins": 128,
                    "max_source_positions": 6000,
                    "activation_function": "swiglu"
                },
                "use_conv2d_stem": true,
                "downsample_hidden_size": 480,
                "sampling_rate": 16000,
                "chunk_seconds": 60,
                "whisper_adapter_in_dim": 1280,
                "whisper_adapter_out_dim": 5120
            }
        }
        """
    }

    private func decode(_ json: String) throws -> SharpAIDots3NoteConfiguration {
        try JSONDecoder.json5().decode(SharpAIDots3NoteConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesTopLevelFields() throws {
        let config = try decode(configJSON)

        XCTAssertEqual(config.modelType, "dots3_note")
        XCTAssertEqual(config.hiddenSize, 5120)
        XCTAssertEqual(config.hiddenLayers, 46)
        XCTAssertEqual(config.layerTypes.count, 46)
        XCTAssertEqual(config.routedExperts, 256)
        XCTAssertEqual(config.sharedExperts, 1)
        XCTAssertEqual(config.numExpertsPerTok, 8)
        XCTAssertEqual(config.indexTopK, 2048)
        XCTAssertEqual(config.attentionGateType, "headwise")
        XCTAssertEqual(config.swaAttentionGateType, "headwise")
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testDecodesVisionAndAudioBlocks() throws {
        let config = try decode(configJSON)

        XCTAssertEqual(config.vision?.embedDim, 1536)
        XCTAssertEqual(config.vision?.hiddenLayers, 42)
        XCTAssertEqual(config.vision?.pyramidNumRouted.count, 42)
        XCTAssertEqual(config.vision?.adapterOutDim, 5120)
        XCTAssertEqual(config.audio?.encoderType, "dots")
        XCTAssertEqual(config.audio?.whisper.encoderLayers, 32)
        XCTAssertEqual(config.audio?.whisper.activationFunction, "swiglu")
        XCTAssertEqual(config.audio?.chunkSeconds, 60)
    }

    func testFullAttentionGeometryUsesMLABranch() throws {
        let config = try decode(configJSON)
        let geometry = try XCTUnwrap(config.attentionGeometry(at: 0))

        XCTAssertEqual(geometry.layerType, .fullAttention)
        XCTAssertEqual(geometry.attentionHeads, 128)
        XCTAssertEqual(geometry.kvHeads, 128)
        XCTAssertEqual(geometry.qLoraRank, 1024)
        XCTAssertEqual(geometry.kvLoraRank, 512)
        XCTAssertEqual(geometry.qkNopeHeadDim, 128)
        XCTAssertEqual(geometry.qkRopeHeadDim, 64)
        XCTAssertEqual(geometry.qHeadDim, 192)
        XCTAssertEqual(geometry.vHeadDim, 128)
        XCTAssertEqual(geometry.ropeTheta, 80_000_000)
        XCTAssertNil(geometry.slidingWindow)
    }

    func testSlidingAttentionGeometryUsesSWABranch() throws {
        let config = try decode(configJSON)
        let geometry = try XCTUnwrap(config.attentionGeometry(at: 2))

        XCTAssertEqual(geometry.layerType, .slidingAttention)
        XCTAssertEqual(geometry.attentionHeads, 64)
        XCTAssertEqual(geometry.kvHeads, 64)
        XCTAssertEqual(geometry.kvLoraRank, 1024)
        XCTAssertEqual(geometry.qkNopeHeadDim, 192)
        XCTAssertEqual(geometry.qHeadDim, 256)
        XCTAssertEqual(geometry.ropeTheta, 50_000)
        XCTAssertEqual(geometry.slidingWindow, 513)
    }

    func testFullAttentionLayersMatchFixturePattern() throws {
        let config = try decode(configJSON)
        let fullLayers = (0 ..< 46).filter {
            config.attentionGeometry(at: $0)?.layerType == .fullAttention
        }

        XCTAssertEqual(fullLayers, [0, 1] + Array(stride(from: 5, through: 45, by: 4)))
    }

    func testAttentionGeometryOutOfRangeIsNil() throws {
        let config = try decode(configJSON)

        XCTAssertNil(config.attentionGeometry(at: -1))
        XCTAssertNil(config.attentionGeometry(at: 46))
    }

    func testFirstLayerIsDenseThenMoE() throws {
        let config = try decode(configJSON)

        XCTAssertFalse(config.isMoELayer(0))
        XCTAssertTrue(config.isMoELayer(1))
        XCTAssertTrue(config.isMoELayer(45))
    }

    func testRejectsUnknownLayerType() {
        let json = configJSON.replacingOccurrences(
            of: "\"sliding_attention\"", with: "\"linear_attention\"")

        XCTAssertThrowsError(try decode(json))
    }

    func testValidationRejectsLayerCountMismatch() throws {
        let json = configJSON.replacingOccurrences(
            of: "\"num_hidden_layers\": 46", with: "\"num_hidden_layers\": 45")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    /// Was a skip. The published config.json has no `mtp` or `nextn` key, so the base config
    /// carries no MTP section to decode. The checkpoint index does carry the MTP weights at
    /// `model.layers.46.*` (eh_proj, enorm, hnorm, ...) and `model.mtp.embed_tokens.weight`. vLLM
    /// nvidia/mtp.py builds them with num_nextn_predict_layers, which Dots3NoteConfig defaults to 1
    /// (vllm/transformers_utils/configs/dots3_note.py). The MTP type is not written in Swift, so the
    /// creator-level assertion is recorded as an expected failure.
    func testMTPConfigKnownIssueSkipped() throws {
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(configJSON.utf8)) as? [String: Any])
        XCTAssertFalse(object.keys.contains { $0.contains("mtp") || $0.contains("nextn") })

        XCTExpectFailure(
            "KNOWN GAP: dots3_note_mtp creator is not written in Swift. The reference is vLLM "
                + "nvidia/mtp.py; the weights are at model.layers.46.* and model.mtp.embed_tokens.*, "
                + "and num_nextn_predict_layers defaults to 1 in the vLLM config.") {
            XCTAssertNotNil(SharpAIDots3NoteOverlay.creators["dots3_note_mtp"])
        }
    }
}
