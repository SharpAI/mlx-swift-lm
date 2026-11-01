// SharpAI fork overlay tests for inkling_mm_model configuration decoding. Not part of upstream.

import Foundation
import MLXVLM
import XCTest

final class SharpAIInklingMmModelConfigurationTests: XCTestCase {

    // Tiny shape for tests only. The key names are the published config.json names.
    private func textFixture() -> [String: Any] {
        [
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 4,
            "sliding_window_size": 8,
            "d_rel": 2,
            "rel_extent": 16,
            "log_scaling_alpha": 0.1,
            "local_layer_ids": [0],
            "dense_mlp_idx": 1,
            "rms_norm_eps": 0.000001,
            "sconv_kernel_size": 4,
            "dense_intermediate_size": 32,
            "intermediate_size": 8,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "shared_expert_sink": true,
            "route_scale": 1.0,
            "logits_mup_width_multiplier": 1.0,
        ]
    }

    private func audioFixture() -> [String: Any] {
        [
            "n_mel_bins": 8,
            "mel_vocab_size": 16,
            "decoder_dmodel": 16,
        ]
    }

    private func visionFixture() -> [String: Any] {
        [
            "vision_encoder_type": "hmlp",
            "patch_size": 4,
            "temporal_patch_size": 2,
            "n_channels": 3,
            "n_layers": 1,
            "decoder_dmodel": 16,
        ]
    }

    private func document(text: [String: Any]? = nil) -> [String: Any] {
        [
            "model_type": "inkling_mm_model",
            "text_config": text ?? textFixture(),
            "audio_config": audioFixture(),
            "vision_config": visionFixture(),
        ]
    }

    private func decode(_ object: [String: Any]) throws -> SharpAIInklingMmModelConfiguration {
        let data = try JSONSerialization.data(withJSONObject: object)
        return try JSONDecoder().decode(SharpAIInklingMmModelConfiguration.self, from: data)
    }

    func testFixtureDecodesExplicitValues() throws {
        let config = try decode(document())

        XCTAssertEqual(config.text.vocabularySize, 32)
        XCTAssertEqual(config.text.hiddenSize, 16)
        XCTAssertEqual(config.text.hiddenLayers, 2)
        XCTAssertEqual(config.text.convKernelSize, 4)
        XCTAssertEqual(config.text.routedExperts, 4)
        XCTAssertEqual(config.text.logitsMupWidthMultiplier, 1.0)
        XCTAssertEqual(config.text.intermediateSize, 32)
        XCTAssertEqual(config.text.moeIntermediateSize, 8)
        XCTAssertEqual(config.text.layerTypes, [.hybridSliding, .hybrid])
        XCTAssertEqual(config.text.mlpLayerTypes, [.dense, .sparse])
        XCTAssertNil(config.imageTokenId)
    }

    func testEmptyConfigFailsWithoutCheckpointKeys() {
        XCTAssertThrowsError(try decode(["model_type": "inkling_mm_model"]))
    }

    func testMissingTextKeyFailsInsteadOfDefaulting() {
        var text = textFixture()
        text["hidden_size"] = nil

        XCTAssertThrowsError(try decode(document(text: text)))
    }

    func testMissingSectionFailsInsteadOfDefaulting() {
        var object = document()
        object["audio_config"] = nil

        XCTAssertThrowsError(try decode(object))
    }

    func testLayerPatternsFallBackToTransformersRules() throws {
        // transformers: without layer_types or local_layer_ids, layer i is sliding unless (i + 1) % 6 == 0.
        // Without mlp_layer_types, dense_mlp_idx defaults to 0, so every layer is sparse.
        var text = textFixture()
        text["num_hidden_layers"] = 6
        text["local_layer_ids"] = nil
        text["dense_mlp_idx"] = nil
        let config = try decode(document(text: text))

        XCTAssertEqual(
            config.text.layerTypes,
            [.hybridSliding, .hybridSliding, .hybridSliding, .hybridSliding, .hybridSliding, .hybrid])
        XCTAssertEqual(config.text.mlpLayerTypes, [MLPKind.sparse, .sparse, .sparse, .sparse, .sparse, .sparse])
    }

    func testExplicitLayerTypesWin() throws {
        var text = textFixture()
        text["layer_types"] = ["hybrid", "hybrid_sliding"]
        let config = try decode(document(text: text))

        XCTAssertEqual(config.text.layerTypes, [.hybrid, .hybridSliding])
    }

    func testUnknownLayerTypeFailsDecoding() {
        var text = textFixture()
        text["layer_types"] = ["full_attention", "hybrid"]

        XCTAssertThrowsError(try decode(document(text: text)))
    }

    func testLayerTypeCountMismatchFailsValidation() throws {
        var text = textFixture()
        text["layer_types"] = ["hybrid"]
        let config = try decode(document(text: text))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testTooManyActiveExpertsFailsValidation() throws {
        var text = textFixture()
        text["num_experts_per_tok"] = 8
        let config = try decode(document(text: text))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testVisionEncoderOtherThanHMLPFailsValidation() throws {
        var vision = visionFixture()
        vision["vision_encoder_type"] = "conv"
        var object = document()
        object["vision_config"] = vision
        let config = try decode(object)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    // Checkpoint shapes from the thinkingmachines/Inkling safetensors headers. The planner must
    // reproduce them: vision linear_0 is [128, 75] and linear_3 is [6144, 9600].
    func testVisionPlannerMatchesCheckpointShapes() {
        let stages = SharpAIInklingMmVisionPlanner.stages(
            temporalPatchSize: 2, patchSize: 40, layers: 4, channels: 3, decoderDModel: 6144)

        XCTAssertEqual(stages.count, 4)
        XCTAssertEqual(stages[0].inputChannels, 75)
        XCTAssertEqual(stages[0].outputChannels, 128)
        XCTAssertEqual(stages[3].inputChannels, 9600)
        XCTAssertEqual(stages[3].outputChannels, 6144)
        XCTAssertEqual(stages.map(\.hasNorm), [true, true, true, false])
    }
}

private typealias MLPKind = SharpAIInklingMmModelMLPType
