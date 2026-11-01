// SharpAI fork overlay tests for the inkling_model creator and text model. Not part of upstream.
//
// Expected values are derived from the transformers modeling source
// (`models/inkling/modeling_inkling.py`), not from this implementation.

import Foundation
@testable import MLXLLM
import MLX
import MLXLMCommon
import MLXNN
import XCTest

final class SharpAIInklingModelRegistrationTests: XCTestCase {

    /// Small text config that exercises every branch: sliding and full layers, a dense layer
    /// followed by sparse MoE layers, shared experts, log scaling, and the padded vocabulary.
    private let tinyJSON = """
        {
            "model_type": "inkling_model",
            "architectures": ["InklingForCausalLM"],
            "vocab_size": 64,
            "unpadded_vocab_size": 60,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 1,
            "swa_head_dim": 8,
            "sliding_window_size": 4,
            "d_rel": 4,
            "rel_extent": 8,
            "log_scaling_n_floor": 2,
            "log_scaling_alpha": 0.1,
            "local_layer_ids": [0, 1, 3],
            "dense_mlp_idx": 1,
            "sconv_kernel_size": 4,
            "intermediate_size": 16,
            "dense_intermediate_size": 48,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "n_shared_experts": 2,
            "route_scale": 8.0,
            "logits_mup_width_multiplier": 4.0
        }
        """

    private func makeCreator() throws -> (Data) throws -> any LanguageModel {
        try XCTUnwrap(SharpAIInklingModelOverlay.creators["inkling_model"])
    }

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIInklingModelOverlay.creators["inkling_model"])
        XCTAssertNotNil(SharpAIInklingModelOverlay.creators["inkling_mm_model"])
    }

    func testValidConfigBuildsTextModel() throws {
        let creator = try makeCreator()
        let model = try XCTUnwrap(creator(Data(tinyJSON.utf8)) as? SharpAIInklingModel)

        XCTAssertEqual(model.vocabularySize, 64)
        // local_layer_ids [0, 1, 3] are sliding (1 kv head); layer 2 is full attention (2 kv heads).
        XCTAssertEqual(model.kvHeads, [1, 1, 2, 1])
    }

    func testMalformedConfigFailsBeforeConstruction() throws {
        let creator = try makeCreator()

        XCTAssertThrowsError(try creator(Data("not json".utf8)))
    }

    func testTinyForwardPassIsCausalAndUnpadded() throws {
        let creator = try makeCreator()
        let model = try XCTUnwrap(creator(Data(tinyJSON.utf8)) as? SharpAIInklingModel)

        // The two inputs differ only in the last token. Causal attention and causal short
        // convolutions mean logits before the last position must not change.
        let tokensA = MLXArray([Int32(3), 7, 11, 19, 23, 5]).reshaped([1, 6])
        let tokensB = MLXArray([Int32(3), 7, 11, 19, 23, 42]).reshaped([1, 6])
        let logitsA = model(tokensA, cache: nil)
        let logitsB = model(tokensB, cache: nil)

        // Logits are sliced to the unpadded vocabulary (60 of 64 rows).
        XCTAssertEqual(logitsA.shape, [1, 6, 60])
        XCTAssertTrue(logitsA.asArray(Float.self).allSatisfy { $0.isFinite })

        let prefixA = logitsA[0..., ..<5, 0...]
        let prefixB = logitsB[0..., ..<5, 0...]
        let maxDiff = abs(prefixA - prefixB).max().item(Float.self)
        XCTAssertLessThan(maxDiff, 1e-4)
    }

    func testIncrementalDecodeFailsLoudly() throws {
        let creator = try makeCreator()
        let model = try XCTUnwrap(creator(Data(tinyJSON.utf8)) as? SharpAIInklingModel)

        // No KV or short-conv state exists yet, so generation must fail loudly, not silently.
        XCTAssertThrowsError(try model.newCache(parameters: nil))
    }

    func testRelativeBiasMatchesTransformersFormula() {
        // transformers InklingRelativeLogits: bias[q, k] = (relative_states[q] @ proj)[q - k]
        // when 0 <= q - k < rel_extent, and 0 otherwise.
        let relative = SharpAIInklingRelativeLogits(dRel: 1, relExtent: 3)
        // @ParameterInfo properties must be replaced through update(parameters:), not assigned:
        // a direct set fatals in MLXNN ("please call update() on the array rather than setting it").
        _ = relative.update(
            parameters: ModuleParameters.unflattened([
                "proj": MLXArray([Float(1), 2, 3]).reshaped([1, 3])
            ]))

        // One head, d_rel = 1. Query q has relative state q + 1.
        let states = MLXArray((0 ..< 4).map { Float($0 + 1) }).reshaped([1, 4, 1, 1])
        let positions = MLXArray((0 ..< 4).map { Int32($0) })
        let bias = relative.positionBias(
            relativeStates: states, queryPositions: positions, keyPositions: positions)

        XCTAssertEqual(bias.shape, [1, 1, 4, 4])
        let expected: [Float] = [
            1, 0, 0, 0,
            4, 2, 0, 0,
            9, 6, 3, 0,
            0, 12, 8, 4,
        ]
        XCTAssertEqual(bias.reshaped([16]).asArray(Float.self), expected)
    }

    func testShortConvCausalDepthwiseOnEachStream() {
        // Four sconv streams: keys, values, attention output, MLP output. Each is the same causal
        // depthwise module. Channel 0 uses only the newest tap (output = 2x). Channel 1 uses only
        // the tap three steps back (output = x[t] + x[t - 3]).
        let taps = MLXArray([Float(0), 0, 0, 1, 1, 0, 0, 0]).reshaped([2, 1, 4])
        let inputValues: [Float] = (0 ..< 5).flatMap { (t: Int) -> [Float] in
            let base: Float = Float(t + 1)
            return [base, base * 10]
        }
        let input = MLXArray(inputValues).reshaped([1, 5, 2])
        let expected: [Float] = [2, 10, 4, 20, 6, 30, 8, 50, 10, 70]

        for stream in ["keys", "values", "attention_output", "mlp_output"] {
            let conv = SharpAIInklingShortConv(channels: 2, kernelSize: 4)
            _ = conv.update(parameters: ModuleParameters.unflattened(["weight": taps]))
            let output = conv(input)
            XCTAssertEqual(output.shape, [1, 5, 2], stream)
            XCTAssertEqual(output.reshaped([10]).asArray(Float.self), expected, stream)
        }
    }

    func testNVFP4ExpertDequantMatchesFormula() throws {
        // ModelOpt NVFP4 routed experts, as in thinkingmachines/Inkling-Small-NVFP4. Layer 3 is
        // quantized there. Layer 2 is kept BF16 by the checkpoint, so it is not used here.
        // Formula: value = e2m1(code) * e4m3(blockScale) * scale2, with element 2i in the low
        // nibble of byte i. The formula matches the BF16 experts of
        // https://huggingface.co/thinkingmachines/Inkling-Small to about 8% relative error,
        // which is FP4 rounding. The expected values below come from that formula. They are
        // exact in bfloat16.
        let packedBytes: [UInt8] = [
            16, 50, 84, 118, 152, 186, 220, 254, 99, 133, 167, 201, 235, 13, 47, 65,
            182, 216, 250, 28, 62, 80, 114, 148, 9, 43, 77, 111, 129, 163, 197, 231,
            133, 167, 201, 235, 13, 47, 65, 99, 216, 250, 28, 62, 80, 114, 148, 182,
            43, 77, 111, 129, 163, 197, 231, 9, 126, 144, 178, 212, 246, 24, 58, 92,
        ]
        let blockScaleBytes: [UInt8] = [0x38, 0x40, 0x30, 0x3C, 0x28, 0x44, 0x38, 0x42]
        let weights: [String: MLXArray] = [
            "model.llm.layers.3.mlp.experts.w13_weight":
                MLXArray(packedBytes).reshaped([2, 4, 8]),
            "model.llm.layers.3.mlp.experts.w13_weight.scale":
                MLXArray(blockScaleBytes).reshaped([2, 4, 1]),
            "model.llm.layers.3.mlp.experts.w13_weight.scale2": MLXArray([Float(0.5), 0.25]),
            "model.llm.layers.3.mlp.experts.w13_weight.input_amax": MLXArray(Float(1)),
            "model.llm.layers.3.mlp.experts.w13_weight.original_shape": MLXArray([
                Int64(2), 4, 16,
            ]),
        ]

        let sanitized = SharpAIInklingModelSanitizer.sanitize(weights)

        // The packed tensor and its four siblings become gate and up. Nothing else remains.
        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "model.layers.3.moe.experts.gate_proj.weight",
                "model.layers.3.moe.experts.up_proj.weight",
            ])

        let gate = try XCTUnwrap(sanitized["model.layers.3.moe.experts.gate_proj.weight"])
        let up = try XCTUnwrap(sanitized["model.layers.3.moe.experts.up_proj.weight"])
        XCTAssertEqual(gate.dtype, .bfloat16)
        XCTAssertEqual(gate.shape, [2, 2, 16])
        XCTAssertEqual(up.shape, [2, 2, 16])

        // Gate rows are 0 and 2 of each expert, and up rows are 1 and 3.
        let expectedGate: [Float] = [
            0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, -0.0, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0,
            -3.0, 1.0, -0.375, -0.0, -0.75, -0.25, -1.5, -0.5, 0.125, -1.0, 0.375, 0.0, 0.75, 0.25,
            1.5, 0.5, -0.125, 0.1875, -0.0, 0.375, -0.0625, -0.03125, -0.125, -0.09375, -0.25,
            -0.1875, 0.0, -0.375, 0.0625, 0.03125, 0.125, 0.09375, 0.25, -0.375, 0.25, -0.75, 0.5,
            -1.5, 1.0, 0.125, -0.0, 0.375, -0.25, 0.75, -0.5, 1.5, -1.0, -0.125, 0.0,
        ]
        let expectedUp: [Float] = [
            1.5, 4.0, 3.0, -0.0, 6.0, -1.0, -0.5, -2.0, -1.5, -4.0, -3.0, 0.0, -6.0, 1.0, 0.5, 2.0,
            -0.375, 0.0, -1.125, 0.75, -2.25, 1.5, -4.5, 3.0, 0.375, -0.0, 1.125, -0.75, 2.25, -1.5,
            4.5, -3.0, -0.0, -2.25, -0.75, -4.5, -1.5, 0.375, -3.0, 1.125, 0.0, 2.25, 0.75, 4.5,
            1.5, -0.375, 3.0, -1.125, -2.5, 3.75, 0.0, -0.3125, 0.625, -0.9375, 1.25, -1.875, 2.5,
            -3.75, -0.0, 0.3125, -0.625, 0.9375, -1.25, 1.875,
        ]
        XCTAssertEqual(gate.asType(.float32).asArray(Float.self), expectedGate)
        XCTAssertEqual(up.asType(.float32).asArray(Float.self), expectedUp)
    }
}
