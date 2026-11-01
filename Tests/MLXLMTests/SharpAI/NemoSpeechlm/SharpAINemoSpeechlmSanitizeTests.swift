// SharpAI fork overlay tests for nemo_speechlm key sanitization. Not part of upstream.
// The real-name cases use key names read from the checkpoint header
// (https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B, model.safetensors) and the
// rules from vLLM nemotron3_5_transcribe.py. Tensor values are synthetic; only names and shapes are checked.
// The fixture.* keys are synthetic and exercise the pass-through path.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAINemoSpeechlmSanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    func testPassThroughKeepsFixtureKeys() throws {
        let weights: [String: MLXArray] = [
            "fixture.a.weight": scalar(),
            "fixture.b.weight": scalar(),
        ]
        let sanitized = try SharpAINemoSpeechlmSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            ["fixture.a.weight", "fixture.b.weight"])
    }

    func testMapsCheckpointNamesToDecoderNames() throws {
        let weights: [String: MLXArray] = [
            "llm.model.embed_tokens.weight": scalar(),
            "llm.model.norm.weight": scalar(),
            "llm.lm_head.weight": scalar(),
            "llm.model.layers.0.mixer._fp32_params.A_log": scalar(),
            "llm.model.layers.1.norm.weight": scalar(),
            "llm.model.layers.1.mixer.gate.weight": scalar(),
            "llm.model.layers.5.mixer.attn_module._extra_state": scalar(),
            "llm.mtp.layers.0.eh_proj.weight": scalar(),
            "ctc_timestamp.decoder.weight": scalar(),
            "perception.proj.weight": scalar(),
        ]
        let sanitized = try SharpAINemoSpeechlmSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "backbone.embeddings.weight",
                "backbone.norm_f.weight",
                "lm_head.weight",
                "backbone.layers.0.mixer.A_log",
                "backbone.layers.1.norm.weight",
                "backbone.layers.1.mixer.gate.weight",
                "perception.proj.weight",
            ])
    }

    func testUnpacksStackedExpertsPerExpertTransposed() throws {
        // NeMo packs experts as (experts, in, out) and vLLM transposes each expert.
        // gate_and_up_projs is (2, 4, 3): hidden 4, inter 3. down_projs is (2, 3, 4).
        let up = MLXArray((0 ..< 24).map { Float($0) }, [2, 4, 3])
        let down = MLXArray((0 ..< 24).map { Float($0) }, [2, 3, 4])
        let weights: [String: MLXArray] = [
            "llm.model.layers.1.mixer.experts.gate_and_up_projs": up,
            "llm.model.layers.1.mixer.experts.down_projs": down,
        ]
        let sanitized = try SharpAINemoSpeechlmSanitizer.sanitize(weights)

        let up1 = try XCTUnwrap(sanitized["backbone.layers.1.mixer.experts.1.up_proj.weight"])
        XCTAssertEqual(up1.shape, [3, 4])
        // Output[i, h] = up[1][h][i] = 12 + 3h + i, so [0, 0] is 12 and [0, 1] is 15.
        XCTAssertEqual(up1[0, 0].item(Float.self), 12)
        XCTAssertEqual(up1[0, 1].item(Float.self), 15)

        let down0 = try XCTUnwrap(sanitized["backbone.layers.1.mixer.experts.0.down_proj.weight"])
        XCTAssertEqual(down0.shape, [4, 3])
        // Output[c, r] = down[0][r][c] = 4r + c, so [1, 0] is 1 and [1, 2] is 9.
        XCTAssertEqual(down0[1, 0].item(Float.self), 1)
        XCTAssertEqual(down0[1, 2].item(Float.self), 9)

        XCTAssertNil(sanitized["backbone.layers.1.mixer.experts.down_projs"])
        XCTAssertNil(sanitized["backbone.layers.1.mixer.experts.gate_and_up_projs"])
    }

    func testUnknownDecoderKeyThrows() {
        let weights: [String: MLXArray] = [
            "llm.unknown_block.weight": scalar()
        ]

        XCTAssertThrowsError(try SharpAINemoSpeechlmSanitizer.sanitize(weights)) { error in
            guard case SharpAINemoSpeechlmError.unmappedDecoderWeight = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testEmptyKeyThrows() {
        let weights: [String: MLXArray] = [
            "": scalar()
        ]

        XCTAssertThrowsError(try SharpAINemoSpeechlmSanitizer.sanitize(weights)) { error in
            guard case SharpAINemoSpeechlmError.emptyWeightKey = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
