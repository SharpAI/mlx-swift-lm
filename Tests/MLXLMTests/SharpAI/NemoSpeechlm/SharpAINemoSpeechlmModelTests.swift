// SharpAI fork overlay: nemo_speechlm known-issue test. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAINemoSpeechlmModelTests: XCTestCase {

    /// KNOWN GAP, not a skip. Verified sources: the HF config and safetensors header of
    /// nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B, and vLLM nemotron3_5_transcribe.py and
    /// nemotron3_5_transcribe_audio.py. The ASR transformer encoder and the key mapping now exist
    /// (see testAudioEncoderASRStackShape and the sanitizer tests). The full forward is still
    /// missing: the log-mel preprocessor, the Sortformer speaker branch and ParallelExpertEncoder
    /// fuse (which the audio embedding depends on), and an embedding-input entry point on
    /// NemotronHModel (it takes token ids only). The creator therefore still throws.
    /// Once it returns a model instead of throwing, XCTExpectFailure reports an unexpected success,
    /// which is the signal to turn this into a real forward-pass shape test.
    func testTinyShapeForwardPass() throws {
        let data = Data(
            #"{"model_type": "nemo_speechlm", "architectures": ["Nemotron3_5TranscribeForConditionalGeneration"]}"#
                .utf8)

        try XCTExpectFailure(
            "KNOWN GAP: nemo_speechlm full forward pass is not implemented. Missing: the log-mel preprocessor (perception.preprocessor STFT and filterbank), the Sortformer speaker branch with speaker-cache update and the ParallelExpertEncoder fuse that produce the audio embedding (vLLM nemotron3_5_transcribe_audio.py), the MTP head (llm.mtp.*, not needed for ASR), and an embedding-input entry point on NemotronHModel (repo NemotronH.swift takes token ids only, and is outside this overlay)."
        ) {
            let creator = try XCTUnwrap(SharpAINemoSpeechlmOverlay.creators["nemo_speechlm"])
            XCTAssertNoThrow(try creator(data))
        }
    }

    /// Real shape test for the ASR transformer encoder: frame stacking (factor 8), a tiny width,
    /// two layers, and per-row lengths. Input (2, 20, 16) gives 20 frames padded to 24, so 3 stacked
    /// frames. Row 1 has 12 valid frames, so it has ceil(12 / 8) = 2 valid output frames.
    func testAudioEncoderASRStackShape() throws {
        let config = SharpAINemoSpeechlmEncoderConfiguration(
            featIn: 16, dModel: 32, nHeads: 4, nLayers: 2, ffExpansion: 4,
            subsamplingFactor: 8, qkvBias: false, qkNorm: true, preBlockNorm: true,
            ropeBase: 10000, rotaryFraction: 0.5)
        let encoder = try SharpAINemoSpeechlmAudioEncoder(config)
        let features = MLXArray((0 ..< 2 * 20 * 16).map { Float($0 % 37) / 37 }, [2, 20, 16])

        let (out, outLengths) = encoder(features, lengths: [20, 12])

        XCTAssertEqual(out.shape, [2, 3, 32])
        XCTAssertEqual(outLengths, [3, 2])
    }

    func testAudioEncoderRejectsUnimplementedAttention() {
        let config = SharpAINemoSpeechlmEncoderConfiguration(
            featIn: 16, dModel: 32, nHeads: 4, nLayers: 1, ffExpansion: 4,
            subsamplingFactor: 8, qkvBias: false, qkNorm: true, preBlockNorm: true,
            ropeBase: 10000, rotaryFraction: 0.5, attnMode: "causal")

        XCTAssertThrowsError(try SharpAINemoSpeechlmAudioEncoder(config))
    }
}
