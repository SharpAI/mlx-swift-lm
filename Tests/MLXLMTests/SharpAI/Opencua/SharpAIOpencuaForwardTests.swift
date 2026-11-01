import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// Tiny-shape text-only forward pass with random weights. No checkpoint is loaded.
final class SharpAIOpencuaForwardTests: XCTestCase {

    private func makeModel() throws -> SharpAIOpencuaModel {
        let data = Data(SharpAIOpencuaTestFixtures.tinyJSON.utf8)
        let config = try SharpAIOpencuaTestFixtures.decode()
        return SharpAIOpencuaModel(try config.qwen25VLConfiguration(from: data))
    }

    func testTextOnlyForwardHasVocabularyLogits() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(1), 2, 3]).expandedDimensions(axis: 0)

        let output = model(LMInput.Text(tokens: tokens), cache: nil, state: nil)

        XCTAssertEqual(output.logits.shape, [1, 3, 128])
    }

    func testRegistrationCreatesOpencuaModel() throws {
        let creator = try XCTUnwrap(SharpAIOpencuaOverlay.creators["opencua"])
        let model = try creator(Data(SharpAIOpencuaTestFixtures.tinyJSON.utf8))

        XCTAssertTrue(model is SharpAIOpencuaModel)
    }

    func testRegistrationRejectsConfigWithoutTextConfig() throws {
        let creator = try XCTUnwrap(SharpAIOpencuaOverlay.creators["opencua"])

        XCTAssertThrowsError(try creator(Data(#"{"model_type": "opencua"}"#.utf8)))
    }

    func testRegistrationDoesNotShadowQwen25VL() {
        XCTAssertNil(SharpAIOpencuaOverlay.creators["qwen2_5_vl"])
    }

    // MARK: - Image path

    private enum ForwardTestError: Error {
        case notLogits
    }

    /// One 4x4 patch grid. Fixture patch shape: 3 channels, temporal 1, patch 16.
    private let imageGrid = THW(1, 4, 4)
    private let patchDim = 3 * 1 * 16 * 16
    /// media_placeholder_token_id in tinyJSON.
    private let placeholderId: Int32 = 6
    /// 2x2 spatial merge: 16 patches become 4 image tokens.
    private let mergedImageTokens = 4

    /// The OpenCUA processor contract, reproduced without a tokenizer. The single placeholder
    /// for an image expands to one token per merged patch, with no vision start/end wrapping.
    /// Layout: [1, 2, image x 4, 3].
    private func imagePromptTokens() -> MLXArray {
        var ids: [Int32] = [1, 2]
        ids += [Int32](repeating: placeholderId, count: mergedImageTokens)
        ids.append(3)
        return MLXArray(ids).expandedDimensions(axis: 0)
    }

    private func runImagePrompt(_ model: SharpAIOpencuaModel, pixels: MLXArray) throws -> LMOutput {
        let input = LMInput(
            text: .init(tokens: imagePromptTokens()),
            image: LMInput.ProcessedImage(pixels: pixels, frames: [imageGrid]))
        let cache = try model.newCache(parameters: nil)
        let result = try model.prepare(input, cache: cache, state: nil, prefill: .init())
        guard case .logits(let out) = result else {
            throw ForwardTestError.notLogits
        }
        return out
    }

    func testImageEndToEndForward() throws {
        let model = try makeModel()
        let tokens = imagePromptTokens()

        let first = try runImagePrompt(
            model, pixels: MLXRandom.normal([imageGrid.product, patchDim]))

        XCTAssertEqual(first.logits.shape, [1, tokens.dim(1), 128])
        // A NaN anywhere makes the max NaN, and NaN is not finite.
        XCTAssertTrue(first.logits.max().item(Float.self).isFinite)

        // The last position attends to the image tokens, so different pixels must change its logits.
        let second = try runImagePrompt(
            model, pixels: MLXRandom.normal([imageGrid.product, patchDim]))
        let lastFirst = first.logits[0..., -1, 0...]
        let lastSecond = second.logits[0..., -1, 0...]
        let diff = abs(lastFirst - lastSecond).max().item(Float.self)
        XCTAssertGreaterThan(diff, 0)
    }

    /// Real OpenCUA runs Qwen2ForCausalLM with 1D position_ids, so this 7-token prompt has rope
    /// delta 0. Qwen25VL.getRopeIndex gives the image tokens 2D M-RoPE positions instead, which
    /// makes the delta -2.
    /// Verified against: https://huggingface.co/xlangai/OpenCUA-7B/raw/main/modeling_opencua.py
    /// (`self.language_model = Qwen2ForCausalLM(config.text_config)`; position_ids are passed
    /// through from the caller) and .../config.json (text_config has no rope_scaling).
    /// Tried: a subclass override is not possible, because Qwen25VL.prepare calls the static
    /// `Qwen25VL.getRopeIndex` by name and reaches its language model through private members.
    /// Needed: a hook in Qwen25VL.swift (for example a flag that skips M-RoPE for 1D-position
    /// models), which is outside this batch's allowed files.
    func testImagePromptUsesOneDimensionalPositionsKnownGap() throws {
        let model = try makeModel()
        let out = try runImagePrompt(
            model, pixels: MLXRandom.normal([imageGrid.product, patchDim]))
        let delta = try XCTUnwrap(
            out.state?[LMOutput.Key<MLXArray>("qwen25vl.ropeDeltas")]
        ).item(Int.self)

        XCTExpectFailure(
            "KNOWN GAP: OpenCUA image positions must be 1D (modeling_opencua.py: Qwen2ForCausalLM with caller-supplied position_ids), but Qwen25VL.getRopeIndex (Libraries/MLXVLM/Models/Qwen25VL.swift, static, called from the private Qwen25VL.prepare) applies 2D M-RoPE to image tokens (delta -2 instead of 0). Needs a Qwen25VL.swift hook, which is outside this batch's allowed files."
        ) {
            XCTAssertEqual(delta, 0)
        }
    }
}
