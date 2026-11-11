// SharpAI fork overlay tests for the nemotron_parse processor. Not part of upstream.
// The expected values are worked out by hand from hf_nemotron_parse_processor.py: no reference
// implementation runs air-gapped, so pixel values are NOT compared with cv2.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// One token per UTF-8 byte, so a test can read the prompt back from the token ids.
private struct ByteTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map { UInt8(truncatingIfNeeded: $0) }, as: UTF8.self)
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

final class SharpAINemotronParseProcessorTests: XCTestCase {

    private func solid(_ color: CIColor, width: Int, height: Int) -> UserInput.Image {
        .ciImage(CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: width, height: height)))
    }

    private func processor(_ json: String) throws -> SharpAINemotronParseProcessor {
        try SharpAINemotronParseOverlay.makeProcessor(
            configurationData: Data(json.utf8), tokenizer: ByteTokenizer())
    }

    // MARK: Configuration

    func testHubPreprocessorConfigurationIsRead() throws {
        let configuration = try JSONDecoder.json5().decode(
            SharpAINemotronParseImageConfiguration.self,
            from: Data(SharpAINemotronParseTestFixtures.hubPreprocessorJSON.utf8))
        XCTAssertEqual(configuration.finalHeight, 2048)
        XCTAssertEqual(configuration.finalWidth, 1648)
        XCTAssertTrue(configuration.doNormalize)
        XCTAssertEqual(configuration.imageMean, [0.48145466, 0.4578275, 0.40821073])
    }

    func testDefaultsMatchTheReferenceConstructor() throws {
        let configuration = try JSONDecoder.json5().decode(
            SharpAINemotronParseImageConfiguration.self, from: Data("{}".utf8))
        XCTAssertEqual(configuration.finalHeight, 2048)
        XCTAssertEqual(configuration.finalWidth, 1648)
        // The class default is do_normalize False, and CLIP mean and std when none are given.
        XCTAssertFalse(configuration.doNormalize)
        XCTAssertEqual(configuration.imageStd, [0.26862954, 0.26130258, 0.27577711])
    }

    func testSizeKeysFollowTheReferencePrecedence() throws {
        func decode(_ json: String) throws -> SharpAINemotronParseImageConfiguration {
            try JSONDecoder.json5().decode(
                SharpAINemotronParseImageConfiguration.self, from: Data(json.utf8))
        }
        // longest_edge as a pair wins over height and width.
        var c = try decode(#"{"size": {"height": 10, "width": 20, "longest_edge": [30, 40]}}"#)
        XCTAssertEqual([c.finalHeight, c.finalWidth], [30, 40])
        // longest_edge as one number is a square.
        c = try decode(#"{"size": {"longest_edge": 64}}"#)
        XCTAssertEqual([c.finalHeight, c.finalWidth], [64, 64])
        // height and width alone.
        c = try decode(#"{"size": {"height": 48, "width": 32}}"#)
        XCTAssertEqual([c.finalHeight, c.finalWidth], [48, 32])
        // size overrides final_size.
        c = try decode(#"{"final_size": [1, 2], "size": {"height": 48, "width": 32}}"#)
        XCTAssertEqual([c.finalHeight, c.finalWidth], [48, 32])
        // a single number in final_size is a square.
        c = try decode(#"{"final_size": 96}"#)
        XCTAssertEqual([c.finalHeight, c.finalWidth], [96, 96])
    }

    // MARK: Geometry

    func testFittedSizeShrinksButNeverEnlarges() {
        // Tall: 4000x1000 (HxW) is cut to 2048 high, width int(2048 * 0.25) = 512.
        var size = SharpAINemotronParseProcessor.fittedSize(
            height: 4000, width: 1000, maxHeight: 2048, maxWidth: 1648)
        XCTAssertEqual([size.height, size.width], [2048, 512])
        // Wide: 1000x4000 keeps its height until the width cap, then width 1648, height int(1648 / 4) = 412.
        size = SharpAINemotronParseProcessor.fittedSize(
            height: 1000, width: 4000, maxHeight: 2048, maxWidth: 1648)
        XCTAssertEqual([size.height, size.width], [412, 1648])
        // Small images are left alone.
        size = SharpAINemotronParseProcessor.fittedSize(
            height: 500, width: 300, maxHeight: 2048, maxWidth: 1648)
        XCTAssertEqual([size.height, size.width], [500, 300])
        // Truncation, not rounding: the width is int(7 * 1.5) = int(10.5) = 10.
        size = SharpAINemotronParseProcessor.fittedSize(
            height: 14, width: 21, maxHeight: 7, maxWidth: 1000)
        XCTAssertEqual([size.height, size.width], [7, 10])
    }

    func testPaddingIsCentredWithTheExtraOnTheBottomAndRight() {
        // 1x1 image into 4x5: pad height 3 gives top 1 and bottom 2, pad width 4 gives left 2 and right 2.
        let image = MLXArray.zeros([1, 3, 1, 1])
        let padded = SharpAINemotronParseProcessor.padWhite(image, toHeight: 4, width: 5)
        XCTAssertEqual(padded.shape, [1, 3, 4, 5])
        // Only (row 1, column 2) is the original zero.
        for row in 0 ..< 4 {
            for column in 0 ..< 5 {
                let expected: Float = (row == 1 && column == 2) ? 0 : 255
                XCTAssertEqual(padded[0, 0, row, column].item(Float.self), expected, "row \(row) column \(column)")
            }
        }
    }

    func testBilinearResizeUsesHalfPixelCentres() {
        // One row [0, 10, 20, 30] to width 2: source coordinates 0.5 and 2.5, so [5, 25].
        let row = MLXArray([0, 10, 20, 30] as [Float]).reshaped(1, 1, 1, 4)
        let resized = SharpAINemotronParseProcessor.bilinearResize(row, height: 1, width: 2)
        XCTAssertEqual(resized.shape, [1, 1, 1, 2])
        XCTAssertEqual(resized.asArray(Float.self), [5, 25])

        // Width 4 to 8: the first output samples at -0.25 and clamps to 0, the last clamps to 30.
        let wide = SharpAINemotronParseProcessor.bilinearResize(row, height: 1, width: 8)
        XCTAssertEqual(wide.asArray(Float.self), [0, 3, 8, 13, 18, 23, 28, 30])
    }

    func testBilinearResizeKeepsAConstantImageConstant() {
        let constant = MLXArray.full([1, 3, 7, 5], values: MLXArray(Float(200)))
        let resized = SharpAINemotronParseProcessor.bilinearResize(constant, height: 3, width: 2)
        XCTAssertEqual(resized.shape, [1, 3, 3, 2])
        XCTAssertEqual(MLX.abs(resized - 200).max().item(Float.self), 0)
    }

    // MARK: Pipeline

    /// A 16x16 black image into a 32x32 canvas: white border, black centre, scaled to [0, 1] only.
    func testBlackImageIsCentredOnAWhiteCanvas() async throws {
        let json = #"{"final_size": [32, 32], "do_normalize": false}"#
        var input = UserInput(prompt: "")
        input.images = [solid(CIColor.black, width: 16, height: 16)]
        let lm = try await processor(json).prepare(input: input)
        let pixels = try XCTUnwrap(lm.image?.pixels)
        XCTAssertEqual(pixels.shape, [1, 3, 32, 32])
        XCTAssertEqual(pixels[0, 0, 0, 0].item(Float.self), 1, accuracy: 1e-6)  // border: white
        XCTAssertEqual(pixels[0, 1, 31, 31].item(Float.self), 1, accuracy: 1e-6)
        XCTAssertEqual(pixels[0, 2, 16, 16].item(Float.self), 0, accuracy: 1e-6)  // centre: black
        XCTAssertEqual(pixels[0, 0, 8, 8].item(Float.self), 0, accuracy: 1e-6)  // first original pixel
        XCTAssertEqual(pixels[0, 0, 7, 8].item(Float.self), 1, accuracy: 1e-6)  // one above it is padding
    }

    func testNormalisationUsesTheConfiguredMeanAndStd() async throws {
        let json = #"{"final_size": [16, 16], "do_normalize": true, "image_mean": [0.5, 0.25, 0.0], "image_std": [0.5, 0.25, 1.0]}"#
        var input = UserInput(prompt: "")
        input.images = [solid(CIColor.white, width: 16, height: 16)]
        let lm = try await processor(json).prepare(input: input)
        let pixels = try XCTUnwrap(lm.image?.pixels)
        // White is 1 everywhere: (1 - 0.5) / 0.5 = 1, (1 - 0.25) / 0.25 = 3, (1 - 0) / 1 = 1.
        XCTAssertEqual(pixels[0, 0, 5, 5].item(Float.self), 1, accuracy: 1e-5)
        XCTAssertEqual(pixels[0, 1, 5, 5].item(Float.self), 3, accuracy: 1e-5)
        XCTAssertEqual(pixels[0, 2, 5, 5].item(Float.self), 1, accuracy: 1e-5)
    }

    func testAnOversizedImageIsShrunkToFitBeforePadding() async throws {
        let json = #"{"final_size": [16, 32], "do_normalize": false}"#
        var input = UserInput(prompt: "")
        input.images = [solid(CIColor.black, width: 64, height: 64)]  // square, so height is the binding side
        let lm = try await processor(json).prepare(input: input)
        let pixels = try XCTUnwrap(lm.image?.pixels)
        XCTAssertEqual(pixels.shape, [1, 3, 16, 32])
        // 64x64 shrinks to 16x16, then 8 columns of white pad each side.
        XCTAssertEqual(pixels[0, 0, 8, 7].item(Float.self), 1, accuracy: 1e-6)
        XCTAssertEqual(pixels[0, 0, 8, 8].item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(pixels[0, 0, 8, 23].item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(pixels[0, 0, 8, 24].item(Float.self), 1, accuracy: 1e-6)
    }

    // MARK: Prompt

    func testTheDefaultTaskPromptIsUsedForPlainText() async throws {
        var input = UserInput(prompt: "Please parse this page")
        input.images = [solid(CIColor.white, width: 16, height: 16)]
        let lm = try await processor(#"{"final_size": [16, 16]}"#).prepare(input: input)
        let text = ByteTokenizer().decode(tokenIds: lm.text.tokens.asArray(Int.self), skipSpecialTokens: false)
        XCTAssertEqual(text, "</s><s><predict_bbox><predict_classes><output_markdown>")
        XCTAssertEqual(lm.text.tokens.shape, [1, text.utf8.count])
    }

    func testATaskPromptIsPassedThroughVerbatim() async throws {
        let prompt = "</s><s><predict_bbox><predict_classes><output_no_text>"
        var input = UserInput(prompt: prompt)
        input.images = [solid(CIColor.white, width: 16, height: 16)]
        let lm = try await processor(#"{"final_size": [16, 16]}"#).prepare(input: input)
        let text = ByteTokenizer().decode(tokenIds: lm.text.tokens.asArray(Int.self), skipSpecialTokens: false)
        XCTAssertEqual(text, prompt)
    }

    // MARK: Errors

    func testNoImageIsRejected() async throws {
        do {
            _ = try await processor("{}").prepare(input: UserInput(prompt: "hi"))
            XCTFail("expected missingImage")
        } catch SharpAINemotronParseError.missingImage {
        }
    }

    func testTwoImagesAreRejected() async throws {
        var input = UserInput(prompt: "hi")
        input.images = [
            solid(CIColor.white, width: 16, height: 16), solid(CIColor.white, width: 16, height: 16),
        ]
        do {
            _ = try await processor("{}").prepare(input: input)
            XCTFail("expected needsExactlyOneImage")
        } catch SharpAINemotronParseProcessorError.needsExactlyOneImage(let count) {
            XCTAssertEqual(count, 2)
        }
    }

    func testTheProcessorClassIsTheCheckpointsName() {
        XCTAssertEqual(SharpAINemotronParseProcessor.processorClass, "NemotronParseProcessor")
    }
}
