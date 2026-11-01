// SharpAI fork overlay tests for qwen3_asr length math, prompt, output parsing, and alignment.
// Pure Swift: no MLX arrays and no weights. Not part of upstream.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLXVLM
import XCTest

final class SharpAIQwen3AsrTranscriptionTests: XCTestCase {

    func testAudioTokenCountMatchesHubFormula() {
        XCTAssertEqual(SharpAIQwen3AsrAudioMath.audioTokenCount(frames: 100, nWindow: 50), 13)
        XCTAssertEqual(SharpAIQwen3AsrAudioMath.audioTokenCount(frames: 3000, nWindow: 50), 390)
        XCTAssertEqual(SharpAIQwen3AsrAudioMath.audioTokenCount(frames: 0, nWindow: 50), 0)
        XCTAssertEqual(SharpAIQwen3AsrAudioMath.audioTokenCount(frames: 50, nWindow: 50), 7)
        // One full chunk (13) plus a half chunk (7).
        XCTAssertEqual(SharpAIQwen3AsrAudioMath.audioTokenCount(frames: 150, nWindow: 50), 20)
    }

    func testPostConvLengthOfFullChunkIsThirteen() {
        XCTAssertEqual(SharpAIQwen3AsrAudioMath.postConvLength(100), 13)
        XCTAssertEqual(SharpAIQwen3AsrAudioMath.postConvLength(0), 0)
    }

    func testWindowBoundsSplitTail() {
        let bounds = SharpAIQwen3AsrAudioMath.windowBounds(total: 13, windowTokens: 5)
        XCTAssertEqual(bounds.map { $0.0 }, [0, 5, 10])
        XCTAssertEqual(bounds.map { $0.1 }, [5, 10, 13])
    }

    func testPromptWrapsOnePlaceholderPerToken() throws {
        let prompt = try SharpAIQwen3AsrTranscription.prompt(audioTokenCount: 2, language: "en")

        XCTAssertTrue(
            prompt.contains("<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|>"))
        XCTAssertTrue(prompt.hasSuffix("language English<asr_text>"))
    }

    func testPromptWithoutLanguageEndsAtAssistantTurn() throws {
        let prompt = try SharpAIQwen3AsrTranscription.prompt(audioTokenCount: 1)
        XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n"))
    }

    func testPromptRejectsUnknownLanguage() {
        XCTAssertThrowsError(
            try SharpAIQwen3AsrTranscription.prompt(audioTokenCount: 1, language: "xx"))
    }

    func testLanguageNameAcceptsCodeOrName() {
        XCTAssertEqual(SharpAIQwen3AsrTranscription.languageName(for: "zh"), "Chinese")
        XCTAssertEqual(SharpAIQwen3AsrTranscription.languageName(for: "Korean"), "Korean")
        XCTAssertNil(SharpAIQwen3AsrTranscription.languageName(for: "klingon"))
    }

    func testParseLanguageAndText() {
        XCTAssertEqual(
            SharpAIQwen3AsrTranscription.parse("language English<asr_text>Hello world."),
            SharpAIQwen3AsrTranscript(language: "English", transcription: "Hello world."))
    }

    func testParseLanguageNoneGivesNilLanguage() {
        XCTAssertEqual(
            SharpAIQwen3AsrTranscription.parse("language None<asr_text>hi"),
            SharpAIQwen3AsrTranscript(language: nil, transcription: "hi"))
    }

    func testParseWithoutTagKeepsWholeText() {
        XCTAssertEqual(
            SharpAIQwen3AsrTranscription.parse("just text"),
            SharpAIQwen3AsrTranscript(language: nil, transcription: "just text"))
    }

    func testParseEmptyInput() {
        XCTAssertEqual(
            SharpAIQwen3AsrTranscription.parse("   "),
            SharpAIQwen3AsrTranscript(language: nil, transcription: ""))
    }

    func testParseDropsEverythingBeforeAssistantTurn() {
        let raw = "system prompt\nassistant\nlanguage Chinese<asr_text>你好"
        XCTAssertEqual(
            SharpAIQwen3AsrTranscription.parse(raw),
            SharpAIQwen3AsrTranscript(language: "Chinese", transcription: "你好"))
    }

    func testParseCollapsesLongCharacterRuns() {
        let run = String(repeating: "a", count: 25)
        XCTAssertEqual(
            SharpAIQwen3AsrTranscription.parse("language English<asr_text>\(run)bc").transcription,
            "abc")
    }

    func testFixTimestampsReplacesOutlierWithNeighbour() {
        XCTAssertEqual(
            SharpAIQwen3AsrTranscription.fixTimestamps([0, 80, 160, 5, 240]),
            [0, 80, 160, 160, 240])
    }

    func testFixTimestampsInterpolatesLongOutlierRun() {
        // LIS keeps 0, 50, 400. The three falling outliers between 50 and 400 interpolate
        // with step 87.5: 137.5, 225, 312.5, truncated to Int.
        XCTAssertEqual(
            SharpAIQwen3AsrTranscription.fixTimestamps([0, 50, 40, 30, 20, 400]),
            [0, 50, 137, 225, 312, 400])
    }

    func testAlignmentItemsConvertBinsToSeconds() throws {
        let items = try SharpAIQwen3AsrTranscription.alignmentItems(
            words: ["a", "b"], predictedBins: [0, 1, 1, 2], segmentTimeMs: 80)

        XCTAssertEqual(items.map { $0.text }, ["a", "b"])
        XCTAssertEqual(items[0].startTime, 0.0, accuracy: 1e-9)
        XCTAssertEqual(items[0].endTime, 0.08, accuracy: 1e-9)
        XCTAssertEqual(items[1].startTime, 0.08, accuracy: 1e-9)
        XCTAssertEqual(items[1].endTime, 0.16, accuracy: 1e-9)
    }

    func testAlignmentItemsRejectTooFewTimestamps() {
        XCTAssertThrowsError(
            try SharpAIQwen3AsrTranscription.alignmentItems(
                words: ["a", "b"], predictedBins: [0, 1], segmentTimeMs: 80))
    }

    func testForcedAlignerLanguagesIncludeEnglish() {
        XCTAssertTrue(SharpAIQwen3AsrTranscription.forcedAlignerLanguages.contains("English"))
        XCTAssertFalse(SharpAIQwen3AsrTranscription.forcedAlignerLanguages.contains("Thai"))
    }
}
