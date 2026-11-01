// SharpAI fork overlay tests for granite_speech_plus audio placeholder merging. Not part of upstream.
// Uses the internal merge entry point, so this file imports the module with @testable.
// UNVERIFIED: this file was written without compiling. Run it before relying on it.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIGraniteSpeechPlusMergeTests: XCTestCase {

    private func tinyModel() throws -> SharpAIGraniteSpeechPlusForConditionalGeneration {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIGraniteSpeechPlusConfiguration.self,
            from: Data(SharpAIGraniteSpeechPlusFixtures.tinyJSON.utf8))
        return SharpAIGraniteSpeechPlusForConditionalGeneration(configuration)
    }

    func testAudioRowsReplacePlaceholdersInOrder() throws {
        let model = try tinyModel()
        // Token 63 is the audio placeholder in the fixture. Two placeholders, two audio rows.
        let tokens = MLXArray([Int32(1), 63, 2, 63]).expandedDimensions(axis: 0)
        let rowOne = [Float](repeating: 1, count: 32)
        let rowTwo = [Float](repeating: 2, count: 32)
        let audio = MLXArray(rowOne + rowTwo, [1, 2, 32])

        let merged = try model.mergeAudio(tokens: tokens, audioEmbeddings: audio)

        XCTAssertEqual(merged.shape, [1, 4, 32])
        XCTAssertEqual(merged[0, 1].asArray(Float.self), rowOne)
        XCTAssertEqual(merged[0, 3].asArray(Float.self), rowTwo)
    }

    func testPlaceholderCountMustMatchAudioRows() throws {
        let model = try tinyModel()
        let tokens = MLXArray([Int32(63), 63]).expandedDimensions(axis: 0)
        let audio = MLXArray([Float](repeating: 0, count: 3 * 32), [1, 3, 32])

        XCTAssertThrowsError(try model.mergeAudio(tokens: tokens, audioEmbeddings: audio)) {
            error in
            guard let typed = error as? SharpAIGraniteSpeechPlusError,
                case .audioTokenMismatch(let tokenCount, let featureCount) = typed
            else {
                XCTFail("expected audioTokenMismatch, got \(error)")
                return
            }
            XCTAssertEqual(tokenCount, 2)
            XCTAssertEqual(featureCount, 3)
        }
    }
}
