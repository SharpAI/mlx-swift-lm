// SharpAI fork overlay tests for the glmasr 128-mel front end. No network, no weights.

import Foundation
import MLX
@testable import MLXVLM
import XCTest

final class SharpAIGlmasrFeatureExtractorTests: XCTestCase {

    func testSlaneyMelScaleBreakpoints() {
        XCTAssertEqual(SharpAIGlmasrFeatureExtractor.hzToMel(0), 0, accuracy: 1e-9)
        XCTAssertEqual(SharpAIGlmasrFeatureExtractor.hzToMel(1000), 15, accuracy: 1e-9)
        XCTAssertEqual(SharpAIGlmasrFeatureExtractor.hzToMel(200), 3, accuracy: 1e-9)
        let roundTrip = SharpAIGlmasrFeatureExtractor.melToHz(
            SharpAIGlmasrFeatureExtractor.hzToMel(4_000))
        XCTAssertEqual(roundTrip, 4_000, accuracy: 1e-6)
    }

    func testFilterBankHasOneTriangleFilterPerMelBin() {
        let bins = SharpAIGlmasrFeatureExtractor.nFft / 2 + 1
        let filters = SharpAIGlmasrFeatureExtractor.melFilterBank(numMelBins: 128)

        XCTAssertEqual(filters.count, bins * 128)
        XCTAssertTrue(filters.allSatisfy { $0 >= 0 })
        for mel in 0 ..< 128 {
            let column = (0 ..< bins).map { filters[$0 * 128 + mel] }
            XCTAssertTrue(column.contains { $0 > 0 }, "mel bin \(mel) has no weight")
        }
    }

    func testReflectPadMirrorsBothEdges() {
        let padded = SharpAIGlmasrFeatureExtractor.reflectPad([1, 2, 3, 4], pad: 2)
        XCTAssertEqual(padded, [3, 2, 1, 2, 3, 4, 3, 2])
    }

    func testOneSecondPadsToThirtySecondChunk() throws {
        let extractor = SharpAIGlmasrFeatureExtractor()
        let features = try extractor.logMel(samples: [Float](repeating: 0, count: 16_000))

        // 480000 padded samples, 1 + 480000 / 160 frames, last frame dropped.
        XCTAssertEqual(features.shape, [128, 3_000])
    }

    func testLongerAudioUsesWholeChunks() throws {
        let extractor = SharpAIGlmasrFeatureExtractor()
        let features = try extractor.logMel(
            samples: [Float](repeating: 0, count: 31 * 16_000))

        XCTAssertEqual(features.shape, [128, 6_000])
    }

    func testSilenceFloorsToKnownValue() throws {
        let extractor = SharpAIGlmasrFeatureExtractor()
        let features = try extractor.logMel(samples: [Float](repeating: 0, count: 16_000))

        // Silence gives log10(1e-10) = -10 everywhere. Then (-10 + 4) / 4 = -1.5.
        XCTAssertEqual(features.max().item(Float.self), -1.5, accuracy: 1e-4)
        XCTAssertEqual(features.min().item(Float.self), -1.5, accuracy: 1e-4)
    }

    func testInputFeaturesAddBatchAxis() throws {
        let extractor = SharpAIGlmasrFeatureExtractor()
        let features = try extractor.inputFeatures(
            samples: [Float](repeating: 0, count: 16_000))

        XCTAssertEqual(features.shape, [1, 128, 3_000])
    }

    func testEmptyAudioThrows() {
        let extractor = SharpAIGlmasrFeatureExtractor()
        XCTAssertThrowsError(try extractor.logMel(samples: []))
    }
}
