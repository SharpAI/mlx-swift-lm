// Deterministic, air-gapped tests for the shared Whisper-style log-mel front end.
// No weights, no network. Nothing in the repo consumes the extractor yet.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class WhisperLogMelExtractorTests: XCTestCase {

    private func sine(hz: Double, seconds: Double, amplitude: Float = 0.5, rate: Int = 16_000)
        -> [Float]
    {
        (0 ..< Int(seconds * Double(rate))).map {
            amplitude * Float(sin(2 * Double.pi * hz * Double($0) / Double(rate)))
        }
    }

    func testDefaultConstantsMatchWhisper() {
        let c = WhisperLogMelConfiguration()
        XCTAssertEqual(c.sampleRate, 16_000)
        XCTAssertEqual(c.nFft, 400)
        XCTAssertEqual(c.hopLength, 160)
        XCTAssertEqual(c.nMels, 80)
        XCTAssertEqual(c.chunkSeconds, 30)
    }

    func testOneSecondPadsToThirtySecondChunk() throws {
        let mel = try WhisperLogMelExtractor().logMel(samples: sine(hz: 440, seconds: 1))
        XCTAssertEqual(mel.shape, [80, 3000])
    }

    func testUnpaddedShapeIsSamplesOverHop() throws {
        let mel = try WhisperLogMelExtractor().logMel(
            samples: sine(hz: 440, seconds: 1), padToChunk: false)
        XCTAssertEqual(mel.shape, [80, 100])
    }

    func testStrictlyMoreThanOneChunkPadsToTwo() throws {
        let extractor = WhisperLogMelExtractor(.init(nMels: 128))
        let mel = try extractor.logMel(samples: [Float](repeating: 0.1, count: 480_001))
        XCTAssertEqual(mel.shape, [128, 6000])
    }

    func testSilenceHitsTheFloorExactly() throws {
        // log10(1e-10) = -10, every cell equals the peak, so (-10 + 4) / 4 = -1.5.
        let mel = try WhisperLogMelExtractor().logMel(
            samples: [Float](repeating: 0, count: 16_000), padToChunk: false)
        XCTAssertEqual(mel.min().item(Float.self), -1.5, accuracy: 1e-6)
        XCTAssertEqual(mel.max().item(Float.self), -1.5, accuracy: 1e-6)
    }

    func testThousandHertzToneLandsOnItsMelBin() throws {
        let mel = try WhisperLogMelExtractor().logMel(
            samples: sine(hz: 1000, seconds: 2), padToChunk: false)
        // Middle frame, away from the reflect-padded edges.
        let column = mel[0..., 100].asArray(Float.self)
        let peak = column.indices.max { column[$0] < column[$1] }!
        // Slaney scale: 1000 Hz is mel 15; 80 filters span mel 0 ... 45.24 in 81 steps,
        // so the filter centred nearest 1000 Hz is index 26 (centre mel 15.08).
        XCTAssertEqual(peak, 26, accuracy: 1)
        // The tone sits well above the rest of the spectrum.
        let sorted = column.sorted()
        XCTAssertGreaterThan(sorted[79] - sorted[40], 0.5)
    }

    func testHigherToneLandsOnHigherBin() throws {
        let extractor = WhisperLogMelExtractor()
        func peak(_ hz: Double) throws -> Int {
            let mel = try extractor.logMel(samples: sine(hz: hz, seconds: 2), padToChunk: false)
            let col = mel[0..., 100].asArray(Float.self)
            return col.indices.max { col[$0] < col[$1] }!
        }
        let low = try peak(300)
        let mid = try peak(1000)
        let high = try peak(4000)
        XCTAssertLessThan(low, mid)
        XCTAssertLessThan(mid, high)
    }

    func testDynamicRangeIsEightDecadesOverFour() throws {
        let mel = try WhisperLogMelExtractor().logMel(
            samples: sine(hz: 1000, seconds: 1), padToChunk: false)
        let range = mel.max().item(Float.self) - mel.min().item(Float.self)
        XCTAssertLessThanOrEqual(range, 2.0 + 1e-5)
        XCTAssertGreaterThan(range, 0.1)
    }

    func testTenfoldAmplitudeShiftsOutputByHalf() throws {
        // Power x100 -> log10 +2 -> (x + 4) / 4 shifts by 0.5. Only valid while the floor
        // is not the peak, which holds for a pure tone.
        let extractor = WhisperLogMelExtractor()
        let base = try extractor.logMel(
            samples: sine(hz: 1000, seconds: 1, amplitude: 0.05), padToChunk: false)
        let loud = try extractor.logMel(
            samples: sine(hz: 1000, seconds: 1, amplitude: 0.5), padToChunk: false)
        let delta = loud.max().item(Float.self) - base.max().item(Float.self)
        XCTAssertEqual(delta, 0.5, accuracy: 1e-3)
    }

    func testHannWindowIsPeriodic() {
        let w = WhisperLogMelExtractor.hannWindow(nFft: 400)
        XCTAssertEqual(w.count, 400)
        XCTAssertEqual(w[0], 0, accuracy: 1e-7)
        XCTAssertEqual(w[200], 1, accuracy: 1e-6)
        XCTAssertEqual(w[100], 0.5, accuracy: 1e-6)
        XCTAssertEqual(w[399], w[1], accuracy: 1e-6)
    }

    func testSlaneyMelScaleBreakpoints() {
        XCTAssertEqual(WhisperLogMelExtractor.hzToMel(0), 0, accuracy: 1e-9)
        XCTAssertEqual(WhisperLogMelExtractor.hzToMel(200), 3, accuracy: 1e-9)
        XCTAssertEqual(WhisperLogMelExtractor.hzToMel(1000), 15, accuracy: 1e-9)
        // 15 + ln(8) * 27 / ln(6.4): the log segment above 1 kHz.
        XCTAssertEqual(
            WhisperLogMelExtractor.hzToMel(8000), 15 + log(8.0) * 27 / log(6.4), accuracy: 1e-9)
        XCTAssertEqual(
            WhisperLogMelExtractor.melToHz(WhisperLogMelExtractor.hzToMel(4000)), 4000,
            accuracy: 1e-6)
    }

    func testFilterBankIsNonNegativeAndEveryFilterHasWeight() {
        for nMels in [80, 128] {
            let c = WhisperLogMelConfiguration(nMels: nMels)
            let bins = c.nFft / 2 + 1
            let f = WhisperLogMelExtractor.melFilterBank(c)
            XCTAssertEqual(f.count, bins * nMels)
            XCTAssertTrue(f.allSatisfy { $0 >= 0 })
            for m in 0 ..< nMels {
                XCTAssertTrue((0 ..< bins).contains { f[$0 * nMels + m] > 0 }, "mel \(m) empty")
            }
        }
    }

    func testReflectPadMirrorsBothEdges() {
        XCTAssertEqual(
            WhisperLogMelExtractor.reflectPad([1, 2, 3, 4], pad: 2), [3, 2, 1, 2, 3, 4, 3, 2])
    }

    func testEmptyAndTooShortInputThrow() {
        let extractor = WhisperLogMelExtractor()
        XCTAssertThrowsError(try extractor.logMel(samples: []))
        XCTAssertThrowsError(try extractor.logMel(samples: [0.1, 0.2], padToChunk: false))
    }
}
