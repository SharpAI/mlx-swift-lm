// SharpAI fork overlay tests for the qwen4_exp PLE n-gram hashing. Not part of upstream.
//
// The expected values are the checkpoint's own buffers, read from
// model.language_model.layers.1.ple.ple_embedding.{layer_multipliers, ngram_heads_vocab_sizes,
// ngram_heads_offsets} in Qwen/Qwen3.8-Flash-Next (model-00005-of-00131.safetensors), and the
// reference Python functions _build_layer_multipliers and _find_nth_prime_after reproduce them.
// The shift case is worked out by hand from _shift_right_ignore_eos.

import Foundation
import MLXVLM
import XCTest

final class SharpAIQwen4ExpPLETests: XCTestCase {

    func testLayerMultipliersMatchCheckpoint() {
        let multipliers = SharpAIQwen4ExpNGramHashing.layerMultipliers(
            unigramVocabSize: 248_320, ngramSize: 3, pleLayerIndex: 0, seed: 1234)

        XCTAssertEqual(multipliers, [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071])
    }

    func testHeadVocabSizesAndOffsetsMatchCheckpoint() {
        let sizes = SharpAIQwen4ExpNGramHashing.headVocabSizes(
            base: 20_000_000, ngramHeads: 16, pleLayerIndex: 0)

        XCTAssertEqual(
            sizes,
            [
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063,
                20_000_069, 20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147,
                20_000_153, 20_000_159, 20_000_161, 20_000_171,
            ])
        XCTAssertEqual(
            SharpAIQwen4ExpNGramHashing.headVocabOffsets(sizes),
            [
                0, 20_000_003, 40_000_026, 60_000_059, 80_000_106, 100_000_165,
                120_000_228, 140_000_297, 160_000_374, 180_000_455, 200_000_548,
                220_000_655, 240_000_802, 260_000_955, 280_001_114, 300_001_275,
            ])
        // Padded table rows: ceil(total / 128) * 128 = 320001536, which is 128 shards of 2500012.
        let total = sizes.reduce(0, +)
        XCTAssertEqual((total + 127) / 128 * 128, 320_001_536)
    }

    func testShiftRightIgnoresSegmentsAcrossEOS() {
        // EOS is 2. Position 1 is EOS itself, so the token after it starts a new segment.
        let shifted = SharpAIQwen4ExpNGramHashing.shiftRightIgnoreEOS(
            [5, 2, 7, 8, 2, 9], shift: 1, eos: 2)

        XCTAssertEqual(shifted, [2, 5, 2, 7, 8, 2])
    }

    func testIsPrimeAndNthPrime() {
        XCTAssertTrue(SharpAIQwen4ExpNGramHashing.isPrime(19_999_999 + 4))
        XCTAssertEqual(SharpAIQwen4ExpNGramHashing.nthPrimeAfter(10, count: 1), 11)
        XCTAssertEqual(SharpAIQwen4ExpNGramHashing.nthPrimeAfter(10, count: 3), 17)
    }
}
