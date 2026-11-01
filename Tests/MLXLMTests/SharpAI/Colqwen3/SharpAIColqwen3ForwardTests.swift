import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// Tiny random-weight forward passes. Shapes and norms only; no real weights.
final class SharpAIColqwen3ForwardTests: XCTestCase {

    private func tinyModel() throws -> SharpAIColqwen3Model {
        SharpAIColqwen3Model(try SharpAIColqwen3TestFixtures.tinyConfiguration())
    }

    func testTextForwardShapeAndUnitNorm() throws {
        let model = try tinyModel()
        let ids = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])

        let out = try model.embed(inputIds: ids)
        eval(out)

        XCTAssertEqual(out.shape, [1, 4, 8])
        let norms = sqrt((out * out).sum(axis: -1)).asArray(Float.self)
        for norm in norms {
            XCTAssertEqual(norm, 1, accuracy: 1e-3)
        }
    }

    func testPaddedPositionIsZeroAndOutputFinite() throws {
        let model = try tinyModel()
        let ids = MLXArray([Int32(0), 4, 5, 1, 2, 3]).reshaped([2, 3])
        let mask = MLXArray([Int32(1), 1, 1, 0, 1, 1]).reshaped([2, 3])

        let out = try model.embed(inputIds: ids, attentionMask: mask)
        eval(out)

        XCTAssertEqual(out.shape, [2, 3, 8])
        let values = out.asArray(Float.self)
        // Row 1, position 0 is padding; its 8 values start at (1 * 3 + 0) * 8.
        XCTAssertEqual(Array(values[24 ..< 32]), Array(repeating: Float(0), count: 8))
        XCTAssertTrue(values.allSatisfy { $0.isFinite })
    }

    /// UNVERIFIED: the image path is checked only for shapes. The grid and token layout
    /// follow Qwen3VL conventions; no real ColQwen3 image has been run.
    func testImageForwardShape() throws {
        let model = try tinyModel()
        // One 4x4 patch grid merges to 4 visual tokens, bracketed by vision start and end.
        let ids = MLXArray([Int32(1), 11, 9, 9, 9, 9, 12, 2]).reshaped([1, 8])
        let pixels = MLXRandom.normal([16, 12])
        let mask = MLXArray([Int32](repeating: 1, count: 8)).reshaped([1, 8])

        let out = try model.embed(
            inputIds: ids,
            pixelValues: pixels,
            imageGridTHW: [THW(1, 4, 4)],
            attentionMask: mask)
        eval(out)

        XCTAssertEqual(out.shape, [1, 8, 8])
        XCTAssertTrue(out.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    /// Deepstack features must land on each row's own image tokens, so a batch of two
    /// documents embeds the same as the two documents embedded alone.
    func testBatchedDeepstackMatchesSingleRows() throws {
        let json = SharpAIColqwen3TestFixtures.tinyConfigJSON.replacingOccurrences(
            of: "\"deepstack_visual_indexes\": []", with: "\"deepstack_visual_indexes\": [0]")
        let config = try JSONDecoder().decode(
            SharpAIColqwen3Configuration.self, from: Data(json.utf8))
        let model = SharpAIColqwen3Model(config)

        let rowA: [Int32] = [1, 11, 9, 9, 9, 9, 12, 2]
        let rowB: [Int32] = [3, 11, 9, 9, 9, 9, 12, 4]
        let pixelsA = MLXRandom.normal([16, 12])
        let pixelsB = MLXRandom.normal([16, 12])
        let grid = THW(1, 4, 4)

        let batched = try model.embed(
            inputIds: MLXArray(rowA + rowB).reshaped([2, 8]),
            pixelValues: concatenated([pixelsA, pixelsB], axis: 0),
            imageGridTHW: [grid, grid])
        let singleA = try model.embed(
            inputIds: MLXArray(rowA).reshaped([1, 8]), pixelValues: pixelsA,
            imageGridTHW: [grid])
        let singleB = try model.embed(
            inputIds: MLXArray(rowB).reshaped([1, 8]), pixelValues: pixelsB,
            imageGridTHW: [grid])
        eval(batched, singleA, singleB)

        XCTAssertEqual(batched.shape, [2, 8, 8])
        let expected = concatenated([singleA, singleB], axis: 0)
        let diff = abs(batched - expected).max().item(Float.self)
        XCTAssertLessThan(diff, 1e-3)
    }

    private func deepstackModel() throws -> SharpAIColqwen3Model {
        let json = SharpAIColqwen3TestFixtures.tinyConfigJSON.replacingOccurrences(
            of: "\"deepstack_visual_indexes\": []", with: "\"deepstack_visual_indexes\": [0]")
        let config = try JSONDecoder().decode(
            SharpAIColqwen3Configuration.self, from: Data(json.utf8))
        return SharpAIColqwen3Model(config)
    }

    /// A text-only row batched with an image row must keep plain arange positions, and the
    /// image row must keep its own grid.
    func testBatchedTextOnlyRowMatchesSingleRows() throws {
        let model = try deepstackModel()
        let rowA: [Int32] = [1, 11, 9, 9, 9, 9, 12, 2]
        let rowB: [Int32] = [3, 4, 5, 6, 7, 8, 5, 4]
        let pixelsA = MLXRandom.normal([16, 12])
        let grid = THW(1, 4, 4)
        let mask = MLXArray([Int32](repeating: 1, count: 16)).reshaped([2, 8])

        let batched = try model.embed(
            inputIds: MLXArray(rowA + rowB).reshaped([2, 8]),
            pixelValues: pixelsA,
            imageGridTHW: [grid],
            attentionMask: mask)
        let singleA = try model.embed(
            inputIds: MLXArray(rowA).reshaped([1, 8]), pixelValues: pixelsA,
            imageGridTHW: [grid])
        let singleB = try model.embed(inputIds: MLXArray(rowB).reshaped([1, 8]))
        eval(batched, singleA, singleB)

        let expected = concatenated([singleA, singleB], axis: 0)
        let diff = abs(batched - expected).max().item(Float.self)
        XCTAssertLessThan(diff, 1e-3)
    }

    /// Rows with different image sizes (4 and 2 visual tokens) need right padding; the second
    /// row must use its own grid, not the first one, and only valid positions are compared.
    func testBatchedDifferentGridsMatchSingleRows() throws {
        let model = try deepstackModel()
        let rowA: [Int32] = [1, 11, 9, 9, 9, 9, 12, 2]
        let rowC: [Int32] = [3, 11, 9, 9, 12, 4]
        let paddedC: [Int32] = rowC + [0, 0]
        let pixelsA = MLXRandom.normal([16, 12])
        let pixelsC = MLXRandom.normal([8, 12])
        let gridA = THW(1, 4, 4)
        let gridC = THW(1, 2, 4)
        let mask = MLXArray(
            [Int32](repeating: 1, count: 8) + [Int32](repeating: 1, count: 6) + [0, 0]
        ).reshaped([2, 8])

        let batched = try model.embed(
            inputIds: MLXArray(rowA + paddedC).reshaped([2, 8]),
            pixelValues: concatenated([pixelsA, pixelsC], axis: 0),
            imageGridTHW: [gridA, gridC],
            attentionMask: mask)
        let singleA = try model.embed(
            inputIds: MLXArray(rowA).reshaped([1, 8]), pixelValues: pixelsA,
            imageGridTHW: [gridA])
        let singleC = try model.embed(
            inputIds: MLXArray(rowC).reshaped([1, 6]), pixelValues: pixelsC,
            imageGridTHW: [gridC])
        eval(batched, singleA, singleC)

        let diffA = abs(batched[0 ..< 1, 0..., 0...] - singleA).max().item(Float.self)
        let diffC = abs(batched[1 ..< 2, 0 ..< 6, 0...] - singleC).max().item(Float.self)
        XCTAssertLessThan(diffA, 1e-3)
        XCTAssertLessThan(diffC, 1e-3)
    }

    func testPixelValuesWithoutGridThrows() throws {
        let model = try tinyModel()
        let ids = MLXArray([Int32(1), 11, 9, 9, 9, 9, 12, 2]).reshaped([1, 8])
        let pixels = MLXRandom.normal([16, 12])

        XCTAssertThrowsError(try model.embed(inputIds: ids, pixelValues: pixels))
    }

    func testMaxSimMatchesHandComputedValue() {
        let query = MLXArray([Float(1), 0, 0, 1]).reshaped([2, 2])
        let document = MLXArray([Float(1), 0, 0, 1]).reshaped([2, 2])

        let score = SharpAIColqwen3Scoring.maxSim(query: query, document: document)

        XCTAssertEqual(score.item(Float.self), 2, accuracy: 1e-5)
    }

    func testBatchedScoresExcludePaddedDocumentRows() {
        // Doc 1 has one real row (-1, -1) and one padded zero row that must not count.
        let queries = MLXArray([Float(1), 0, 0, 1]).reshaped([1, 2, 2])
        let documents = MLXArray([Float(1), 0, 0, 1, -1, -1, 0, 0]).reshaped([2, 2, 2])
        let queryMask = MLXArray([Int32(1), 1]).reshaped([1, 2])
        let documentMask = MLXArray([Int32(1), 1, 1, 0]).reshaped([2, 2])

        let scores = SharpAIColqwen3Scoring.scores(
            queries: queries,
            documents: documents,
            queryMask: queryMask,
            documentMask: documentMask)
        eval(scores)

        XCTAssertEqual(scores.shape, [1, 2])
        let values = scores.asArray(Float.self)
        XCTAssertEqual(values[0], 2, accuracy: 1e-5)
        XCTAssertEqual(values[1], -2, accuracy: 1e-5)
    }
}
