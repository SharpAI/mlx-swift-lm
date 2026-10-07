import Foundation
import MLX
@testable import MLXLMCommon
import MLXNN
import XCTest

/// `QuantizedSwitchLinear.computeExpertsFused` (the opt-in `MLX_MOE_STACKED=1` path) feeds
/// `gatherQuantizedMM` a `slotPerToken` of LRU cache-slot ids. Those are grouped into runs but are
/// NOT ascending. mlx core 0.32 made `sortedIndices: true` require ascending ids for large inputs,
/// so declaring them sorted produced garbage for prefill-sized batches.
final class StackedExpertSlotOrderTests: XCTestCase {

    func testFusedStackedComputeIsCorrectForNonAscendingSlots() throws {
        let numExperts = 8, inputDims = 128, outputDims = 64
        MLXRandom.seed(7)

        let dense = SwitchLinear(
            inputDims: inputDims, outputDims: outputDims, numExperts: numExperts, bias: false)
        // SwitchLinear initialises its weights to zero, which would make every result trivially 0.
        dense.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXRandom.normal([numExperts, outputDims, inputDims])
            ]))
        let layer = QuantizedSwitchLinear(dense, groupSize: 64, bits: 4, mode: .affine)

        // Slot s holds expert slotExperts[s]; the order is deliberately not ascending.
        let slotExperts: [Int32] = [5, 2, 7, 0, 3, 6, 1, 4]
        let stacked = layer.weight[MLXArray(slotExperts).asType(.uint32)]

        // Prefill-sized batch (>= 64 rows), grouped into runs of equal slot like the real caller,
        // with the run order scrambled so slot ids are not ascending (LRU reuse).
        let tokens = 256
        let runSlots: [UInt32] = [3, 0, 6, 1, 7, 2, 5, 4]
        let slotPerToken = (0 ..< tokens).map { runSlots[$0 * numExperts / tokens] }
        let x = MLXRandom.normal([tokens, 1, inputDims])

        let out = layer.computeExpertsFused(
            x, stackedBuffer: stacked, slotPerToken: MLXArray(slotPerToken),
            slotExperts: slotExperts)

        // Reference: dequantize each token's expert and multiply directly.
        var rows = [MLXArray]()
        for t in 0 ..< tokens {
            let e = Int(slotExperts[Int(slotPerToken[t])])
            let w = dequantized(
                layer.weight[e], scales: layer.scales[e], biases: layer.biases![e],
                groupSize: 64, bits: 4)
            rows.append(matmul(x[t], w.T))
        }
        let expected = MLX.stacked(rows, axis: 0)

        // Relative tolerance: a misordered-index kernel result is uncorrelated with the reference
        // (error ~ the output magnitude), while a correct one is within quantised-matmul rounding.
        let maxErr = abs(out - expected).max().item(Float.self)
        let scale = abs(expected).max().item(Float.self)
        XCTAssertLessThan(
            maxErr / scale, 0.02, "fused stacked output diverges from per-expert reference")
    }
}
