// SharpAI fork overlay: inkling_mm_model vision and audio towers. Not part of upstream mlx-swift-lm.
//
// Ported from transformers `modeling_inkling.py` (Verified against:
// https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/inkling/modeling_inkling.py),
// the `plan_out_scales` planner included. The audio tower is an embedding of dMel bins summed over the
// mel axis, followed by an RMSNorm. The vision tower is an HMLP encoder: fold time and space into depth,
// project, then RMSNorm and GELU on every layer except the last.
// The towers are built and their weights load. Neither is wired into the LM input path yet, so text-only
// generation does not use them (see `SharpAIInklingMmModel.imageFeatures` and `audioFeatures`).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// One HMLP encoder layer, planned from the config. `inputChannels` and `outputChannels` are the
/// projection's input and output widths.
public struct SharpAIInklingMmVisionStage: Equatable, Sendable {
    public let index: Int
    public let inputChannels: Int
    public let outputChannels: Int
    public let temporalFold: Int
    public let spatialFold: Int
    public let hasNorm: Bool
}

/// Plans the HMLP stages. The steps match transformers `plan_out_scales`.
public enum SharpAIInklingMmVisionPlanner {

    /// Prime factors of `n` in ascending order.
    static func primeFactors(_ value: Int) -> [Int] {
        var n = value
        var factors: [Int] = []
        while n % 2 == 0, n > 0 {
            factors.append(2)
            n /= 2
        }
        var p = 3
        while p * p <= n {
            while n % p == 0 {
                factors.append(p)
                n /= p
            }
            p += 2
        }
        if n > 1 {
            factors.append(n)
        }
        return factors
    }

    /// Cumulative product of the given values.
    static func cumulativeProducts(_ values: [Int]) -> [Int] {
        var running = 1
        return values.map { value in
            running *= value
            return running
        }
    }

    /// Scale rows `(t, h, w, c)`. Row 0 is the input, then one row per spatial scale, then one row per
    /// temporal scale, and the last row is the output. The count is `1 + spatial + temporal` rows.
    public static func scales(
        temporalPatchSize: Int, patchSize: Int, layers: Int, channels: Int
    ) -> [[Int]] {
        let h = cumulativeProducts(Array(primeFactors(patchSize).reversed()))
        let t = cumulativeProducts(Array(primeFactors(temporalPatchSize).reversed()))
        guard let hLast = h.last else { return [[1, 1, 1, channels]] }

        // torch: ceil(h^2 * C / 64) * 64 and ceil(h_last^2 * C * t) * 64, in integers.
        let spatialChannels = h.map { ((($0 * $0 * channels) + 63) / 64) * 64 }
        let temporalChannels = t.map { hLast * hLast * channels * $0 * 64 }

        var rows: [[Int]] = [[1, 1, 1, channels]]
        for i in 0 ..< h.count {
            rows.append([1, h[i], h[i], spatialChannels[i]])
        }
        for j in 0 ..< t.count {
            rows.append([t[j], hLast, hLast, temporalChannels[j]])
        }

        // Each row is assigned to one of `layers + 1` ideal log-scales, with the cost of |log distance|.
        let total = Double(patchSize * patchSize * temporalPatchSize * channels)
        let logTotal = log(total)
        let logIdeal = (0 ... layers).map { Double($0) * logTotal / Double(layers) }
        let logSizes = rows.map { row in log(Double(row[0] * row[1] * row[2])) }
        let cost = logIdeal.map { ideal in logSizes.map { abs(ideal - $0) } }

        var picks: [Int]
        if layers >= rows.count {
            picks = cost.map { row in
                (0 ..< row.count).min { row[$0] < row[$1] } ?? 0
            }
        } else {
            picks = bestAssignment(cost)
        }
        picks[0] = 0
        picks[picks.count - 1] = rows.count - 1
        return picks.map { rows[$0] }
    }

    /// Minimum-cost assignment of every row to a distinct column (exact search with pruning). The result
    /// matches scipy `linear_sum_assignment` apart from the choice among exact ties.
    static func bestAssignment(_ cost: [[Double]]) -> [Int] {
        let rowCount = cost.count
        let colCount = cost.first?.count ?? 0
        var best = [Int](repeating: 0, count: rowCount)
        var bestTotal = Double.infinity
        var current = [Int](repeating: 0, count: rowCount)
        var used = [Bool](repeating: false, count: colCount)

        func search(_ row: Int, _ total: Double) {
            if total >= bestTotal { return }
            if row == rowCount {
                bestTotal = total
                best = current
                return
            }
            for col in 0 ..< colCount where !used[col] {
                used[col] = true
                current[row] = col
                search(row + 1, total + cost[row][col])
                used[col] = false
            }
        }
        search(0, 0)
        return best
    }

    /// The HMLP stages for an encoder of `layers` layers. The last stage projects to `decoderDModel`.
    public static func stages(
        temporalPatchSize: Int, patchSize: Int, layers: Int, channels: Int, decoderDModel: Int
    ) -> [SharpAIInklingMmVisionStage] {
        let s = scales(
            temporalPatchSize: temporalPatchSize, patchSize: patchSize, layers: layers,
            channels: channels)
        return (0 ..< layers).map { (i: Int) -> SharpAIInklingMmVisionStage in
            let start = s[i]
            let end = s[i + 1]
            let temporalFold = end[0] / start[0]
            let spatialFold = end[1] / start[1]
            let shuffle = temporalFold * spatialFold * (end[2] / start[2])
            let isLast = i == layers - 1
            return SharpAIInklingMmVisionStage(
                index: i,
                inputChannels: start[3] * shuffle,
                outputChannels: isLast ? decoderDModel : end[3],
                temporalFold: temporalFold,
                spatialFold: spatialFold,
                hasNorm: !isLast)
        }
    }
}

/// Audio tower: the dMel bin embedding is summed over the mel axis, then normalized.
/// The checkpoint names are `model.audio.encoder` (`[mel_bins * mel_vocab, hidden]`) and `model.audio.final_norm`.
final class SharpAIInklingMmAudioTower: Module {
    @ModuleInfo(key: "encoder") var encoder: Embedding
    @ModuleInfo(key: "final_norm") var finalNorm: SharpAIInklingMmRMSNorm
    let offsets: MLXArray

    init(_ config: SharpAIInklingMmModelAudioConfiguration) {
        _encoder.wrappedValue = Embedding(
            embeddingCount: config.nMelBins * config.melVocabularySize,
            dimensions: config.decoderDModel)
        _finalNorm.wrappedValue = SharpAIInklingMmRMSNorm(dimensions: config.decoderDModel, eps: 1e-6)
        offsets = MLXArray((0 ..< config.nMelBins).map { Int32($0 * config.melVocabularySize) })
    }

    /// `ids` is `[num_audios, frames, mel_bins]` of dMel bin ids. Returns `[num_audios, frames, hidden]`.
    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        finalNorm(encoder(ids + offsets).sum(axis: -2))
    }
}

/// Vision tower. Each stage is a `linear_<i>` projection, plus a `norm_<i>` RMSNorm when the stage has one.
/// The checkpoint names are `model.visual.layers.linear_<i>`, `model.visual.layers.norm_<i>` and
/// `model.visual.final_norm`.
final class SharpAIInklingMmVisionTower: Module {
    @ModuleInfo(key: "layers") var layers: [String: Module]
    @ModuleInfo(key: "final_norm") var finalNorm: SharpAIInklingMmRMSNorm
    let stages: [SharpAIInklingMmVisionStage]

    init(_ config: SharpAIInklingMmModelVisionConfiguration, textHiddenSize: Int) {
        let planned = SharpAIInklingMmVisionPlanner.stages(
            temporalPatchSize: config.temporalPatchSize,
            patchSize: config.patchSize,
            layers: config.encoderLayers,
            channels: config.numChannels,
            decoderDModel: textHiddenSize)
        stages = planned

        var built: [String: Module] = [:]
        for stage in planned {
            built["linear_\(stage.index)"] = Linear(
                stage.inputChannels, stage.outputChannels, bias: false)
            if stage.hasNorm {
                built["norm_\(stage.index)"] = SharpAIInklingMmRMSNorm(
                    dimensions: stage.outputChannels, eps: 1e-6)
            }
        }
        _layers.wrappedValue = built
        _finalNorm.wrappedValue = SharpAIInklingMmRMSNorm(dimensions: textHiddenSize, eps: 1e-6)
    }

    /// `pixelValues` is `[num_patches, time, height, width, channels]`.
    /// Returns `[num_patches, 1, text_hidden]`.
    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var h = pixelValues
        for stage in stages {
            if stage.temporalFold > 1 || stage.spatialFold > 1 {
                h = foldTimeSpaceToDepth(
                    h, temporalFold: stage.temporalFold, spatialFold: stage.spatialFold)
            }
            guard let projection = layers["linear_\(stage.index)"] as? Linear else {
                preconditionFailure("inkling_mm_model vision projection \(stage.index) is missing")
            }
            h = projection(h)
            if stage.hasNorm {
                guard let norm = layers["norm_\(stage.index)"] as? SharpAIInklingMmRMSNorm else {
                    preconditionFailure("inkling_mm_model vision norm \(stage.index) is missing")
                }
                h = gelu(norm(h))
            }
        }
        h = finalNorm(h)
        return h.reshaped([pixelValues.dim(0), 1, -1])
    }

    /// `[B, T, H, W, C]` to `[B, T / t, H / s, W / s, C * t * s * s]`, as transformers
    /// `fold_timespace_to_depth`.
    private func foldTimeSpaceToDepth(
        _ h: MLXArray, temporalFold: Int, spatialFold: Int
    ) -> MLXArray {
        let batch = h.dim(0)
        let time = h.dim(1)
        let height = h.dim(2)
        let width = h.dim(3)
        let channels = h.dim(4)
        let tNew = time / temporalFold
        let hNew = height / spatialFold
        let wNew = width / spatialFold
        let grouped = h.reshaped([
            batch, tNew, temporalFold, hNew, spatialFold, wNew, spatialFold, channels,
        ])
        let permuted = grouped.transposed(0, 1, 3, 5, 2, 4, 6, 7)
        return permuted.reshaped([
            batch, tNew, hNew, wNew, temporalFold * spatialFold * spatialFold * channels,
        ])
    }
}
