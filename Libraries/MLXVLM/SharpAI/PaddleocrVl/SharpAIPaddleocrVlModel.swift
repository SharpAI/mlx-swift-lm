// SharpAI fork overlay: PaddleOCR-VL vision-language model. Not part of upstream mlx-swift-lm.
//
// Input: `LMInput.image` pixels are (patches, P, P, 3), channels last, with `frames` giving a
// (1, h, w) patch grid per image. No image processor is written, so no UserInputProcessor is
// registered.
// M-RoPE: text gets sequential positions. Image tokens get t/h/w positions on the merged grid.
// Decoding resumes at `cacheOffset + ropeDelta`, and the delta is carried in LMOutput.State.
// UNVERIFIED: the placeholder token id and the image position rule come from memory of the
// reference model. Check them on a real prompt before trusting output.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Carries the M-RoPE delta (next position minus cache offset) between calls.
let sharpAIPaddleocrVlRopeDeltaKey = LMOutput.Key<Int>("paddleocrvl.ropeDelta")

public class SharpAIPaddleocrVl: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") var visionTower: SharpAIPaddleocrVlVisionTower
    @ModuleInfo var projector: SharpAIPaddleocrVlProjector
    @ModuleInfo(key: "language_model") var languageModel: SharpAIPaddleocrVlLanguageModel

    public let config: SharpAIPaddleocrVlConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: SharpAIPaddleocrVlConfiguration) {
        self.config = config
        self._visionTower.wrappedValue = SharpAIPaddleocrVlVisionTower(config.visionConfiguration)
        self._projector.wrappedValue = SharpAIPaddleocrVlProjector(
            config.visionConfiguration, textHiddenSize: config.textConfiguration.hiddenSize)
        self._languageModel.wrappedValue = SharpAIPaddleocrVlLanguageModel(
            config.textConfiguration)
        super.init()
    }

    /// Absolute M-RoPE positions for one row of tokens, as (3, 1, count), and the next free
    /// position. Text advances by one. Each image block takes its t/h/w offsets from one base.
    static func mropePositions(
        tokenIds: [Int], frames: [THW], imageTokenId: Int, mergeSize: Int, anchor: Int
    ) throws -> (positions: MLXArray, next: Int) {
        var expected = 0
        for frame in frames {
            guard frame.t == 1 else {
                throw SharpAIPaddleocrVlError.unsupportedTemporalFrames(frame.t)
            }
            guard frame.h % mergeSize == 0, frame.w % mergeSize == 0 else {
                throw SharpAIPaddleocrVlError.invalidImageGrid(
                    height: frame.h, width: frame.w, mergeSize: mergeSize)
            }
            expected += (frame.h / mergeSize) * (frame.w / mergeSize)
        }
        let actual = tokenIds.filter { $0 == imageTokenId }.count
        guard actual == expected else {
            throw SharpAIPaddleocrVlError.placeholderMismatch(expected: expected, actual: actual)
        }

        var axes: [[Int]] = [[], [], []]
        var next = anchor

        func appendText(_ count: Int) {
            for offset in 0 ..< count {
                for axis in 0 ..< 3 {
                    axes[axis].append(next + offset)
                }
            }
            next += count
        }

        var cursor = 0
        for frame in frames {
            guard let start = tokenIds[cursor...].firstIndex(of: imageTokenId) else {
                throw SharpAIPaddleocrVlError.placeholderMismatch(expected: expected, actual: actual)
            }
            appendText(start - cursor)

            let height = frame.h / mergeSize
            let width = frame.w / mergeSize
            let base = next
            for row in 0 ..< height {
                for column in 0 ..< width {
                    axes[0].append(base)
                    axes[1].append(base + row)
                    axes[2].append(base + column)
                }
            }
            next = base + max(height, width)
            cursor = start + height * width
        }
        appendText(tokenIds.count - cursor)

        let flat = (axes[0] + axes[1] + axes[2]).map { Int32($0) }
        let positions = MLXArray(flat).reshaped(3, 1, tokenIds.count)
        return (positions, next)
    }

    /// Sequential positions for a text run, repeated on all three axes: (3, 1, length).
    static func textPositions(start: Int, length: Int) -> MLXArray {
        let row = MLXArray((start ..< start + length).map { Int32($0) })
        return concatenated([row, row, row], axis: 0).reshaped(3, 1, length)
    }

    /// Vision features for every image, concatenated: (image tokens, textHidden).
    func imageFeatures(pixels: MLXArray, frames: [THW]) throws -> MLXArray {
        let patchSize = config.visionConfiguration.patchSize
        guard pixels.ndim == 4, pixels.dim(1) == patchSize, pixels.dim(2) == patchSize,
            pixels.dim(-1) == config.visionConfiguration.numChannels
        else {
            throw SharpAIPaddleocrVlError.invalidPixelShape(pixels.shape)
        }
        let patchTotal = frames.reduce(0) { $0 + $1.h * $1.w }
        guard pixels.dim(0) == patchTotal else {
            throw SharpAIPaddleocrVlError.invalidPixelShape(pixels.shape)
        }

        var outputs = [MLXArray]()
        var offset = 0
        for frame in frames {
            guard frame.t == 1 else {
                throw SharpAIPaddleocrVlError.unsupportedTemporalFrames(frame.t)
            }
            let count = frame.h * frame.w
            let patches = pixels[offset ..< offset + count]
            offset += count
            let hidden = visionTower(patches, height: frame.h, width: frame.w)
            outputs.append(projector(hidden, height: frame.h, width: frame.w))
        }
        return concatenated(outputs, axis: 0)
    }

    /// Writes the image features over the image placeholder positions.
    private func mergeImageFeatures(
        _ embeddings: MLXArray, tokenIds: [Int], features: MLXArray
    ) -> MLXArray {
        let indices = tokenIds.enumerated().compactMap { index, id in
            id == config.imageTokenId ? index : nil
        }
        guard !indices.isEmpty else { return embeddings }

        var result = embeddings
        result[0..., MLXArray(indices), 0...] = features.expandedDimensions(axis: 0)
            .asType(embeddings.dtype)
        return result
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        guard tokens.dim(0) == 1 else {
            throw SharpAIPaddleocrVlError.batchUnsupported(tokens.dim(0))
        }

        let cacheOffset = cache.first?.offset ?? 0
        let anchor = cacheOffset + (state?[sharpAIPaddleocrVlRopeDeltaKey] ?? 0)
        let tokenIds = tokens.flattened().asArray(Int.self)
        let frames = input.image?.frames ?? []

        let (positions, next) = try SharpAIPaddleocrVl.mropePositions(
            tokenIds: tokenIds, frames: frames, imageTokenId: config.imageTokenId,
            mergeSize: config.visionConfiguration.spatialMergeSize, anchor: anchor)

        var embeddings = languageModel.model.embedTokens(tokens)
        if let image = input.image, !frames.isEmpty {
            let features = try imageFeatures(pixels: image.pixels, frames: frames)
                .asType(embeddings.dtype)
            embeddings = mergeImageFeatures(embeddings, tokenIds: tokenIds, features: features)
        }

        let total = tokenIds.count
        let processed = try prefill.forEachChunk(total: total) { range in
            _ = languageModel.forward(
                embeddings[0..., range],
                positionIds: positions[0..., 0..., range],
                cache: cache)
            eval(cache)
        }

        let logits = languageModel.forward(
            embeddings[0..., processed...],
            positionIds: positions[0..., 0..., processed...],
            cache: cache)
        prefill.progress?(total, total)

        var outState = LMOutput.State()
        outState[sharpAIPaddleocrVlRopeDeltaKey] = next - (cacheOffset + total)
        return .logits(LMOutput(logits: logits, state: outState))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        var tokens = input.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }

        let delta = state?[sharpAIPaddleocrVlRopeDeltaKey] ?? 0
        let offset = cache?.first?.offset ?? 0
        let positions = SharpAIPaddleocrVl.textPositions(
            start: offset + delta, length: tokens.dim(1))
        let logits = languageModel.forward(
            languageModel.model.embedTokens(tokens), positionIds: positions, cache: cache)

        var outState = LMOutput.State()
        outState[sharpAIPaddleocrVlRopeDeltaKey] = delta
        return LMOutput(logits: logits, state: outState)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(LMInput.Text(tokens: inputs), cache: cache, state: nil).logits
    }

    /// Maps HF checkpoint names to this module's names. Returns nil for dropped keys.
    static func remapKey(_ key: String, textLayerCount: Int) -> String? {
        if key.hasPrefix("mtp.") { return nil }
        if key.contains("rotary_emb.inv_freq") || key.hasSuffix("position_ids") { return nil }
        // Unused by this tower: the packed position table and any SigLIP pooling head.
        if key.contains("packing_position_embedding") || key.contains("vision_model.head") {
            return nil
        }

        // More specific prefixes must come first.
        let rules: [(from: String, to: String)] = [
            ("visual.vision_model.", "vision_tower."),
            ("vision_model.", "vision_tower."),
            ("visual.", "vision_tower."),
            ("mlp_AR.", "projector."),
            ("model.language_model.", "language_model.model."),
            ("model.visual.", "vision_tower."),
            ("model.", "language_model.model."),
            ("lm_head.", "language_model.lm_head."),
        ]
        var mapped = key
        for rule in rules where key.hasPrefix(rule.from) {
            mapped = rule.to + key.dropFirst(rule.from.count)
            break
        }

        // Drop next-token-prediction layers past the last decoder layer.
        let layerPrefix = "language_model.model.layers."
        if mapped.hasPrefix(layerPrefix) {
            let rest = mapped.dropFirst(layerPrefix.count)
            if let index = Int(rest.prefix(while: { $0 != "." })), index >= textLayerCount {
                return nil
            }
        }
        return mapped
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let untied = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: config.textConfiguration.tieWordEmbeddings)

        var mapped = [String: MLXArray]()
        for (key, value) in untied {
            guard
                let newKey = SharpAIPaddleocrVl.remapKey(
                    key, textLayerCount: config.textConfiguration.hiddenLayers)
            else { continue }
            mapped[newKey] = value
        }

        // HF stores conv weights as (out, in, kH, kW). MLX wants (out, kH, kW, in).
        let channels = config.visionConfiguration.numChannels
        let patchKey = "vision_tower.embeddings.patch_embedding.weight"
        var sanitized = [String: MLXArray]()
        sanitized.reserveCapacity(mapped.count)
        for (key, value) in mapped {
            if key == patchKey, value.ndim == 4, value.dim(-1) != channels {
                sanitized[key] = value.transposed(0, 2, 3, 1)
            } else {
                sanitized[key] = value
            }
        }
        return sanitized
    }
}
