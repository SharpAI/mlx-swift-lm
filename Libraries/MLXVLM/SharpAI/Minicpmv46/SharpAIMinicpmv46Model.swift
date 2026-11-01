// SharpAI fork overlay: MiniCPM-V 4.6 vision-language model. Not part of upstream mlx-swift-lm.
//
// Text runs through Qwen35Language.LanguageModel. Image and video features replace the
// placeholder tokens in the embedding sequence.
//
// Input convention: `LMInput.image` / `LMInput.video` pixels are `(slices, H, W, 3)`
// channels-last, with every slice on the same grid. The slice processor is not written.
// UNVERIFIED: M-RoPE positions are sequential for all tokens, as in the vLLM reference
// (`get_mrope_input_positions`). Confirm against a real prompt before trusting output.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public class SharpAIMinicpmv46: Module, VLMModel {
    @ModuleInfo(key: "vpm") var vpm: SharpAIMinicpmv46VisionTower
    @ModuleInfo(key: "vit_merger") var vitMerger: SharpAIMinicpmv46WindowMerger
    @ModuleInfo(key: "merger") var merger: SharpAIMinicpmv46Merger
    @ModuleInfo(key: "language_model") var languageModel: Qwen35Language.LanguageModel

    public let config: SharpAIMinicpmv46Configuration
    let useVitMerger: Bool

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }

    /// UNVERIFIED: returns the whole backbone. The Qwen35 `layers` property is fileprivate,
    /// so the per-layer LoRA targets cannot be reached from this file.
    public var loraLayers: [Module] {
        [languageModel.model]
    }

    public init(_ config: SharpAIMinicpmv46Configuration) throws {
        self.config = config
        self.useVitMerger = config.insertLayerId >= 0
        let vision = config.visionConfiguration
        let qwenConfiguration = try config.makeQwen35Configuration()
        self._vpm.wrappedValue = SharpAIMinicpmv46VisionTower(
            vision, dropLastLayer: config.dropVisionLastLayer)
        self._vitMerger.wrappedValue = SharpAIMinicpmv46WindowMerger(vision)
        self._merger.wrappedValue = SharpAIMinicpmv46Merger(
            hiddenSize: vision.hiddenSize, llmDim: config.textConfiguration.hiddenSize)
        self._languageModel.wrappedValue = Qwen35Language.LanguageModel(qwenConfiguration)
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.makeCache(parameters: parameters)
    }

    /// Encodes one slice `(Hpx, Wpx, 3)` into `(tokens, llmDim)`.
    func encodeSlice(_ slice: MLXArray) throws -> MLXArray {
        let patchSize = config.visionConfiguration.patchSize
        let pixelHeight = slice.dim(0)
        let pixelWidth = slice.dim(1)
        guard pixelHeight % patchSize == 0, pixelWidth % patchSize == 0 else {
            throw SharpAIMinicpmv46Error.invalidPixelShape([pixelHeight, pixelWidth, slice.dim(2)])
        }
        let height = pixelHeight / patchSize
        let width = pixelWidth / patchSize
        let mergeFactor = useVitMerger ? 4 : 2
        guard height % mergeFactor == 0, width % mergeFactor == 0 else {
            throw SharpAIMinicpmv46Error.invalidImageGrid(
                height: height, width: width, mergeFactor: mergeFactor)
        }

        let dtype = vpm.embeddings.patchEmbedding.weight.dtype
        let pixels = slice.expandedDimensions(axis: 0).asType(dtype)
        var hidden = vpm.embeddings(pixels, height: height, width: width)
        hidden = hidden.expandedDimensions(axis: 0)

        let layers = vpm.encoder.layers
        let splitIndex = min(max(config.insertLayerId + 1, 0), layers.count)
        for layer in layers[0 ..< splitIndex] {
            hidden = layer(hidden)
        }

        var gridHeight = height
        var gridWidth = width
        if useVitMerger {
            hidden = vitMerger(hidden[0], height: height, width: width).expandedDimensions(axis: 0)
            gridHeight = height / 2
            gridWidth = width / 2
        }

        for layer in layers[splitIndex ..< layers.count] {
            hidden = layer(hidden)
        }

        hidden = vpm.postLayernorm(hidden)
        return merger(hidden[0], height: gridHeight, width: gridWidth)
    }

    /// Encodes `(slices, Hpx, Wpx, 3)` and concatenates the token rows of every slice.
    func encodeSlices(_ pixels: MLXArray) throws -> MLXArray {
        guard pixels.ndim == 4, pixels.dim(-1) == config.visionConfiguration.numChannels else {
            throw SharpAIMinicpmv46Error.invalidPixelShape(pixels.shape)
        }
        var outputs = [MLXArray]()
        for index in 0 ..< pixels.dim(0) {
            outputs.append(try encodeSlice(pixels[index]))
        }
        return concatenated(outputs, axis: 0)
    }

    /// Replaces the `tokenId` placeholders in `embeddings` with features, in order.
    private func mergeFeatures(
        _ embeddings: MLXArray, tokens: MLXArray, pixels: MLXArray, tokenId: Int
    ) throws -> MLXArray {
        let features = try encodeSlices(pixels).asType(embeddings.dtype)
        let ids = tokens.flattened().asArray(Int.self)
        let positions = ids.enumerated().compactMap { $1 == tokenId ? $0 : nil }
        guard positions.count == features.dim(0) else {
            throw SharpAIMinicpmv46Error.placeholderMismatch(
                expected: positions.count, actual: features.dim(0))
        }
        if positions.isEmpty {
            return embeddings
        }

        var segments = [MLXArray]()
        var start = 0
        for (index, position) in positions.enumerated() {
            if position > start {
                segments.append(embeddings[0, start ..< position])
            }
            segments.append(features[index ..< index + 1])
            start = position + 1
        }
        if start < embeddings.dim(1) {
            segments.append(embeddings[0, start...])
        }
        return concatenated(segments, axis: 0).expandedDimensions(axis: 0)
    }

    private func anchorOffset(_ cache: [KVCache]?) -> Int {
        guard let cache, !cache.isEmpty else { return 0 }
        guard let index = languageModel.model.faIdx ?? languageModel.model.ssmIdx,
            index < cache.count
        else { return 0 }
        return cache[index].offset
    }

    /// Sequential positions, repeated across the three M-RoPE channels.
    private func positionIds(start: Int, length: Int) -> MLXArray {
        let row = MLXArray((start ..< start + length).map { Int32($0) })
            .reshaped(1, 1, length)
        return concatenated([row, row, row], axis: 0)
    }

    private func decode(
        tokens: MLXArray, embeddings: MLXArray, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        languageModel(
            tokens,
            inputsEmbeds: embeddings,
            cache: cache?.map { Optional($0) },
            state: state,
            positionIds: positionIds(start: anchorOffset(cache), length: embeddings.dim(1))
        )
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        var embeddings = languageModel.model.embedTokens(tokens)
        if let image = input.image {
            embeddings = try mergeFeatures(
                embeddings, tokens: tokens, pixels: image.pixels, tokenId: config.imageTokenId)
        }
        if let video = input.video {
            embeddings = try mergeFeatures(
                embeddings, tokens: tokens, pixels: video.pixels, tokenId: config.videoTokenId)
        }

        let totalTokens = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: totalTokens) { range in
            _ = decode(
                tokens: tokens[0..., range],
                embeddings: embeddings[0..., range],
                cache: cache, state: nil)
            eval(cache)
        }

        let result = decode(
            tokens: tokens[0..., processed...],
            embeddings: embeddings[0..., processed...],
            cache: cache, state: nil)
        prefill.progress?(totalTokens, totalTokens)
        return .logits(result)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        var tokens = input.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        let embeddings = languageModel.model.embedTokens(tokens)
        return decode(tokens: tokens, embeddings: embeddings, cache: cache, state: state)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(LMInput.Text(tokens: inputs), cache: cache, state: nil).logits
    }

    /// Maps HF checkpoint names to this module's names. Returns nil for dropped keys.
    static func remapKey(_ key: String) -> String? {
        if key.hasPrefix("mtp.") { return nil }
        if key.contains("rotary_emb.inv_freq") || key.contains("position_ids") { return nil }
        // More specific prefixes must come first.
        // UNVERIFIED: vit_merger and merger prefixes are guesses. No checkpoint is available.
        let rules: [(from: String, to: String)] = [
            ("model.vision_tower.vit_merger.", "vit_merger."),
            ("model.vision_tower.", "vpm."),
            ("model.vpm.", "vpm."),
            ("model.vit_merger.", "vit_merger."),
            ("model.merger.", "merger."),
            ("model.language_model.", "language_model.model."),
            ("lm_head.", "language_model.lm_head."),
        ]
        for rule in rules where key.hasPrefix(rule.from) {
            return rule.to + key.dropFirst(rule.from.count)
        }
        return key
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let untied = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: config.textConfiguration.tieWordEmbeddings)

        var mapped = [String: MLXArray]()
        for (key, value) in untied {
            guard let newKey = SharpAIMinicpmv46.remapKey(key) else { continue }
            mapped[newKey] = value
        }

        // Same conv1d and norm rules as the Qwen35 text path.
        let hasUnsanitizedConv1d = mapped.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let normSuffixes = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        var sanitized = [String: MLXArray]()
        sanitized.reserveCapacity(mapped.count)
        for (key, original) in mapped {
            var value = original
            if key.hasPrefix("language_model.") {
                if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                    value = value.movedAxis(source: 2, destination: 1)
                }
                if hasUnsanitizedConv1d, value.ndim == 1,
                    normSuffixes.contains(where: { key.hasSuffix($0) })
                {
                    value = value + MLXArray(1, dtype: value.dtype)
                }
            } else if key.hasPrefix("vpm.") && key.contains("patch_embedding.weight") {
                // HF stores (out, in, kH, kW). MLX conv wants (out, kH, kW, in).
                if value.dim(-1) != config.visionConfiguration.numChannels {
                    value = value.transposed(0, 2, 3, 1)
                }
            }
            sanitized[key] = value
        }
        return sanitized
    }
}
