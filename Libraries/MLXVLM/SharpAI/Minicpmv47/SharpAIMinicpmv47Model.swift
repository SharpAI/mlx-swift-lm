// SharpAI fork overlay: MiniCPM-V 4.7 vision-language model. Not part of upstream mlx-swift-lm.
//
// Reference: transformers modeling_minicpmv4_7.py (MiniCPMV4_7Model, MiniCPMV4_7ForConditionalGeneration)
// and the openbmb/MiniCPM-V-4.7-1B config.json.
//
// Text runs through Qwen35Language.LanguageModel (qwen3_5_text, hybrid linear attention). Image and
// video features replace the placeholder tokens in the embedding sequence. Prompt positions are canvas
// M-RoPE when the prompt has image or video crops, and sequential otherwise.
//
// Input convention: `LMInput.image` / `LMInput.video` pixels are `(slices, H, W, 3)` channels-last with
// every slice on one grid. The first slice of a frame is its thumbnail and the rest are detail slices,
// in prompt order. The slice processor is not written here, so the caller builds the crops and markers.
// UNVERIFIED: logits have not been compared with the HF reference. The canvas port is covered only by
// hand-computed position tests.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public class SharpAIMinicpmv47: Module, VLMModel {
    @ModuleInfo(key: "vpm") var vpm: SharpAIMinicpmv47VisionTower
    @ModuleInfo(key: "vit_merger") var vitMerger: SharpAIMinicpmv47WindowMerger
    @ModuleInfo(key: "merger") var merger: SharpAIMinicpmv47Merger
    @ModuleInfo(key: "language_model") var languageModel: Qwen35Language.LanguageModel

    public let config: SharpAIMinicpmv47Configuration
    let textConfig: Qwen35Configuration.TextConfiguration
    let visionConfig: SharpAIMinicpmv47VisionConfiguration
    let useVitMerger: Bool
    let canvas: SharpAIMinicpmv47CanvasPositions

    /// Carries the canvas delta (HF `rope_deltas`) between steps. It is added to the sequential
    /// positions of tokens after a canvas prefill, and lives in the request state, not on the model.
    static let ropeDeltaKey = LMOutput.Key<MLXArray>("minicpmv47.ropeDelta")

    public var vocabularySize: Int { textConfig.vocabularySize }

    /// UNVERIFIED: returns the whole backbone. The Qwen35 `layers` property is fileprivate,
    /// so the per-layer LoRA targets cannot be reached from this file.
    public var loraLayers: [Module] {
        [languageModel.model]
    }

    public init(_ config: SharpAIMinicpmv47Configuration) throws {
        try config.validateModelConfiguration()
        guard let text = config.textConfiguration, let vision = config.visionConfiguration else {
            throw ModelFactoryError.invalidConfiguration(
                "minicpmv4_7: text_config and vision_config are required")
        }
        let windowMerge = config.downsampleMode == "16x"
        self.config = config
        self.textConfig = text
        self.visionConfig = vision
        self.useVitMerger = windowMerge
        self.canvas = SharpAIMinicpmv47CanvasPositions(
            imageStartId: config.imageStartId,
            imageEndId: config.imageEndId,
            sliceStartId: config.sliceStartId,
            sliceEndId: config.sliceEndId,
            newlineId: config.newlineId,
            mergeFactor: SharpAIMinicpmv47.totalMergeFactor(
                windowMerge: windowMerge, mergerTimes: config.mergerTimes))

        let qwenConfiguration = try SharpAIMinicpmv47.makeQwen35Configuration(
            text: text, vision: vision)
        self._vpm.wrappedValue = SharpAIMinicpmv47VisionTower(vision)
        self._vitMerger.wrappedValue = SharpAIMinicpmv47WindowMerger(vision)
        self._merger.wrappedValue = SharpAIMinicpmv47Merger(
            hiddenSize: vision.hiddenSize, llmDim: text.hiddenSize, times: config.mergerTimes)
        self._languageModel.wrappedValue = Qwen35Language.LanguageModel(qwenConfiguration)
        super.init()
    }

    /// Patches per LLM token along one axis: 2 for the window merger (16x mode) times 2 for every
    /// MLP merger round.
    static func totalMergeFactor(windowMerge: Bool, mergerTimes: Int) -> Int {
        (windowMerge ? 2 : 1) * (1 << max(mergerTimes, 1))
    }

    /// Builds the Qwen35 configuration that the text backbone needs.
    ///
    /// Only `text_config` is used. The `vision_config` block is a placeholder that satisfies the
    /// Qwen35 decoder, and no Qwen3-VL tower is built from it.
    static func makeQwen35Configuration(
        text: Qwen35Configuration.TextConfiguration, vision: SharpAIMinicpmv47VisionConfiguration
    ) throws -> Qwen35Configuration {
        let textData = try JSONEncoder().encode(text)
        let textObject: Any = try JSONSerialization.jsonObject(with: textData)
        let root: [String: Any] = [
            "model_type": "qwen3_5",
            "text_config": textObject,
            "vision_config": [
                "model_type": "qwen3_vl",
                "depth": vision.numHiddenLayers,
                "hidden_size": vision.hiddenSize,
                "intermediate_size": vision.intermediateSize,
                "out_hidden_size": text.hiddenSize,
                "num_heads": vision.numAttentionHeads,
                "patch_size": vision.patchSize,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "num_position_embeddings": 4096,
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: root)
        return try JSONDecoder().decode(Qwen35Configuration.self, from: data)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.makeCache(parameters: parameters)
    }

    /// Encodes one slice `(H, W, 3)` into `(tokens, llmDim)`.
    func encodeSlice(_ slice: MLXArray) throws -> MLXArray {
        let patchSize = visionConfig.patchSize
        let pixelHeight = slice.dim(0)
        let pixelWidth = slice.dim(1)
        guard slice.ndim == 3, slice.dim(2) == visionConfig.numChannels,
            pixelHeight % patchSize == 0, pixelWidth % patchSize == 0
        else {
            throw SharpAIMinicpmv47Error.invalidPixelShape(slice.shape)
        }
        let height = pixelHeight / patchSize
        let width = pixelWidth / patchSize
        // 16x runs the 2x2 window merger first, then every MLP round merges 2x2 more.
        let mergeFactor = Self.totalMergeFactor(
            windowMerge: useVitMerger, mergerTimes: config.mergerTimes)
        guard height % mergeFactor == 0, width % mergeFactor == 0 else {
            throw SharpAIMinicpmv47Error.invalidImageGrid(
                height: height, width: width, mergeFactor: mergeFactor)
        }

        let dtype = vpm.embeddings.patchEmbedding.weight.dtype
        let pixels = slice.expandedDimensions(axis: 0).asType(dtype)
        var hidden = vpm.embeddings(pixels, height: height, width: width)
            .expandedDimensions(axis: 0)

        let layers = vpm.encoder.layers
        var gridHeight = height
        var gridWidth = width
        if useVitMerger {
            // Layers 0...insert_layer_id run on the full grid, then the window merger halves it.
            let split = min(max(config.insertLayerId + 1, 0), layers.count)
            for layer in layers[0 ..< split] {
                hidden = layer(hidden)
            }
            hidden = vitMerger(hidden[0], height: height, width: width).expandedDimensions(axis: 0)
            gridHeight = height / 2
            gridWidth = width / 2
            for layer in layers[split...] {
                hidden = layer(hidden)
            }
        } else {
            for layer in layers {
                hidden = layer(hidden)
            }
        }

        hidden = vpm.postLayernorm(hidden)
        return try merger(hidden[0], height: gridHeight, width: gridWidth)
    }

    /// Encodes `(slices, Hpx, Wpx, 3)` and concatenates the token rows of every slice.
    func encodeSlices(_ pixels: MLXArray) throws -> MLXArray {
        guard pixels.ndim == 4, pixels.dim(-1) == visionConfig.numChannels else {
            throw SharpAIMinicpmv47Error.invalidPixelShape(pixels.shape)
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
            throw SharpAIMinicpmv47Error.placeholderMismatch(
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

    /// Patch grids `(h, w)` of every crop in `pixels`, one per slice.
    private func crops(_ pixels: MLXArray?) -> [(Int, Int)] {
        guard let pixels = pixels else { return [] }
        let patchSize = visionConfig.patchSize
        let grid = (pixels.dim(1) / patchSize, pixels.dim(2) / patchSize)
        return Array(repeating: grid, count: pixels.dim(0))
    }

    private func anchorOffset(_ cache: [KVCache]?) -> Int {
        guard let cache, !cache.isEmpty else { return 0 }
        guard let index = languageModel.model.faIdx ?? languageModel.model.ssmIdx,
            index < cache.count
        else { return 0 }
        return cache[index].offset
    }

    /// Reads the canvas delta saved in `state`, or 0 when there is none.
    static func ropeDelta(_ state: LMOutput.State?) -> Int {
        guard let array = state?[ropeDeltaKey] else { return 0 }
        return array.item(Int.self)
    }

    /// Sequential positions `start ..< start + length`, shifted by `delta`, repeated over the
    /// three M-RoPE channels.
    func linearPositions(start: Int, length: Int, delta: Int) -> MLXArray {
        let values = (0 ..< length).map { Int32(start + $0 + delta) }
        let row = MLXArray(values).reshaped(1, 1, length)
        return concatenated([row, row, row], axis: 0)
    }

    /// Position ids `(3, 1, count)` for prompt tokens `range`. Canvas ids are used when the prompt
    /// has crops, otherwise sequential positions offset by the cache.
    private func promptPositions(
        _ range: Range<Int>, base: Int, canvasIds: MLXArray?, delta: Int
    ) -> MLXArray {
        if let canvasIds {
            return canvasIds[0..., 0..., range]
        }
        return linearPositions(start: base + range.lowerBound, length: range.count, delta: delta)
    }

    private func decode(
        tokens: MLXArray, embeddings: MLXArray, cache: [KVCache]?, state: LMOutput.State?,
        positionIds: MLXArray
    ) -> LMOutput {
        languageModel(
            tokens,
            inputsEmbeds: embeddings,
            cache: cache?.map { Optional($0) },
            state: state,
            positionIds: positionIds
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

        let base = anchorOffset(cache)
        var canvasIds: MLXArray?
        // HF computes canvas positions only on a fresh cache, and keeps the delta for later chunks.
        // A warm cache resumes with the delta saved in `state`, never one left on the model.
        var ropeDelta = base == 0 ? 0 : Self.ropeDelta(state)
        if base == 0 {
            if input.image != nil || input.video != nil {
                let ids = tokens.flattened().asArray(Int.self)
                let modalities = ids.map { id -> Int in
                    if id == config.imageTokenId { return 1 }
                    if id == config.videoTokenId { return 2 }
                    return 0
                }
                let result = try canvas.compute(
                    tokenIds: ids,
                    modalities: modalities,
                    imageGrids: crops(input.image?.pixels),
                    videoGrids: crops(input.video?.pixels))
                ropeDelta = result.delta
                let flat = result.positions.flatMap { $0.map { Int32($0) } }
                canvasIds = MLXArray(flat).reshaped(3, 1, ids.count)
            }
        }

        let totalTokens = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: totalTokens) { range in
            _ = decode(
                tokens: tokens[0..., range],
                embeddings: embeddings[0..., range],
                cache: cache,
                state: nil,
                positionIds: promptPositions(
                    range.lowerBound ..< range.upperBound, base: base, canvasIds: canvasIds,
                    delta: ropeDelta))
            eval(cache)
        }

        let result = decode(
            tokens: tokens[0..., processed...],
            embeddings: embeddings[0..., processed...],
            cache: cache,
            state: nil,
            positionIds: promptPositions(
                processed ..< totalTokens, base: base, canvasIds: canvasIds, delta: ropeDelta))
        prefill.progress?(totalTokens, totalTokens)
        var nextState = result.state ?? state ?? LMOutput.State()
        nextState[Self.ropeDeltaKey] = MLXArray(Int32(ropeDelta))
        return .logits(LMOutput(logits: result.logits, state: nextState))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        var tokens = input.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        let embeddings = languageModel.model.embedTokens(tokens)
        let base = anchorOffset(cache)
        return decode(
            tokens: tokens,
            embeddings: embeddings,
            cache: cache,
            state: state,
            positionIds: linearPositions(
                start: base, length: embeddings.dim(1), delta: Self.ropeDelta(state)))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(LMInput.Text(tokens: inputs), cache: cache, state: nil).logits
    }

    /// Maps HF checkpoint names to this module's names. Returns nil for dropped keys.
    static func remapKey(_ key: String) -> String? {
        if key.hasPrefix("mtp.") { return nil }
        if key.contains("rotary_emb.inv_freq") || key.contains("position_ids") { return nil }
        // More specific prefixes must come first. The window merger sits under vision_tower in HF.
        let rules: [(from: String, to: String)] = [
            ("model.vision_tower.vit_merger.", "vit_merger."),
            ("model.vision_tower.", "vpm."),
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
            from: weights, tiedWordEmbeddings: textConfig.tieWordEmbeddings)

        var mapped = [String: MLXArray]()
        for (key, value) in untied {
            guard let newKey = SharpAIMinicpmv47.remapKey(key) else { continue }
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
                if value.dim(-1) != visionConfig.numChannels {
                    value = value.transposed(0, 2, 3, 1)
                }
            }
            sanitized[key] = value
        }
        return sanitized
    }
}
