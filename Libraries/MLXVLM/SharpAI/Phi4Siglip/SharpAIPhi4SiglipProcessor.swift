// SharpAI fork overlay: phi4-siglip NaFlex input processor. Not part of upstream mlx-swift-lm.
//
// Each image is resized to the largest SigLIP2 NaFlex grid whose patch count stays within
// `max_num_patches`, keeping the aspect ratio. It is normalized with mean and std 0.5 and cut
// into flat `patch * patch * C` rows. One row is one image token.
// UNVERIFIED: the NaFlex scale search bounds, the bicubic resampler (CoreImage, not PIL),
// `min_num_patches` (parsed, not enforced), and the `<image>` placeholder text.

import CoreImage
import Foundation
import MLX
import MLXLMCommon

public struct SharpAIPhi4SiglipProcessor: UserInputProcessor {

    static let imageMean: (CGFloat, CGFloat, CGFloat) = (0.5, 0.5, 0.5)
    static let imageStd: (CGFloat, CGFloat, CGFloat) = (0.5, 0.5, 0.5)
    // UNVERIFIED: "<image>" may not be one BPE token in the chat template context.
    static let imagePlaceholder = "<image>"

    private let configuration: SharpAIPhi4SiglipConfiguration
    private let tokenizer: any Tokenizer

    public init(_ configuration: SharpAIPhi4SiglipConfiguration, tokenizer: any Tokenizer) {
        self.configuration = configuration
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        var messages = chatMessages(from: input)

        guard !input.images.isEmpty else {
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools,
                additionalContext: input.additionalContext)
            return LMInput(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
        }

        // UNVERIFIED: the real checkpoint has no image_token_index, so image requests throw here.
        guard let imageTokenIndex = configuration.imageTokenIndex else {
            throw SharpAIPhi4SiglipError.imageTokenIndexMissing
        }

        var blocks: [MLXArray] = []
        var frames: [THW] = []
        for image in input.images {
            let (patches, grid) = try preprocess(
                image: try image.asCIImage(), processing: input.processing)
            blocks.append(patches)
            frames.append(THW(1, grid.height, grid.width))
        }

        guard let last = messages.lastIndex(where: { ($0["role"] as? String) == "user" }) else {
            throw ModelFactoryError.invalidConfiguration(
                "phi4-siglip: images need a user message to attach to")
        }
        let existing = (messages[last]["content"] as? String) ?? ""
        messages[last]["content"] =
            String(repeating: Self.imagePlaceholder, count: input.images.count) + existing

        let rawTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        let placeholder = tokenizer.encode(
            text: Self.imagePlaceholder, addSpecialTokens: false)
        let tokens = try Self.expandPlaceholders(
            rawTokens, placeholder: placeholder, imageToken: imageTokenIndex,
            counts: frames.map { $0.h * $0.w })

        return LMInput(
            text: .init(tokens: MLXArray(tokens).expandedDimensions(axis: 0)),
            image: .init(pixels: concatenated(blocks, axis: 0), frames: frames))
    }

    private func chatMessages(from input: UserInput) -> [[String: any Sendable]] {
        switch input.prompt {
        case .text(let text):
            return [["role": "user", "content": text]]
        case .messages(let messages):
            return messages
        case .chat(let messages):
            return messages.map { message -> [String: any Sendable] in
                ["role": message.role.rawValue, "content": message.content]
            }
        }
    }

    /// Returns the flat patch rows `[N, p * p * C]` and the patch grid for one image.
    private func preprocess(image: CIImage, processing: UserInput.Processing) throws -> (
        MLXArray, (height: Int, width: Int)
    ) {
        let patchSize = configuration.visionConfiguration.patchSize
        let base = MediaProcessing.apply(
            MediaProcessing.inSRGBToneCurveSpace(image), processing: processing)

        let size = Self.naflexSize(
            width: Int(base.extent.width.rounded()),
            height: Int(base.extent.height.rounded()),
            patchSize: patchSize, maxPatches: configuration.maxNumPatches)

        let resized = MediaProcessing.resampleBicubic(
            base, to: CGSize(width: CGFloat(size.width), height: CGFloat(size.height)))
        let normalized = MediaProcessing.normalize(
            resized, mean: Self.imageMean, std: Self.imageStd)
        let pixels = MediaProcessing.asMLXArray(normalized)[0]

        let grid = (height: size.height / patchSize, width: size.width / patchSize)
        return (Self.patchify(pixels, patchSize: patchSize), grid)
    }

    /// Pixel size of the largest NaFlex grid with at most `maxPatches` patches. Each side is
    /// rounded up to a patch multiple. The scale is found by bisection, as in SigLIP2 NaFlex.
    static func naflexSize(width: Int, height: Int, patchSize: Int, maxPatches: Int) -> (
        height: Int, width: Int
    ) {
        var low = 1e-4
        var high = 100.0
        while high - low >= 1e-5 {
            let mid = (low + high) / 2
            if patchCount(height: height, width: width, scale: mid, patchSize: patchSize)
                <= maxPatches
            {
                low = mid
            } else {
                high = mid
            }
        }
        return (
            scaledSide(height, scale: low, patchSize: patchSize),
            scaledSide(width, scale: low, patchSize: patchSize)
        )
    }

    static func patchCount(height: Int, width: Int, scale: Double, patchSize: Int) -> Int {
        (scaledSide(height, scale: scale, patchSize: patchSize) / patchSize)
            * (scaledSide(width, scale: scale, patchSize: patchSize) / patchSize)
    }

    static func scaledSide(_ value: Int, scale: Double, patchSize: Int) -> Int {
        let patches = (Double(value) * scale / Double(patchSize)).rounded(.up)
        return max(1, Int(patches)) * patchSize
    }

    /// `[C, H, W]` to `[rows * cols, p * p * C]`, with each row ordered `(row, col, channel)`.
    static func patchify(_ pixels: MLXArray, patchSize: Int) -> MLXArray {
        let (channels, height, width) = (pixels.dim(0), pixels.dim(1), pixels.dim(2))
        let rows = height / patchSize
        let cols = width / patchSize
        return pixels.reshaped(channels, rows, patchSize, cols, patchSize)
            .transposed(1, 3, 2, 4, 0)
            .reshaped(rows * cols, patchSize * patchSize * channels)
    }

    /// Replaces each `placeholder` run, in order, with `counts[i]` copies of `imageToken`.
    static func expandPlaceholders(
        _ tokens: [Int], placeholder: [Int], imageToken: Int, counts: [Int]
    ) throws -> [Int] {
        guard !placeholder.isEmpty else {
            throw SharpAIPhi4SiglipError.imagePlaceholderMismatch(
                expected: counts.count, actual: 0)
        }

        var output: [Int] = []
        var index = 0
        var image = 0
        while index < tokens.count {
            let end = index + placeholder.count
            if image < counts.count, end <= tokens.count,
                Array(tokens[index ..< end]) == placeholder
            {
                output += Array(repeating: imageToken, count: counts[image])
                image += 1
                index = end
            } else {
                output.append(tokens[index])
                index += 1
            }
        }

        guard image == counts.count else {
            throw SharpAIPhi4SiglipError.imagePlaceholderMismatch(
                expected: counts.count, actual: image)
        }
        return output
    }
}
