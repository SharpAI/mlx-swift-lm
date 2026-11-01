// SharpAI fork overlay: eagle_2_5_vl input processor. Not part of upstream mlx-swift-lm.
//
// Tiles each image with the InternVL grid (SharpAIEagle25VlTilePlan), appends the thumbnail,
// and writes `<img>` + `<IMG_CONTEXT>` * tiles * seq + `</img>` before the last user turn.
// Port of vLLM's InternVLChatProcessor.get_image_repl.
// UNVERIFIED: whether the tokenizer maps `<IMG_CONTEXT>` to img_context_token_id, and the
// resampler (CoreImage bicubic) differs slightly from PIL bicubic.

import CoreImage
import Foundation
import MLX
import MLXLMCommon

public struct SharpAIEagle25VlProcessor: UserInputProcessor {

    static let imageMean: (CGFloat, CGFloat, CGFloat) = (0.485, 0.456, 0.406)
    static let imageStd: (CGFloat, CGFloat, CGFloat) = (0.229, 0.224, 0.225)

    private let configuration: SharpAIEagle25VlConfiguration
    private let tokenizer: any Tokenizer

    public init(_ configuration: SharpAIEagle25VlConfiguration, tokenizer: any Tokenizer) {
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

        var pixelBlocks: [MLXArray] = []
        var placeholder = ""
        for image in input.images {
            let pixels = try preprocess(
                image: image.asCIImage(), processing: input.processing)
            pixelBlocks.append(pixels)
            let contextCount = pixels.dim(0) * configuration.imageSeqLength
            placeholder +=
                "<img>" + String(repeating: "<IMG_CONTEXT>", count: contextCount) + "</img>"
        }

        guard let last = messages.lastIndex(where: { ($0["role"] as? String) == "user" }) else {
            throw ModelFactoryError.invalidConfiguration(
                "eagle_2_5_vl: images need a user message to attach to")
        }
        let existing = (messages[last]["content"] as? String) ?? ""
        messages[last]["content"] = placeholder + existing

        let tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        return LMInput(
            text: .init(tokens: MLXArray(tokens).expandedDimensions(axis: 0)),
            image: .init(pixels: concatenated(pixelBlocks, axis: 0)))
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

    /// Returns `[T, C, S, S]` pixels for one image: grid tiles in row-major order, then the thumbnail.
    private func preprocess(image: CIImage, processing: UserInput.Processing) throws -> MLXArray {
        let size = configuration.effectiveImageSize
        let plan = SharpAIEagle25VlTilePlan.plan(
            width: Int(image.extent.width.rounded()),
            height: Int(image.extent.height.rounded()),
            imageSize: size,
            minDynamicPatch: configuration.minDynamicPatch,
            maxDynamicPatch: configuration.maxDynamicPatch,
            dynamicImageSize: configuration.dynamicImageSize,
            useThumbnail: configuration.useThumbnail)

        let base = MediaProcessing.apply(
            MediaProcessing.inSRGBToneCurveSpace(image), processing: processing)

        let resized = MediaProcessing.resampleBicubic(
            base, to: CGSize(width: plan.targetWidth, height: plan.targetHeight))
        let grid = MediaProcessing.asMLXArray(
            MediaProcessing.normalize(resized, mean: Self.imageMean, std: Self.imageStd))

        var tiles: [MLXArray] = []
        tiles.reserveCapacity(plan.tileCount)
        for row in 0 ..< plan.rows {
            for column in 0 ..< plan.columns {
                tiles.append(
                    grid[
                        0..., 0...,
                        (row * size) ..< ((row + 1) * size),
                        (column * size) ..< ((column + 1) * size)])
            }
        }

        if plan.hasThumbnail {
            let thumbnail = MediaProcessing.resampleBicubic(
                base, to: CGSize(width: size, height: size))
            tiles.append(
                MediaProcessing.asMLXArray(
                    MediaProcessing.normalize(thumbnail, mean: Self.imageMean, std: Self.imageStd)))
        }

        return concatenated(tiles, axis: 0)
    }
}
