// SharpAI fork overlay: qianfan_ocr input processor. Not part of upstream mlx-swift-lm.
//
// InternVL preprocessing: each image is tiled with the dynamic grid (SharpAIEagle25VlTilePlan,
// the same plan InternVL uses), a thumbnail tile is appended when there is more than one tile,
// and the text gets `<img>` + `<IMG_CONTEXT>` * tiles * tokensPerTile + `</img>` followed by a
// newline in front of the last user message. Several images are labelled `Image-N: `, as
// InternVL's multi-image chat does.
//
// Pixels come out as `[tiles, S, S, 3]` (channel last), the layout SharpAIQianfanOcr expects.
//
// UNVERIFIED, none of it checked against a real checkpoint or against the Python processor:
//  - ImageNet mean and std are assumed for this checkpoint (InternVL default).
//  - The tokenizer must map `<IMG_CONTEXT>` to `image_token_id` (151671 in the Hub config). A
//    mismatch surfaces as an image-token-count error from the model, not from here.
//  - CoreImage bicubic is not PIL bicubic, so pixel values differ slightly from the reference.
//  - The newline after `</img>` and the `Image-N:` labels follow InternVL's chat convention, not
//    Qianfan's own template.

import CoreImage
import Foundation
import MLX
import MLXLMCommon

public struct SharpAIQianfanOcrProcessor: UserInputProcessor {

    static let imageMean: (CGFloat, CGFloat, CGFloat) = (0.485, 0.456, 0.406)
    static let imageStd: (CGFloat, CGFloat, CGFloat) = (0.229, 0.224, 0.225)

    private let configuration: SharpAIQianfanOcrConfiguration
    private let tokenizer: any Tokenizer

    public init(_ configuration: SharpAIQianfanOcrConfiguration, tokenizer: any Tokenizer) {
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
        for (index, image) in input.images.enumerated() {
            let pixels = try preprocess(image: image.asCIImage(), processing: input.processing)
            pixelBlocks.append(pixels)
            if input.images.count > 1 {
                placeholder += "Image-\(index + 1): "
            }
            let contextCount = pixels.dim(0) * configuration.tokensPerTile
            placeholder +=
                "<img>" + String(repeating: "<IMG_CONTEXT>", count: contextCount) + "</img>\n"
        }

        guard let last = messages.lastIndex(where: { ($0["role"] as? String) == "user" }) else {
            throw ModelFactoryError.invalidConfiguration(
                "qianfan_ocr: images need a user message to attach to")
        }
        let existing = SharpAIEagle25VlProcessor.userText(from: messages[last]["content"])
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

    /// Returns `[T, S, S, 3]` pixels for one image: grid tiles in row-major order, then the
    /// thumbnail when the plan has one.
    private func preprocess(image: CIImage, processing: UserInput.Processing) throws -> MLXArray {
        let size = configuration.visionConfiguration.imageSize
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

        // `asMLXArray` gives channel-first `[1, 3, H, W]`. The model takes `[T, H, W, 3]`.
        return concatenated(tiles, axis: 0).transposed(0, 2, 3, 1)
    }
}
