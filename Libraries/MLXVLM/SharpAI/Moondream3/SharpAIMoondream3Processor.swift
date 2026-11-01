// SharpAI fork overlay: moondream3 input processor. Not part of upstream mlx-swift-lm.
//
// Builds one global crop plus a grid of overlapping local crops (vLLM's preprocess_image).
// The prompt starts with `1 + enc_n_layers^2` image placeholder tokens, which `prepare` replaces.
// Then comes the task text: `<|md_reserved_0|>task<|md_reserved_1|>arg<|md_reserved_2|>`.
// UNVERIFIED: CoreImage bicubic resize instead of PIL LANCZOS, and float math instead of bf16.
// Only the last user message is used.

import CoreImage
import Foundation
import MLX
import MLXLMCommon

public struct SharpAIMoondream3Processor: UserInputProcessor {

    private let configuration: SharpAIMoondream3Configuration
    private let tokenizer: any Tokenizer

    public init(_ configuration: SharpAIMoondream3Configuration, tokenizer: any Tokenizer) {
        self.configuration = configuration
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.images.count <= 1 else {
            throw SharpAIMoondream3Error.singleImageOnly(input.images.count)
        }

        let task = SharpAIMoondream3Prompt.formatTask(lastUserText(from: input))
        let taskTokens = tokenizer.encode(text: task, addSpecialTokens: false)

        guard let image = input.images.first else {
            let tokens = [SharpAIMoondream3Configuration.bosTokenId] + taskTokens
            return LMInput(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
        }

        let placeholder = try imagePlaceholderToken()
        let crops = try preprocess(image.asCIImage(), processing: input.processing)
        let prefix = Array(repeating: placeholder, count: configuration.imagePrefixLength)

        return LMInput(
            text: .init(tokens: MLXArray(prefix + taskTokens).expandedDimensions(axis: 0)),
            image: .init(
                pixels: crops.pixels,
                frames: [THW(1, crops.rows, crops.cols)]))
    }

    private func lastUserText(from input: UserInput) -> String {
        switch input.prompt {
        case .text(let text):
            return text
        case .messages(let messages):
            let last = messages.last(where: { ($0["role"] as? String) == "user" })
            let content = last?["content"] as? String
            return content ?? ""
        case .chat(let messages):
            return messages.last(where: { $0.role.rawValue == "user" })?.content ?? ""
        }
    }

    /// Token that `<|endoftext|><image>` ends with, as vLLM's processor reads it.
    private func imagePlaceholderToken() throws -> Int {
        let ids = tokenizer.encode(text: "<|endoftext|><image>", addSpecialTokens: false)
        guard ids.count >= 2, let last = ids.last else {
            throw SharpAIMoondream3Error.imagePlaceholderUnavailable
        }
        return last
    }

    /// Returns `[1 + local crops, 3, S, S]` in `[-1, 1]`, and the local tile grid.
    private func preprocess(
        _ image: CIImage, processing: UserInput.Processing
    ) throws -> (pixels: MLXArray, rows: Int, cols: Int) {
        let vision = configuration.visionConfiguration
        let margin = vision.overlapMargin
        let totalMargin = 2 * margin * vision.patchSize
        let window = (vision.gridSize - 2 * margin) * vision.patchSize

        let base = MediaProcessing.apply(image, processing: processing)
        let width = Int(base.extent.width.rounded())
        let height = Int(base.extent.height.rounded())
        let tiling = SharpAIMoondream3Tiling.selectTiling(
            height: height - totalMargin, width: width - totalMargin,
            cropSize: window, maxCrops: vision.maxCrops)

        let crop = vision.cropSize
        let targetHeight = tiling.rows * window + totalMargin
        let targetWidth = tiling.cols * window + totalMargin

        let global = MediaProcessing.asMLXArray(
            MediaProcessing.resampleBicubic(base, to: CGSize(width: crop, height: crop)))
        let resized = MediaProcessing.asMLXArray(
            MediaProcessing.resampleBicubic(
                base, to: CGSize(width: targetWidth, height: targetHeight)))

        var crops: [MLXArray] = [global]
        for row in 0 ..< tiling.rows {
            for col in 0 ..< tiling.cols {
                let y0 = row * window
                let x0 = col * window
                crops.append(
                    resized[0..., 0..., y0 ..< (y0 + crop), x0 ..< (x0 + crop)])
            }
        }

        let pixels = MLX.concatenated(crops, axis: 0) * 2 - 1
        return (pixels, tiling.rows, tiling.cols)
    }
}
