// SharpAI fork overlay: nemotron_parse input processor. Not part of upstream mlx-swift-lm.
//
// Port of NemotronParseImageProcessor and the text half of NemotronParseProcessor.
// Verified against: https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1/resolve/main/hf_nemotron_parse_processor.py
// and its preprocessor_config.json. The default task prompt is from the same repo's README.md.
//
// Image path, in the reference's order: convert to RGB, shrink to fit `final_size` keeping the aspect
// ratio (never enlarge, `LongestMaxSizeHW`, sizes truncated with int(), bilinear `cv2.INTER_LINEAR`),
// pad with white to `final_size` (albumentations `PadIfNeeded`, centred), scale to [0, 1], then
// normalise with `image_mean` / `image_std` when `do_normalize` is true. Pixels are NCHW.
//
// Text path: the prompt is tokenised as is, without special tokens (`add_special_tokens=False` in
// the README). A prompt that starts with `</s>` is a task prompt and is used verbatim. Anything else
// (a chat message, for example) is replaced by the default task prompt, because the model is not a chat
// model and only understands the task tokens.
//
// UNVERIFIED: the bilinear resample is written in MLX to match cv2's half-pixel-centre sampling
// and uint8 rounding (half up), but was not compared with cv2 output (no reference runs air-gapped). cv2 uses
// fixed-point weights, so a pixel can differ by one level. Albumentations' padding position is
// assumed to be its default, centre, for the version the checkpoint was made with.

import CoreImage
import Foundation
import MLX
import MLXLMCommon

/// `preprocessor_config.json` of nemotron_parse. Keys and fallbacks follow
/// `NemotronParseImageProcessor.__init__`.
public struct SharpAINemotronParseImageConfiguration: Decodable, Sendable, Equatable {
    public let finalHeight: Int
    public let finalWidth: Int
    public let doNormalize: Bool
    public let imageMean: [Float]
    public let imageStd: [Float]

    /// `OPENAI_CLIP_MEAN` and `OPENAI_CLIP_STD`, used when the file has none.
    static let clipMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    static let clipStd: [Float] = [0.26862954, 0.26130258, 0.27577711]

    enum CodingKeys: String, CodingKey {
        case finalSize = "final_size"
        case size
        case doNormalize = "do_normalize"
        case imageMean = "image_mean"
        case imageStd = "image_std"
    }

    /// `size` as the reference reads it: `longest_edge` (a pair, or one number for a square) wins,
    /// then `height` with `width`.
    private struct Size: Decodable {
        let height: Int?
        let width: Int?
        let longestEdge: SizeValue?

        enum CodingKeys: String, CodingKey {
            case height, width
            case longestEdge = "longest_edge"
        }
    }

    private enum SizeValue: Decodable {
        case one(Int)
        case pair(Int, Int)

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let value = try? container.decode(Int.self) {
                self = .one(value)
            } else {
                let pair = try container.decode([Int].self)
                guard pair.count >= 2 else {
                    throw DecodingError.dataCorruptedError(
                        in: container, debugDescription: "longest_edge needs two values")
                }
                self = .pair(pair[0], pair[1])
            }
        }
    }

    public init(
        finalHeight: Int = 2048, finalWidth: Int = 1648, doNormalize: Bool = false,
        imageMean: [Float]? = nil, imageStd: [Float]? = nil
    ) {
        self.finalHeight = finalHeight
        self.finalWidth = finalWidth
        self.doNormalize = doNormalize
        self.imageMean = imageMean ?? Self.clipMean
        self.imageStd = imageStd ?? Self.clipStd
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        var height = 2048
        var width = 1648
        // The reference accepts a pair or a single number, so a type mismatch is not an error here.
        if let pair = try? container.decodeIfPresent([Int].self, forKey: .finalSize), pair.count >= 2 {
            (height, width) = (pair[0], pair[1])
        } else if let single = try? container.decodeIfPresent(Int.self, forKey: .finalSize) {
            (height, width) = (single, single)
        }
        if let size = try container.decodeIfPresent(Size.self, forKey: .size) {
            if let edge = size.longestEdge {
                switch edge {
                case .one(let value): (height, width) = (value, value)
                case .pair(let h, let w): (height, width) = (h, w)
                }
            } else if let h = size.height, let w = size.width {
                (height, width) = (h, w)
            }
        }
        self.init(
            finalHeight: height, finalWidth: width,
            doNormalize: try container.decodeIfPresent(Bool.self, forKey: .doNormalize) ?? false,
            imageMean: try container.decodeIfPresent([Float].self, forKey: .imageMean),
            imageStd: try container.decodeIfPresent([Float].self, forKey: .imageStd))
    }
}

public enum SharpAINemotronParseProcessorError: LocalizedError {
    case needsExactlyOneImage(Int)
    case invalidImageSize(height: Int, width: Int)

    public var errorDescription: String? {
        switch self {
        case .needsExactlyOneImage(let count):
            return "nemotron_parse takes exactly one image, got \(count)."
        case .invalidImageSize(let height, let width):
            return "nemotron_parse cannot process an image of \(width)x\(height)."
        }
    }
}

public struct SharpAINemotronParseProcessor: UserInputProcessor {

    /// The `processor_class` the checkpoint's `preprocessor_config.json` names.
    public static let processorClass = "NemotronParseProcessor"

    /// README default: bounding boxes, classes and markdown text.
    public static let defaultTaskPrompt = "</s><s><predict_bbox><predict_classes><output_markdown>"

    private let configuration: SharpAINemotronParseImageConfiguration
    private let tokenizer: any Tokenizer

    public init(_ configuration: SharpAINemotronParseImageConfiguration, tokenizer: any Tokenizer) {
        self.configuration = configuration
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.images.count == 1, let image = input.images.first else {
            throw input.images.isEmpty
                ? SharpAINemotronParseError.missingImage
                : SharpAINemotronParseProcessorError.needsExactlyOneImage(input.images.count)
        }
        let ciImage = MediaProcessing.apply(
            MediaProcessing.inSRGBToneCurveSpace(try image.asCIImage()),
            processing: input.processing)
        let pixels = try preprocess(ciImage)

        let tokens = tokenizer.encode(text: taskPrompt(from: input), addSpecialTokens: false)
        return LMInput(
            text: .init(tokens: MLXArray(tokens).expandedDimensions(axis: 0)),
            image: .init(pixels: pixels))
    }

    /// A prompt that starts with `</s>` is a task prompt. Anything else becomes the default.
    func taskPrompt(from input: UserInput) -> String {
        let text: String
        switch input.prompt {
        case .text(let value):
            text = value
        case .messages(let messages):
            text = messages.last.flatMap { SharpAIEagle25VlProcessor.userText(from: $0["content"]) } ?? ""
        case .chat(let messages):
            text = messages.last?.content ?? ""
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.hasPrefix("</s>") ? trimmed : Self.defaultTaskPrompt
    }

    // MARK: - Image

    /// `[1, 3, finalHeight, finalWidth]` pixels.
    func preprocess(_ image: CIImage) throws -> MLXArray {
        // Whole-number levels, as the reference's uint8 array holds them.
        var levels = MLX.floor(MediaProcessing.asMLXArray(image) * 255 + 0.5).asType(.float32)
        let height = levels.dim(2)
        let width = levels.dim(3)
        guard height > 0, width > 0 else {
            throw SharpAINemotronParseProcessorError.invalidImageSize(height: height, width: width)
        }

        let target = Self.fittedSize(
            height: height, width: width,
            maxHeight: configuration.finalHeight, maxWidth: configuration.finalWidth)
        if target.height != height || target.width != width {
            levels = Self.bilinearResize(levels, height: target.height, width: target.width)
        }
        levels = Self.padWhite(
            levels, toHeight: configuration.finalHeight, width: configuration.finalWidth)

        var pixels = levels / 255
        if configuration.doNormalize {
            let mean = MLXArray(configuration.imageMean).reshaped(1, -1, 1, 1)
            let std = MLXArray(configuration.imageStd).reshaped(1, -1, 1, 1)
            pixels = (pixels - mean) / std
        }
        return pixels
    }

    /// `_resize_with_aspect_ratio`: only shrinks, truncates to Int like Python's `int()`.
    static func fittedSize(height: Int, width: Int, maxHeight: Int, maxWidth: Int) -> (
        height: Int, width: Int
    ) {
        let aspect = Double(width) / Double(height)
        var newHeight = height
        var newWidth = width
        if height > maxHeight {
            newHeight = maxHeight
            newWidth = Int(Double(newHeight) * aspect)
        }
        if newWidth > maxWidth {
            newWidth = maxWidth
            newHeight = Int(Double(newWidth) / aspect)
        }
        return (max(newHeight, 1), max(newWidth, 1))
    }

    /// `A.PadIfNeeded(border_mode=BORDER_CONSTANT, value=[255, 255, 255])`, centred: the top
    /// and left pads are `int(pad / 2)`, the rest goes to the bottom and right.
    static func padWhite(_ x: MLXArray, toHeight targetHeight: Int, width targetWidth: Int) -> MLXArray {
        var result = x
        let padHeight = max(0, targetHeight - x.dim(2))
        if padHeight > 0 {
            let top = padHeight / 2
            let bottom = padHeight - top
            let blockTop = MLXArray.full([1, 3, top, result.dim(3)], values: MLXArray(Float(255)))
            let blockBottom = MLXArray.full([1, 3, bottom, result.dim(3)], values: MLXArray(Float(255)))
            result = concatenated([blockTop, result, blockBottom], axis: 2)
        }
        let padWidth = max(0, targetWidth - x.dim(3))
        if padWidth > 0 {
            let left = padWidth / 2
            let right = padWidth - left
            let blockLeft = MLXArray.full([1, 3, result.dim(2), left], values: MLXArray(Float(255)))
            let blockRight = MLXArray.full([1, 3, result.dim(2), right], values: MLXArray(Float(255)))
            result = concatenated([blockLeft, result, blockRight], axis: 3)
        }
        return result
    }

    /// `cv2.resize(..., interpolation=cv2.INTER_LINEAR)` on `[1, C, H, W]` levels: half-pixel
    /// centres, a 2x2 neighbourhood, no antialiasing, edges clamped, result rounded to whole levels.
    static func bilinearResize(_ x: MLXArray, height newHeight: Int, width newWidth: Int) -> MLXArray {
        func axisTable(source: Int, target: Int) -> (low: [Int32], high: [Int32], weight: [Float]) {
            var low = [Int32](repeating: 0, count: target)
            var high = [Int32](repeating: 0, count: target)
            var weight = [Float](repeating: 0, count: target)
            let scale = Double(source) / Double(target)
            for index in 0 ..< target {
                let coordinate = (Double(index) + 0.5) * scale - 0.5
                var base = Int(coordinate.rounded(.down))
                var fraction = coordinate - Double(base)
                if base < 0 {
                    base = 0
                    fraction = 0
                }
                if base >= source - 1 {
                    base = source - 1
                    fraction = 0
                }
                low[index] = Int32(base)
                high[index] = Int32(min(base + 1, source - 1))
                weight[index] = Float(fraction)
            }
            return (low, high, weight)
        }

        let rows = axisTable(source: x.dim(2), target: newHeight)
        let columns = axisTable(source: x.dim(3), target: newWidth)

        let rowWeight = MLXArray(rows.weight).reshaped(1, 1, newHeight, 1)
        let vertical =
            take(x, MLXArray(rows.low), axis: 2) * (1 - rowWeight)
            + take(x, MLXArray(rows.high), axis: 2) * rowWeight

        let columnWeight = MLXArray(columns.weight).reshaped(1, 1, 1, newWidth)
        let both =
            take(vertical, MLXArray(columns.low), axis: 3) * (1 - columnWeight)
            + take(vertical, MLXArray(columns.high), axis: 3) * columnWeight

        // cv2 rounds half up in its fixed-point arithmetic, unlike MLX.round, which rounds half to even.
        return MLX.clip(MLX.floor(both + 0.5), min: 0, max: 255)
    }
}
