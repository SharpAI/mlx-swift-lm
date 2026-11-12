// SharpAI SL-51 PR3: eagle_2_5_vl, moondream3 and phi4-siglip get a working processor hook.
//
// Their processors are built from the model's own `config.json` (not from `preprocessor_config.json`), so the
// loading registry has to hand them those bytes whatever processor files the checkpoint ships. These tests cover
// the wiring (factory registration, processor type, processor bytes) and deterministic preprocessing shapes and
// normalisation constants. They do NOT compare pixel values with a reference implementation: no reference is
// available air-gapped, and CoreImage bicubic is not PIL bicubic.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

/// Byte-per-token double: the rendered prompt is the joined message contents, one token per UTF-8 byte.
private struct ByteTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map { UInt8(truncatingIfNeeded: $0) }, as: UTF8.self)
    }
    func convertTokenToId(_ token: String) -> Int? { token.utf8.count == 1 ? token.utf8.first.map(Int.init) : nil }
    func convertIdToToken(_ id: Int) -> String? { (0 ... 255).contains(id) ? String(UnicodeScalar(UInt8(id))) : nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.compactMap { $0["content"] as? String }.joined(separator: "|").utf8.map(Int.init)
    }
}

@Suite("SharpAI config-backed VLM processors (SL-51 PR3)")
struct SharpAIVLMConfigProcessorTests {

    struct Case: Sendable, CustomStringConvertible {
        let modelType: String
        let processorType: String
        let json: String
        var description: String { modelType }
    }

    static let cases: [Case] = [
        .init(modelType: "eagle_2_5_vl", processorType: "SharpAIEagle25VlProcessor",
              json: SharpAIEagle25VlTestFixtures.tinyJSON),
        .init(modelType: "moondream3", processorType: "SharpAIMoondream3Processor",
              json: SharpAIMoondream3TestFixtures.tinyJSON),
        .init(modelType: "phi4-siglip", processorType: "SharpAIPhi4SiglipProcessor",
              json: SharpAIPhi4SiglipTestFixtures.tinyJSON),
        .init(modelType: "qianfan_ocr", processorType: "SharpAIQianfanOcrProcessor",
              json: SharpAIQianfanOcrTestConfig.tiny()),
    ]

    private static func tempDirectory(files: [String: String]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sharpai-pr3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for (name, text) in files { try Data(text.utf8).write(to: dir.appendingPathComponent(name)) }
        return dir
    }

    private static func solidImage(_ color: CIColor, width: Int, height: Int) -> UserInput.Image {
        .ciImage(CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: width, height: height)))
    }

    @Test("the model_type is wired and resolves in VLMTypeRegistry.shared", arguments: cases)
    func modelTypeIsRegistered(_ c: Case) async {
        #expect(SharpAIVLMModelOverlays.vlmWiredKeys.contains(c.modelType))
        #expect(await VLMTypeRegistry.shared.contains(c.modelType), "\(c.modelType) is not registered")
    }

    @Test("the processor type is registered and builds from config.json bytes", arguments: cases)
    func processorTypeIsRegistered(_ c: Case) async throws {
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data(c.json.utf8), processorType: c.processorType, tokenizer: ByteTokenizer())
        #expect(String(describing: type(of: processor)).hasPrefix("SharpAI"))
    }

    @Test("no processor file in the checkpoint: config.json is the fallback", arguments: cases)
    func fallbackWhenNoProcessorFile(_ c: Case) async throws {
        let dir = try Self.tempDirectory(files: [:])
        defer { try? FileManager.default.removeItem(at: dir) }
        let context = VLMProcessorLoadingContext(modelId: "m", modelType: c.modelType, configurationData: Data(c.json.utf8))
        let resolved = try await resolveProcessorConfiguration(
            from: dir, context: context, registry: VLMProcessorLoadingRegistry.shared)
        #expect(resolved.processorType == c.processorType)
        #expect(resolved.data == Data(c.json.utf8))
    }

    @Test("a preprocessor_config.json in the checkpoint does not replace config.json", arguments: cases)
    func processorFileDoesNotWin(_ c: Case) async throws {
        let dir = try Self.tempDirectory(files: [
            "preprocessor_config.json": #"{"processor_class":"SomethingElseProcessor","image_mean":[0.1,0.2,0.3]}"#
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        let context = VLMProcessorLoadingContext(modelId: "m", modelType: c.modelType, configurationData: Data(c.json.utf8))
        let resolved = try await resolveProcessorConfiguration(
            from: dir, context: context, registry: VLMProcessorLoadingRegistry.shared)
        #expect(resolved.processorType == c.processorType)
        #expect(resolved.data == Data(c.json.utf8))
    }

    @Test("other model types keep their own processor file")
    func otherModelsAreUntouched() async throws {
        let dir = try Self.tempDirectory(files: [
            "preprocessor_config.json": #"{"processor_class":"Qwen3VLProcessor"}"#
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        let context = VLMProcessorLoadingContext(modelId: "m", modelType: "qwen3_vl", configurationData: Data("{}".utf8))
        let resolved = try await resolveProcessorConfiguration(
            from: dir, context: context, registry: VLMProcessorLoadingRegistry.shared)
        #expect(resolved.processorType == "Qwen3VLProcessor")
        #expect(resolved.data == Data(#"{"processor_class":"Qwen3VLProcessor"}"#.utf8))
    }

    @Test("eagle_2_5_vl: one 28x28 image is one tile, normalised with the ImageNet constants")
    func eagleSingleTile() async throws {
        let json = SharpAIEagle25VlTestFixtures.tinyJSON
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data(json.utf8), processorType: "SharpAIEagle25VlProcessor", tokenizer: ByteTokenizer())
        var input = UserInput(prompt: "hi")
        input.images = [Self.solidImage(.white, width: 28, height: 28)]
        let lm = try await processor.prepare(input: input)
        let pixels = try #require(lm.image?.pixels)
        // InternVL tiling: a square image at or below one tile gets exactly one tile and no thumbnail.
        #expect(pixels.shape == [1, 3, 28, 28])
        // White maps to (1 - mean) / std per channel; constants in SharpAIEagle25VlProcessor.imageMean / imageStd.
        let expected: [Float] = [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]
        for channel in 0 ..< 3 {
            let value = pixels[0, channel, 14, 14].item(Float.self)
            #expect(abs(value - expected[channel]) < 0.02, "channel \(channel): \(value) vs \(expected[channel])")
        }
        // One tile times imageSeqLength (1 for the tiny config) context markers in the rendered prompt.
        let text = ByteTokenizer().decode(tokenIds: lm.text.tokens.asArray(Int.self), skipSpecialTokens: false)
        #expect(text.components(separatedBy: "<IMG_CONTEXT>").count - 1 == 1)
        #expect(text.hasPrefix("<img>") && text.contains("</img>"))
    }

    @Test("eagle_2_5_vl: a text-only prompt carries no image")
    func eagleTextOnly() async throws {
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data(SharpAIEagle25VlTestFixtures.tinyJSON.utf8),
            processorType: "SharpAIEagle25VlProcessor", tokenizer: ByteTokenizer())
        let lm = try await processor.prepare(input: UserInput(prompt: "hi"))
        #expect(lm.image == nil)
        #expect(lm.text.tokens.shape == [1, 2])
    }

    @Test("qianfan_ocr: one 28x28 image is one channel-last tile, ImageNet-normalised, one context run")
    func qianfanSingleTile() async throws {
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data(SharpAIQianfanOcrTestConfig.tiny().utf8),
            processorType: "SharpAIQianfanOcrProcessor", tokenizer: ByteTokenizer())
        var input = UserInput(prompt: "hi")
        input.images = [Self.solidImage(.white, width: 28, height: 28)]
        let lm = try await processor.prepare(input: input)
        let pixels = try #require(lm.image?.pixels)
        // Channel last, as SharpAIQianfanOcr.encodeImages requires. A square image gets one tile, no thumbnail.
        #expect(pixels.shape == [1, 28, 28, 3])
        let expected: [Float] = [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]
        for channel in 0 ..< 3 {
            let value = pixels[0, 14, 14, channel].item(Float.self)
            #expect(abs(value - expected[channel]) < 0.02, "channel \(channel): \(value) vs \(expected[channel])")
        }
        // Tiny config: 28 / 14 = 2, shuffled by 2 gives one token per tile, so one context marker in all.
        let text = ByteTokenizer().decode(tokenIds: lm.text.tokens.asArray(Int.self), skipSpecialTokens: false)
        #expect(text.components(separatedBy: "<IMG_CONTEXT>").count - 1 == 1)
        #expect(text == "<img><IMG_CONTEXT></img>\nhi")
    }

    @Test("qianfan_ocr: a 2:1 image is two grid tiles plus a thumbnail, and each tile adds its tokens")
    func qianfanTwoTilesAndThumbnail() async throws {
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data(SharpAIQianfanOcrTestConfig.tiny().utf8),
            processorType: "SharpAIQianfanOcrProcessor", tokenizer: ByteTokenizer())
        var input = UserInput(prompt: "hi")
        input.images = [Self.solidImage(.white, width: 56, height: 28)]
        let lm = try await processor.prepare(input: input)
        let pixels = try #require(lm.image?.pixels)
        #expect(pixels.shape == [3, 28, 28, 3])
        let text = ByteTokenizer().decode(tokenIds: lm.text.tokens.asArray(Int.self), skipSpecialTokens: false)
        #expect(text.components(separatedBy: "<IMG_CONTEXT>").count - 1 == 3)
    }

    @Test("qianfan_ocr: several images are labelled Image-N and stacked in order")
    func qianfanMultipleImages() async throws {
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data(SharpAIQianfanOcrTestConfig.tiny().utf8),
            processorType: "SharpAIQianfanOcrProcessor", tokenizer: ByteTokenizer())
        var input = UserInput(prompt: "hi")
        input.images = [
            Self.solidImage(.white, width: 28, height: 28), Self.solidImage(.black, width: 28, height: 28),
        ]
        let lm = try await processor.prepare(input: input)
        let pixels = try #require(lm.image?.pixels)
        #expect(pixels.shape == [2, 28, 28, 3])
        // White is above black at the centre pixel, so the order of the inputs is kept.
        #expect(pixels[0, 14, 14, 0].item(Float.self) > pixels[1, 14, 14, 0].item(Float.self))
        let text = ByteTokenizer().decode(tokenIds: lm.text.tokens.asArray(Int.self), skipSpecialTokens: false)
        #expect(text.hasPrefix("Image-1: <img>") && text.contains("Image-2: <img>"))
    }

    @Test("qianfan_ocr: a text-only prompt carries no image")
    func qianfanTextOnly() async throws {
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data(SharpAIQianfanOcrTestConfig.tiny().utf8),
            processorType: "SharpAIQianfanOcrProcessor", tokenizer: ByteTokenizer())
        let lm = try await processor.prepare(input: UserInput(prompt: "hi"))
        #expect(lm.image == nil)
        #expect(lm.text.tokens.shape == [1, 2])
    }

    @Test("moondream3: image prefix, crop stack shape and value range")
    func moondreamImage() async throws {
        let json = SharpAIMoondream3TestFixtures.tinyJSON
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data(json.utf8), processorType: "SharpAIMoondream3Processor", tokenizer: ByteTokenizer())
        var input = UserInput(prompt: "hi")
        input.images = [Self.solidImage(.white, width: 28, height: 28)]
        let lm = try await processor.prepare(input: input)
        let image = try #require(lm.image)
        // Global crop plus a 1x1 local grid, 3 channels, crop_size 28 (tiny fixture).
        #expect(image.pixels.shape == [2, 3, 28, 28])
        #expect(image.frames?.first?.h == 1 && image.frames?.first?.w == 1)
        // Pixels are mapped to [-1, 1] (`* 2 - 1`); white sits at +1.
        #expect(abs(image.pixels.max().item(Float.self) - 1) < 0.02)
        #expect(image.pixels.min().item(Float.self) >= -1.0001)
        // 1 + enc_n_layers^2 = 5 placeholder positions come first (fixture comment: "Image prefix is 1 + 2*2").
        let tokens = lm.text.tokens.asArray(Int.self)
        let placeholder = Int(UInt8(ascii: ">"))  // last byte of "<|endoftext|><image>" under the byte double
        #expect(Array(tokens.prefix(5)) == Array(repeating: placeholder, count: 5))
    }

    @Test("phi4-siglip: NaFlex grid within the patch budget, flat patch rows, image token expansion")
    func phi4Image() async throws {
        let json = SharpAIPhi4SiglipTestFixtures.tinyJSON
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data(json.utf8), processorType: "SharpAIPhi4SiglipProcessor", tokenizer: ByteTokenizer())
        var input = UserInput(prompt: "hi")
        input.images = [Self.solidImage(.white, width: 8, height: 8)]
        let lm = try await processor.prepare(input: input)
        let image = try #require(lm.image)
        // patch_size 2, max_num_patches 16, 8x8 input: a 4x4 grid (16 patches) of 2*2*3 = 12 values each.
        #expect(image.pixels.shape == [16, 12])
        #expect(image.frames?.first?.h == 4 && image.frames?.first?.w == 4)
        // mean = std = 0.5, so white maps to (1 - 0.5) / 0.5 = 1.
        #expect(abs(image.pixels.min().item(Float.self) - 1) < 0.02)
        // The "<image>" placeholder becomes one image token (id 31) per patch.
        let tokens = lm.text.tokens.asArray(Int.self)
        #expect(tokens.filter { $0 == 31 }.count == 16)
    }
}
