// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXVLM
import Testing

private struct TemplateTokenizer: Tokenizer {
    let bosToken: String? = nil
    let eosToken: String? = nil
    let unknownToken: String? = nil

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [5, 6, 7, 8] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [5, 6, 7, 8] }
}

@Suite("VLM text-only prepare")
struct VLMTextOnlyPrepareTests {

    static let processors: [(type: String, configuration: String)] = [
        (
            "FastVLMProcessor",
            #"{"image_mean": [0, 0, 0], "image_std": [1, 1, 1], "crop_size": {"width": 8, "height": 8}}"#
        ),
        (
            "Gemma3Processor",
            #"""
            {"processor_class": "Gemma3Processor", "image_processor_type": "Gemma3ImageProcessor",
             "do_normalize": true, "do_rescale": true, "do_resize": true,
             "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "image_seq_length": 4,
             "resample": 2, "rescale_factor": 0.00392156862745098, "size": {"height": 8, "width": 8}}
            """#
        ),
        (
            "Gemma4Processor",
            #"""
            {"processor_class": "Gemma4Processor", "do_normalize": true, "image_seq_length": 280,
             "image_token_id": 258880, "video_token_id": 258884, "video_soft_tokens_per_frame": 70}
            """#
        ),
        ("Gemma4UnifiedProcessor", "{}"),
        (
            "Mistral3Processor",
            #"""
            {"image_processor": {"image_mean": [0, 0, 0], "image_std": [1, 1, 1],
             "size": {"longest_edge": 16}, "patch_size": 4}, "image_token": "[IMG]", "patch_size": 4}
            """#
        ),
        (
            "SmolVLMProcessor",
            #"""
            {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.25, 0.25, 0.25],
             "size": {"longest_edge": 16}, "max_image_size": {"longest_edge": 8},
             "video_sampling": {"fps": 2, "max_frames": 5}, "image_seq_len": 4}
            """#
        ),
    ]

    @Test(
        "Text-only input carries the prompt and no attention mask",
        arguments: processors.map(\.type))
    func noMaskWithoutMedia(processorType: String) async throws {
        let configuration = try #require(Self.processors.first { $0.type == processorType })
        let processor = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data(configuration.configuration.utf8),
            processorType: processorType, tokenizer: TemplateTokenizer())

        let input = try await processor.prepare(input: UserInput(chat: [.user("hello")]))

        #expect(input.text.mask == nil)
        #expect(input.text.tokens.shape == [1, 4])
        #expect(input.image == nil)
    }
}
