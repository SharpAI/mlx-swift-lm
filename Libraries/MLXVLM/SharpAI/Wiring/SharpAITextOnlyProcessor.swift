// SharpAI fork: SL-51 PR1. Not part of upstream mlx-swift-lm.
//
// `VLMModelFactory` always resolves a processor. A model whose vision and audio towers are not ported has no
// processor of its own, so these text-only models share this one. It renders the chat template and tokenizes;
// it does not read images, video or audio, and says so with an error instead of dropping them silently.

import Foundation
import MLX
import MLXLMCommon

/// Failure raised when a text-only model is handed media it cannot read.
public enum SharpAITextOnlyProcessorError: LocalizedError, Equatable {
    case mediaNotSupported(String)

    public var errorDescription: String? {
        switch self {
        case .mediaNotSupported(let kind):
            return "This model is wired as text-only; its \(kind) input path is not ported."
        }
    }
}

/// `processor_config` for ``SharpAITextOnlyProcessor``. It carries no settings; every field in a
/// checkpoint's own file is ignored.
public struct SharpAITextOnlyProcessorConfiguration: Codable, Sendable {
    public init() {}
    public init(from decoder: Decoder) throws {}
    public func encode(to encoder: Encoder) throws {}
}

public struct SharpAITextOnlyProcessor: UserInputProcessor {
    public static let processorType = "SharpAITextOnlyProcessor"

    private let tokenizer: any Tokenizer

    public init(_ configuration: SharpAITextOnlyProcessorConfiguration, tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        if !input.images.isEmpty { throw SharpAITextOnlyProcessorError.mediaNotSupported("image") }
        if !input.videos.isEmpty { throw SharpAITextOnlyProcessorError.mediaNotSupported("video") }
        if !input.audios.isEmpty { throw SharpAITextOnlyProcessorError.mediaNotSupported("audio") }

        let messages = DefaultMessageGenerator().generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        // No attention mask: a batch of one has nothing to pad, and a non-nil mask would make
        // `ChatSession` treat the turn as non-resumable and defeat prompt-cache reuse.
        return LMInput(text: .init(tokens: MLXArray(tokens).expandedDimensions(axis: 0)))
    }
}

/// Points the text-only `model_type`s at ``SharpAITextOnlyProcessor``, whatever their checkpoint declares.
///
/// A real checkpoint of these models may carry `processor_class: InternS1Processor` (or none at all). Neither
/// is registered, and the towers it would feed are not ported, so the processor type is forced here and a
/// missing `preprocessor_config.json` is replaced by an empty configuration.
public struct SharpAITextOnlyProcessorLoadingResolver: VLMProcessorLoadingResolver {
    public let modelTypes: Set<String>

    public init(modelTypes: Set<String>) {
        self.modelTypes = modelTypes
    }

    public func fallbackProcessorConfiguration(
        for context: VLMProcessorLoadingContext
    ) throws -> VLMProcessorConfiguration? {
        guard modelTypes.contains(context.modelType) else { return nil }
        return VLMProcessorConfiguration(
            data: Data("{}".utf8), processorType: SharpAITextOnlyProcessor.processorType)
    }

    public func processorType(
        for context: VLMProcessorLoadingContext, declaredProcessorType: String?
    ) throws -> String? {
        modelTypes.contains(context.modelType) ? SharpAITextOnlyProcessor.processorType : nil
    }
}
