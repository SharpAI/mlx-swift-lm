// SharpAI fork: SL-51 PR3. Not part of upstream mlx-swift-lm.
//
// eagle_2_5_vl, moondream3 and phi4-siglip have processors that are built from the model's own
// `config.json` (image size, tiling limits, patch budget, image token id), not from a
// `preprocessor_config.json`. `VLMModelFactory` hands a processor the checkpoint's processor file, so this
// resolver (1) names the processor type, (2) supplies `config.json` when the checkpoint ships no processor
// file, and (3) replaces the bytes when it does: a shipped `preprocessor_config.json` would not decode as a
// model configuration.

import Foundation

public enum SharpAIConfigBackedProcessors {
    public static let eagle25Vl = "SharpAIEagle25VlProcessor"
    public static let moondream3 = "SharpAIMoondream3Processor"
    public static let phi4Siglip = "SharpAIPhi4SiglipProcessor"
    public static let qianfanOcr = "SharpAIQianfanOcrProcessor"
}

public struct SharpAIConfigBackedProcessorLoadingResolver: VLMProcessorLoadingResolver {
    /// `model_type` to processor type.
    public let processorTypes: [String: String]

    public init(processorTypes: [String: String]) {
        self.processorTypes = processorTypes
    }

    public func fallbackProcessorConfiguration(
        for context: VLMProcessorLoadingContext
    ) throws -> VLMProcessorConfiguration? {
        guard let type = processorTypes[context.modelType] else { return nil }
        return VLMProcessorConfiguration(data: context.configurationData, processorType: type)
    }

    public func processorType(
        for context: VLMProcessorLoadingContext, declaredProcessorType: String?
    ) throws -> String? {
        processorTypes[context.modelType]
    }

    public func processorConfigurationData(
        for context: VLMProcessorLoadingContext
    ) throws -> Data? {
        processorTypes[context.modelType] != nil ? context.configurationData : nil
    }
}
