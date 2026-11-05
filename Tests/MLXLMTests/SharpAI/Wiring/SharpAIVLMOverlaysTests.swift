// SharpAI SL-51 PR1: wiring tests for the VLM overlays (text-only models and the shared processor shim).

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXVLM

/// Byte-per-token double: `applyChatTemplate` joins the message contents, so the rendered prompt is observable.
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

@Suite("SharpAI VLM overlay wiring (SL-51 PR1)")
struct SharpAIVLMOverlaysTests {

    /// `model_type`s from `Libraries/MLXVLM/SharpAI` whose creator always throws or returns a model that cannot run
    /// (docs/research/sl51-vlm-scout-2026-11-01.md). Registering one would make `contains` true while every load
    /// fails, so they must stay out.
    static let blockedKeys: [String] = [
        "NemotronH_Nano_VL_V2",
        "alpamayo2_super",
        "cohere_compass",
        "cosmos3_edge",
        "deepseek_vl_v2",
        "diffusion_gemma",
        "dots3_note",
        "exaone4_5",
        "funaudiochat",
        "glm5_next",
        "granite4_vision",
        "hunyuan_vl",
        "hyperclovax_vision_v2",
        "inkling_mm_model",
        "kanana-1.5-v",
        "llama_nemotron_vl",
        "ming_v3_asr",
        "minimax_m3_vl",
        "mistral_large4",
        "molmo2",
        "muse_glimmer_assistant",
        "muse_spark",
        "nemo_speechlm",
        "ovis2_6_moe",
        "step3p7",
        "step5_mtp",
        "step5_vl",
        "ultravbert",
    ]

    @Test("every wired key resolves in VLMTypeRegistry.shared")
    func wiredKeysAreRegistered() async {
        #expect(SharpAIVLMModelOverlays.vlmWiredKeys.count == 15)
        #expect(SharpAIVLMModelOverlays.textOnlyKeys.isSubset(of: SharpAIVLMModelOverlays.vlmWiredKeys))
        #expect(SharpAIVLMModelOverlays.vlmWiredKeys.subtracting(SharpAIVLMModelOverlays.textOnlyKeys) == ["cosmos3_omni", "eagle_2_5_vl", "intern_s2_preview", "moondream3", "phi4-siglip"])
        for key in SharpAIVLMModelOverlays.vlmWiredKeys.sorted() {
            #expect(await VLMTypeRegistry.shared.contains(key), "\(key) is wired but not registered")
        }
    }

    @Test("the aggregate holds a creator for exactly the wired keys")
    func aggregateMatchesWiredKeys() {
        #expect(Set(SharpAIVLMModelOverlays.vlm.keys) == SharpAIVLMModelOverlays.vlmWiredKeys)
    }

    @Test("creators that cannot produce a working model stay unregistered", arguments: blockedKeys)
    func blockedKeysStayOut(_ key: String) async {
        #expect(!SharpAIVLMModelOverlays.vlmWiredKeys.contains(key))
        #expect(!(await VLMTypeRegistry.shared.contains(key)), "\(key) must not be registered")
    }

    @Test("upstream wins when both sides register the same key")
    func upstreamWins() {
        #expect(SharpAIVLMModelOverlays.upstreamWins("upstream", "ours") == "upstream")
        #expect(SharpAIVLMModelOverlays.vlmWiredKeys.isDisjoint(with: ["paligemma", "qwen3_vl", "qwen3_5", "gemma4"]))
    }

    @Test("the text-only processor type is registered and builds a processor")
    func processorTypeIsRegistered() async throws {
        let p = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data("{}".utf8), processorType: SharpAITextOnlyProcessor.processorType,
            tokenizer: ByteTokenizer())
        #expect(p is SharpAITextOnlyProcessor)
    }

    @Test("text-only models resolve to the shim whatever the checkpoint declares", arguments: SharpAIVLMModelOverlays.textOnlyKeys.sorted())
    func resolverForcesShim(_ key: String) throws {
        let context = VLMProcessorLoadingContext(modelId: "m", modelType: key, configurationData: Data("{}".utf8))
        let registry = VLMProcessorLoadingRegistry.shared
        #expect(try registry.processorType(for: context, declaredProcessorType: "InternS1Processor") == SharpAITextOnlyProcessor.processorType)
        #expect(try registry.processorType(for: context, declaredProcessorType: nil) == SharpAITextOnlyProcessor.processorType)
        let fallback = try registry.fallbackProcessorConfiguration(for: context)
        #expect(fallback?.processorType == SharpAITextOnlyProcessor.processorType)
    }

    @Test("cosmos3_omni resolves to the upstream Qwen3-VL processor")
    func cosmosUsesQwen3VLProcessor() throws {
        let context = VLMProcessorLoadingContext(modelId: "m", modelType: "cosmos3_omni", configurationData: Data("{}".utf8))
        let registry = VLMProcessorLoadingRegistry.shared
        #expect(try registry.processorType(for: context, declaredProcessorType: nil) == "Qwen3VLProcessor")
        #expect(try registry.processorType(for: context, declaredProcessorType: "Cosmos3Processor") == "Qwen3VLProcessor")
    }

    @Test("a model that keeps its own processor is left alone")
    func resolverIgnoresOtherModels() throws {
        let registry = VLMProcessorLoadingRegistry.shared
        for key in ["intern_s2_preview", "paligemma", "qwen3_vl"] {
            let context = VLMProcessorLoadingContext(modelId: "m", modelType: key, configurationData: Data("{}".utf8))
            #expect(try registry.processorType(for: context, declaredProcessorType: "Qwen3VLProcessor") != SharpAITextOnlyProcessor.processorType)
            #expect(try registry.fallbackProcessorConfiguration(for: context)?.processorType != SharpAITextOnlyProcessor.processorType)
        }
    }

    @Test("the shim renders the chat and returns tokens without a mask")
    func shimTokenizesText() async throws {
        let processor = SharpAITextOnlyProcessor(.init(), tokenizer: ByteTokenizer())
        let lm = try await processor.prepare(input: UserInput(prompt: "hi"))
        #expect(lm.text.tokens.shape == [1, 2])
        #expect(lm.text.mask == nil)
        #expect(lm.image == nil)
    }

    @Test("the shim refuses media instead of dropping it")
    func shimRejectsMedia() async {
        let processor = SharpAITextOnlyProcessor(.init(), tokenizer: ByteTokenizer())
        var withImage = UserInput(prompt: "hi")
        withImage.images = [.ciImage(CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4)))]
        await #expect(throws: SharpAITextOnlyProcessorError.mediaNotSupported("image")) {
            _ = try await processor.prepare(input: withImage)
        }
    }
}
