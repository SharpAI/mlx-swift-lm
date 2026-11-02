// SharpAI SL-51: text-only wiring for opencua, air-gapped. No download, no network.
//
// `opencua` is not in `SharpAIVLMModelOverlays` yet (the integrator adds it to `textOnlyKeys`, `vlmWiredKeys` and
// `vlm`). These tests therefore build the same resolver the integrator will get, with `modelTypes: ["opencua"]`,
// so they hold before and after that edit. Image support is deferred: media is rejected, not dropped.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Byte-per-token double; `applyChatTemplate` joins message contents.
private struct OpencuaByteTokenizer: Tokenizer {
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

@Suite("SharpAI opencua text-only wiring (air-gapped)", .serialized)
struct SharpAIOpencuaWiringTests {

    private let registry = VLMProcessorLoadingRegistry(resolvers: [
        SharpAITextOnlyProcessorLoadingResolver(modelTypes: ["opencua"])
    ])

    @Test("opencua resolves to the text-only processor even though the checkpoint declares OpenCUAProcessor")
    func resolvesToShim() throws {
        let context = VLMProcessorLoadingContext(
            modelId: "m", modelType: "opencua", configurationData: Data("{}".utf8))
        #expect(
            try registry.processorType(for: context, declaredProcessorType: "OpenCUAProcessor")
                == SharpAITextOnlyProcessor.processorType)
        #expect(
            try registry.fallbackProcessorConfiguration(for: context)?.processorType
                == SharpAITextOnlyProcessor.processorType)
    }

    @Test("the resolved processor type builds through the processor registry")
    func processorBuilds() async throws {
        let creator = try #require(SharpAIVLMModelOverlays.processors[SharpAITextOnlyProcessor.processorType])
        let processor = try creator(Data("{}".utf8), OpencuaByteTokenizer())
        #expect(processor is SharpAITextOnlyProcessor)
    }

    @Test("media input is rejected with mediaNotSupported, never dropped")
    func mediaRejected() async throws {
        let processor = SharpAITextOnlyProcessor(.init(), tokenizer: OpencuaByteTokenizer())
        var input = UserInput(prompt: "click the button")
        input.images = [.ciImage(CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4)))]
        await #expect(throws: SharpAITextOnlyProcessorError.mediaNotSupported("image")) {
            _ = try await processor.prepare(input: input)
        }
        let text = try await processor.prepare(input: UserInput(prompt: "hi"))
        #expect(text.image == nil)
        #expect(text.text.tokens.shape == [1, 2])
    }

    @Test("a synthetic checkpoint loads through the creator and the real loader, text-only prepare gives finite logits")
    func airgapLoadAndPrepare() async throws {
        let json = SharpAIOpencuaTestFixtures.tinyJSON
        let data = Data(json.utf8)
        let creator = try #require(SharpAIOpencuaOverlay.creators["opencua"])

        let reference = try creator(data)
        var arrays: [String: MLXArray] = [:]
        for (name, p) in reference.parameters().flattened() {
            arrays[name] = SharpAIAirgap.fill(name: name, shape: p.shape, dtype: p.dtype)
        }
        #expect(!arrays.isEmpty)
        reference.update(parameters: ModuleParameters.unflattened(arrays))
        eval(reference)

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sharpai-airgap-opencua-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try data.write(to: dir.appendingPathComponent("config.json"))
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))

        let loaded = try creator(data)
        try await loadWeights(modelDirectory: dir, model: loaded)
        eval(loaded)

        let tokens = SharpAIAirgap.tokens(vocab: 128)
        func logits(_ model: any LanguageModel) throws -> MLXArray {
            let cache = try model.newCache(parameters: nil)
            let result = try model.prepare(
                LMInput(text: .init(tokens: tokens)), cache: cache, state: nil, prefill: PrefillParameters())
            switch result {
            case .logits(let out): return out.logits
            case .tokens(let text): return try model(text, cache: cache, state: nil).logits
            }
        }
        let a = try logits(reference)
        let b = try logits(loaded)
        eval(a, b)
        #expect(a.shape == [1, 5, 128])
        #expect(all(isFinite(a)).item(Bool.self))
        #expect(abs(a).max().item(Float.self) > 0)
        #expect(allClose(a, b, rtol: 1e-3, atol: 1e-4).item(Bool.self))
    }
}
