// SharpAI fork: qwen3_omni_moe text-only shim wiring tests. Air-gapped: deterministic synthetic weights, no network.
//
// These tests do not depend on the integrator's edit of `SharpAIVLMModelOverlays` (textOnlyKeys / vlmWiredKeys):
// the model is built through `SharpAIQwen3OmniMoeOverlay.creators` and the resolver is constructed with
// the literal key, which is what the integrator's wiring produces.

import CoreImage
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

/// Byte-per-token double: the rendered prompt is observable.
private struct OmniByteTokenizer: Tokenizer {
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

@Suite("SharpAI qwen3_omni_moe text-only wiring", .serialized)
struct SharpAIQwen3OmniMoeTextOnlyWiringTests {

    static let key = "qwen3_omni_moe"

    private func resolver() -> SharpAITextOnlyProcessorLoadingResolver {
        SharpAITextOnlyProcessorLoadingResolver(modelTypes: [Self.key])
    }

    private func context(_ type: String) -> VLMProcessorLoadingContext {
        VLMProcessorLoadingContext(modelId: "m", modelType: type, configurationData: Data("{}".utf8))
    }

    @Test("the verified key resolves to the text-only processor whatever the checkpoint declares")
    func resolverForcesShim() throws {
        let r = resolver()
        #expect(
            try r.processorType(for: context(Self.key), declaredProcessorType: "Qwen3OmniMoeProcessor")
                == SharpAITextOnlyProcessor.processorType)
        #expect(
            try r.processorType(for: context(Self.key), declaredProcessorType: nil)
                == SharpAITextOnlyProcessor.processorType)
        #expect(
            try r.fallbackProcessorConfiguration(for: context(Self.key))?.processorType
                == SharpAITextOnlyProcessor.processorType)
    }

    @Test("the unverified qwen3_omni alias and Qwen3-VL are left alone")
    func resolverIgnoresOthers() throws {
        let r = resolver()
        for other in ["qwen3_omni", "qwen3_vl"] {
            #expect(try r.processorType(for: context(other), declaredProcessorType: "X") == nil)
            #expect(try r.fallbackProcessorConfiguration(for: context(other)) == nil)
        }
    }

    @Test("the resolved processor type builds through the processor registry")
    func processorTypeBuilds() async throws {
        let type = try #require(
            try resolver().processorType(for: context(Self.key), declaredProcessorType: nil))
        let p = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: Data("{}".utf8), processorType: type, tokenizer: OmniByteTokenizer())
        #expect(p is SharpAITextOnlyProcessor)
    }

    @Test("text prepares without a mask and media is rejected, not dropped")
    func processorTextAndMedia() async throws {
        let p = SharpAITextOnlyProcessor(.init(), tokenizer: OmniByteTokenizer())
        let lm = try await p.prepare(input: UserInput(prompt: "hi"))
        #expect(lm.text.tokens.shape == [1, 2])
        #expect(lm.text.mask == nil)
        #expect(lm.image == nil && lm.video == nil && lm.audio == nil)

        var image = UserInput(prompt: "hi")
        image.images = [.ciImage(CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4)))]
        await #expect(throws: SharpAITextOnlyProcessorError.mediaNotSupported("image")) {
            _ = try await p.prepare(input: image)
        }
        var video = UserInput(prompt: "hi")
        video.videos = [.init(source: .url(URL(fileURLWithPath: "/nonexistent.mp4")))]
        await #expect(throws: SharpAITextOnlyProcessorError.mediaNotSupported("video")) {
            _ = try await p.prepare(input: video)
        }
        var audio = UserInput(prompt: "hi")
        audio.audios = [.init(source: .array(MLXArray([0, 0, 0] as [Float])))]
        await #expect(throws: SharpAITextOnlyProcessorError.mediaNotSupported("audio")) {
            _ = try await p.prepare(input: audio)
        }
    }

    @Test("synthetic thinker checkpoint loads through the real loader; text-only prepare gives finite logits")
    func loadsSyntheticCheckpoint() async throws {
        let json = SharpAIQwen3OmniMoeTestConfig.tiny()
        let data = Data(json.utf8)
        let creator = try #require(SharpAIQwen3OmniMoeOverlay.creators[Self.key])
        let reference = try creator(data)

        var arrays: [String: MLXArray] = [:]
        for (name, p) in reference.parameters().flattened() {
            arrays[name] = SharpAIAirgap.fill(name: name, shape: p.shape, dtype: p.dtype)
        }
        #expect(!arrays.isEmpty)
        reference.update(parameters: ModuleParameters.unflattened(arrays))
        eval(reference)

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sharpai-airgap-omni-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try data.write(to: dir.appendingPathComponent("config.json"))
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))

        let loaded = try creator(data)
        try await loadWeights(modelDirectory: dir, model: loaded)
        eval(loaded)

        let input = SharpAIAirgap.tokens(vocab: 64)
        func logits(_ model: any LanguageModel) throws -> MLXArray {
            let cache = try model.newCache(parameters: nil)
            let result = try model.prepare(
                LMInput(text: .init(tokens: input)), cache: cache, state: nil, prefill: PrefillParameters())
            guard case .logits(let output) = result else {
                Issue.record("expected logits from prepare")
                return MLXArray(0)
            }
            return output.logits
        }
        let a = try logits(reference)
        let b = try logits(loaded)
        eval(a, b)
        #expect(a.shape == [1, 5, 64])
        #expect(all(isFinite(a)).item(Bool.self), "reference logits not finite")
        #expect(abs(a).max().item(Float.self) > 0, "logits are all zero")
        #expect(allClose(a, b, rtol: 1e-3, atol: 1e-4).item(Bool.self), "loaded logits differ from the reference")
    }
}
