// SharpAI text-only shim tests for openpangu_vl (air-gapped: synthetic weights, no download). Not part of upstream.
//
// The ViT is not ported, so the model is wired text-only: the shared SharpAITextOnlyProcessor rejects media with a
// clear error and the model decodes text. The mrope interleave order is unverified against a checkpoint, but it
// does not affect text-only input.

import CoreImage
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

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

@Suite("SharpAI openpangu_vl text-only shim", .serialized)
struct SharpAIOpenpanguVlTextOnlyTests {

    static let vocab = 64

    static let json = """
        {"model_type":"openpangu_vl","hidden_size":32,"intermediate_size":64,"num_attention_heads":4,
         "num_key_value_heads":2,"num_hidden_layers":2,"rms_norm_eps":1e-05,"rope_theta":10000.0,
         "max_position_embeddings":128,"tie_word_embeddings":false,"vocab_size":64,
         "vision_config":{"depth":1,"hidden_size":16,"intermediate_size":32,"num_heads":2,"in_chans":3,
         "out_hidden_size":32,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2,
         "window_size":112,"fullatt_block_indexes":[0]}}
        """

    @Test("the model_type resolves to the text-only processor and the shim rejects an image")
    func resolvesToShimAndRejectsImage() async throws {
        // A registry holding the same resolver the integrator wires into `shared` for this key.
        let registry = VLMProcessorLoadingRegistry(resolvers: [
            SharpAITextOnlyProcessorLoadingResolver(modelTypes: ["openpangu_vl"])
        ])
        let context = VLMProcessorLoadingContext(
            modelId: "m", modelType: "openpangu_vl", configurationData: Data("{}".utf8))
        for declared in [nil, "Qwen25VLProcessor", "OpenPanguVLProcessor"] as [String?] {
            #expect(try registry.processorType(for: context, declaredProcessorType: declared)
                == SharpAITextOnlyProcessor.processorType)
        }
        let fallback = try registry.fallbackProcessorConfiguration(for: context)
        #expect(fallback?.processorType == SharpAITextOnlyProcessor.processorType)

        let built = try await VLMProcessorTypeRegistry.shared.createModel(
            configuration: fallback!.data, processorType: fallback!.processorType,
            tokenizer: ByteTokenizer())
        let processor = try #require(built as? SharpAITextOnlyProcessor)

        var input = UserInput(prompt: "hi")
        input.images = [.ciImage(CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4)))]
        await #expect(throws: SharpAITextOnlyProcessorError.mediaNotSupported("image")) {
            _ = try await processor.prepare(input: input)
        }
    }

    @Test("a synthetic checkpoint builds through the creator and prepares text-only with finite logits")
    func textOnlyPrepareIsFinite() throws {
        let creator = try #require(SharpAIOpenpanguVlOverlay.creators["openpangu_vl"])
        let model = try creator(Data(Self.json.utf8))

        var arrays: [String: MLXArray] = [:]
        for (name, p) in model.parameters().flattened() {
            arrays[name] = SharpAIAirgap.fill(name: name, shape: p.shape, dtype: p.dtype)
        }
        #expect(!arrays.isEmpty)
        model.update(parameters: ModuleParameters.unflattened(arrays))
        eval(model)

        let tokens = SharpAIAirgap.tokens(vocab: Self.vocab)
        let cache = try model.newCache(parameters: nil)
        let result = try model.prepare(
            LMInput(text: .init(tokens: tokens)), cache: cache, state: nil, prefill: PrefillParameters())
        guard case .logits(let output) = result else {
            Issue.record("expected .logits from prepare")
            return
        }
        let logits = output.logits
        #expect(logits.ndim == 3)
        #expect(logits.dim(0) == tokens.dim(0))
        #expect(logits.dim(2) == Self.vocab)
        #expect(logits.dim(1) >= 1)
        let finite = isFinite(logits).all().item(Bool.self)
        #expect(finite)
    }
}
