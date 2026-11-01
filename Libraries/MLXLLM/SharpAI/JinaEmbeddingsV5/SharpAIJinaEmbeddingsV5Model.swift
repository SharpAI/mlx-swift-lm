// SharpAI fork overlay: jina_embeddings_v5 text model. Not part of upstream.
// Layers come from Qwen3ModelInner (upstream type, reused by composition, not modified).
// The LM head and sanitize are local, following JinaRerankerModel.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public final class SharpAIJinaEmbeddingsV5Model: Module, LanguageModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let tieWordEmbeddings: Bool

    @ModuleInfo(key: "model") var model: Qwen3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ configuration: SharpAIJinaEmbeddingsV5Configuration) {
        let base = configuration.base
        self.vocabularySize = base.vocabularySize
        self.kvHeads = (0 ..< base.hiddenLayers).map { _ in base.kvHeads }
        self.tieWordEmbeddings = base.tieWordEmbeddings
        _model.wrappedValue = Qwen3ModelInner(base)
        if !base.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(base.hiddenSize, base.vocabularySize, bias: false)
        }
    }

    public func prepare(
        _ input: LMInput,
        cache: [KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        .init(logits: callAsFunction(input.tokens, cache: cache))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projectLogits(model(inputs, cache: cache))
    }

    private func projectLogits(_ hiddenStates: MLXArray) -> MLXArray {
        lmHead?(hiddenStates) ?? model.embedTokens.asLinear(hiddenStates)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIJinaEmbeddingsV5Sanitize.sanitize(weights, tiedWordEmbeddings: tieWordEmbeddings)
    }
}
