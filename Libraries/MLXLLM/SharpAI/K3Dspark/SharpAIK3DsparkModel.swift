// SharpAI fork overlay: k3_dspark draft backbone and heads. Not part of upstream mlx-swift-lm.
//
// The draft shares embed_tokens and lm_head with the frozen target. Neither is owned here,
// so the caller passes input embeddings and projects logits with the target head.
// The Markov bias is returned as [B, draftVocab] and is added to the target logits by the caller.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIK3DsparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var markovW1: Embedding
    @ModuleInfo(key: "markov_w2") var markovW2: Linear

    init(vocabSize: Int, draftVocabSize: Int, markovRank: Int) {
        self._markovW1.wrappedValue = Embedding(
            embeddingCount: vocabSize, dimensions: markovRank)
        self._markovW2.wrappedValue = Linear(markovRank, draftVocabSize, bias: false)
    }

    /// Token ids [B, L] to Markov embeddings [B, L, rank].
    func embed(_ tokenIds: MLXArray) -> MLXArray {
        markovW1(tokenIds)
    }

    /// Markov embeddings [B, L, rank] to a draft-vocab bias [B, L, draftVocab].
    func bias(_ markovEmbed: MLXArray) -> MLXArray {
        markovW2(markovEmbed)
    }
}

class SharpAIK3DsparkConfidenceHead: Module {
    let withMarkov: Bool
    @ModuleInfo(key: "proj") var proj: Linear

    init(inputDim: Int, withMarkov: Bool) {
        self.withMarkov = withMarkov
        self._proj.wrappedValue = Linear(inputDim, 1, bias: true)
    }

    /// Hidden [B, L, hidden] and Markov embedding [B, L, rank] to logits [B, L].
    func callAsFunction(_ hidden: MLXArray, markovEmbed: MLXArray) -> MLXArray {
        let x = (withMarkov ? concatenated([hidden, markovEmbed], axis: -1) : hidden)
            .asType(.float32)
        return proj(x).squeezed(axis: -1)
    }
}

class SharpAIK3DsparkBackbone: Module {
    @ModuleInfo(key: "context_proj") var contextProj: Linear
    @ModuleInfo(key: "context_norm") var contextNorm: RMSNorm
    var layers: [SharpAIK3DsparkDecoderLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: RMSNorm
    @ModuleInfo(key: "markov_head") var markovHead: SharpAIK3DsparkMarkovHead
    @ModuleInfo(key: "confidence_head") var confidenceHead: SharpAIK3DsparkConfidenceHead?

    init(_ config: SharpAIK3DsparkConfiguration) {
        self._contextProj.wrappedValue = Linear(
            config.targetHiddenSize * config.numTargetLayers, config.hiddenSize, bias: false)
        self._contextNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.layers = (0 ..< config.numHiddenLayers).map { _ in
            SharpAIK3DsparkDecoderLayer(config)
        }
        self._finalNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._markovHead.wrappedValue = SharpAIK3DsparkMarkovHead(
            vocabSize: config.vocabSize,
            draftVocabSize: config.draftVocabSize,
            markovRank: config.markovRank)
        if config.enableConfidenceHead {
            let inputDim =
                config.hiddenSize + (config.confidenceHeadWithMarkov ? config.markovRank : 0)
            self._confidenceHead.wrappedValue = SharpAIK3DsparkConfidenceHead(
                inputDim: inputDim, withMarkov: config.confidenceHeadWithMarkov)
        } else {
            self._confidenceHead.wrappedValue = nil
        }
    }

    /// Target hidden states [B, Lc, targetHidden * numTargetLayers] to context states [B, Lc, hidden].
    func combineHiddenStates(_ targetHidden: MLXArray) -> MLXArray {
        contextNorm(contextProj(targetHidden))
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray, targetHidden: MLXArray?, contextOffset: Int, draftOffset: Int
    ) -> MLXArray {
        let context = targetHidden.map { self.combineHiddenStates($0) }
        var h = inputsEmbeds
        for layer in layers {
            h = layer(h, context: context, contextOffset: contextOffset, draftOffset: draftOffset)
        }
        return finalNorm(h)
    }
}

class SharpAIK3DsparkDraftModel: Module {
    let configuration: SharpAIK3DsparkConfiguration
    @ModuleInfo(key: "model") var model: SharpAIK3DsparkBackbone

    init(_ configuration: SharpAIK3DsparkConfiguration) {
        self.configuration = configuration
        self._model.wrappedValue = SharpAIK3DsparkBackbone(configuration)
    }

    /// Draft block states [B, L, hidden]. Offsets give the first position of each block.
    func callAsFunction(
        _ inputsEmbeds: MLXArray, targetHidden: MLXArray? = nil,
        contextOffset: Int = 0, draftOffset: Int = 0
    ) -> MLXArray {
        model(
            inputsEmbeds, targetHidden: targetHidden,
            contextOffset: contextOffset, draftOffset: draftOffset)
    }

    func markovEmbed(_ tokenIds: MLXArray) -> MLXArray {
        model.markovHead.embed(tokenIds)
    }

    func markovBias(_ markovEmbed: MLXArray) -> MLXArray {
        model.markovHead.bias(markovEmbed)
    }

    /// Per-position acceptance probability, or nil when the checkpoint has no confidence head.
    func confidence(_ hidden: MLXArray, markovEmbed: MLXArray) -> MLXArray? {
        model.confidenceHead.map { sigmoid($0(hidden, markovEmbed: markovEmbed)) }
    }
}
