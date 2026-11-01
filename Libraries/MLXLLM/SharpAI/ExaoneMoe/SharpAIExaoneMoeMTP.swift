//
//  SharpAIExaoneMoeMTP.swift
//  MLXLLM
//
//  Multi-token-prediction (MTP) predictor for `exaone_moe`, as in vLLM
//  exaone_moe_mtp.py. One step: normalize the input embedding and the target hidden
//  state, concatenate them, project with `fc`, run the MTP decoder layer(s), then
//  apply the final norm.
//  Not wired as a drafter. This does not conform to MTPDrafterModel. The MTP
//  predictor shares the target model's embedding and LM head (vLLM loads them from
//  the top level when `mtp.embed_tokens` is absent). UNVERIFIED: whether the target
//  hidden state is taken before or after the final norm.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIExaoneMoeMTPPredictor: Module {
    let numPredictLayers: Int

    @ModuleInfo(key: "fc") var fc: Linear
    let layers: [SharpAIExaoneMoeDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: RMSNorm

    init(_ config: SharpAIExaoneMoeConfiguration) {
        let count = max(config.numNextnPredictLayers, 1)
        self.numPredictLayers = count

        _fc.wrappedValue = Linear(config.hiddenSize * 2, config.hiddenSize, bias: false)
        self.layers = (0 ..< count).map {
            SharpAIExaoneMoeDecoderLayer(config, layer: $0, mtp: true)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFCNormHidden.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFCNormEmbedding.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// One MTP step. `step` selects the layer when there are several, as vLLM
    /// `spec_step_idx` does. `cache` holds one entry per MTP layer.
    func callAsFunction(
        _ inputIds: MLXArray, hidden: MLXArray, embedding: Embedding,
        cache: [KVCache]? = nil, step: Int = 0
    ) -> MLXArray {
        let index = step % numPredictLayers
        let inputEmbeds = preFCNormEmbedding(embedding(inputIds))
        var h = fc(concatenated([inputEmbeds, preFCNormHidden(hidden)], axis: -1))

        let layer = layers[index]
        let layerCache = cache?[index]
        let mask = createAttentionMask(h: h, cache: layerCache, windowSize: layer.slidingWindow)
        h = layer(h, mask: mask, cache: layerCache)
        return norm(h)
    }

    /// Strips the `mtp.` prefix and keeps only MTP weights. Expert stacking is not
    /// needed because MTP layers use a dense MLP.
    static func extractWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights where key.hasPrefix("mtp.") {
            result[String(key.dropFirst(4))] = value
        }
        return result
    }
}
