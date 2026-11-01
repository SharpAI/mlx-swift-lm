// SharpAI fork overlay: muse_spark_dspark draft backbone and heads. Not part of upstream mlx-swift-lm.
//
// Dense DSpark drafter. The target supplies the block input embeddings and the target hidden
// states, so this class owns no embed_tokens and no mask embedding. Structure and arithmetic
// follow vLLM's MuseSparkDSparkModel (vllm/models/muse_spark/common/dspark.py) and the Muse 1x
// layers (vllm/models/muse_spark/nvidia/model.py, common/layers.py).
// UNVERIFIED: the causal and sliding-window bounds over the context and draft keys (positions
// are absolute; the window is `query - key < window`), and the order of the Markov bias and
// the output logit transform. vLLM's segmented-causal backend is not read here.
// UNVERIFIED: vLLM rounds normed activations to the model's compute dtype. This port keeps the
// stream dtype and computes norms in fp32.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Weightless RMS norm over the last axis, computed in fp32 and cast back.
fileprivate func museWeightlessRMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    let y = x.asType(.float32)
    let variance = sum(y * y, axes: [-1], keepDims: true) / Float(x.dim(-1))
    return (y * rsqrt(variance + eps)).asType(x.dtype)
}

/// Muse RMS norm. `weightFirst` applies the weight before the norm, as the learnable-scale FFN
/// norm does. Zero-centered gamma adds `gainCenter` to the stored weight.
class SharpAIMuseSparkDsparkRMSNorm: Module {
    let eps: Float
    let zeroCenteredGamma: Bool
    let gainCenter: Float
    let weightFirst: Bool
    @ParameterInfo(key: "weight") var weight: MLXArray?

    init(
        dimensions: Int, eps: Float, hasWeight: Bool, zeroCenteredGamma: Bool = false,
        gainCenter: Float = 1, weightFirst: Bool = false
    ) {
        self.eps = eps
        self.zeroCenteredGamma = zeroCenteredGamma
        self.gainCenter = gainCenter
        self.weightFirst = weightFirst
        self._weight.wrappedValue = hasWeight ? MLXArray.ones([dimensions]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gain: MLXArray? = nil
        if let weight {
            let stored = weight.asType(.float32)
            gain = zeroCenteredGamma ? stored + gainCenter : stored
        }
        var y = x.asType(.float32)
        if weightFirst, let gain { y = y * gain }
        let variance = sum(y * y, axes: [-1], keepDims: true) / Float(x.dim(-1))
        y = y * rsqrt(variance + eps)
        if !weightFirst, let gain { y = y * gain }
        return y.asType(x.dtype)
    }
}

/// Gated residual `alpha * stream + beta * branch` with `beta = sigmoid(w)` and
/// `alpha = sqrt(max(eps, sigmoid(-w) * (1 + beta)))`, where `w = gate / 0.3`. Without the gate
/// it is a plain sum.
class SharpAIMuseSparkDsparkResidual: Module {
    let temperature: Float = 0.3
    let eps: Float = 1e-3
    @ParameterInfo(key: "gate") var gate: MLXArray?

    init(dimensions: Int, useGate: Bool) {
        self._gate.wrappedValue = useGate ? MLXArray.zeros([dimensions]) : nil
    }

    func callAsFunction(stream: MLXArray, branch: MLXArray) -> MLXArray {
        let dtype = stream.dtype
        guard let gate else {
            return stream + branch.asType(dtype)
        }
        let w = gate.asType(.float32) / temperature
        let beta = sigmoid(w)
        let alphaSquared = sigmoid(-w) * (beta + Float(1))
        let alpha = sqrt(maximum(alphaSquared, MLXArray(eps)))
        let blended = alpha * stream.asType(.float32) + beta * branch.asType(.float32)
        return blended.asType(dtype)
    }
}

class SharpAIMuseSparkDsparkAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let eps: Float
    let useQkNorm: Bool
    let useONorm: Bool
    let scale: Float
    let slidingWindow: Int?
    let rope: RoPELayer?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_gate_proj") var oGateProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "pre_qkv_proj_norm") var preQKVProjNorm: SharpAIMuseSparkDsparkRMSNorm

    init(_ config: SharpAIMuseSparkDsparkConfiguration, layerIndex: Int) {
        numHeads = config.numAttentionHeads
        numKVHeads = config.numKeyValueHeads
        headDim = config.headDim
        eps = config.rmsNormEps
        useQkNorm = config.useQkNorm
        useONorm = config.useAttnONorm
        slidingWindow = config.slidingWindow(forLayer: layerIndex)

        // Query scale used with QK norm, as in vLLM's MuseSpark1xAttention.
        let scaleQueryBy: Float =
            config.useSqrtAttnScaling
            ? 3.87 : 43.7840518911 / Float(config.headDim).squareRoot()
        var attentionScale = pow(Float(config.headDim), -0.5)
        if config.useQkNorm {
            attentionScale *= scaleQueryBy
        }
        scale = attentionScale

        let ropeLayer: RoPELayer? =
            config.layerUsesRope(layerIndex)
            ? initializeRope(
                dims: config.headDim, base: config.ropeTheta, traditional: true,
                scalingConfig: nil, maxPositionEmbeddings: config.maxPositionEmbeddings)
            : nil
        rope = ropeLayer

        let qDim = config.numAttentionHeads * config.headDim
        let kvDim = config.numKeyValueHeads * config.headDim
        self._qProj.wrappedValue = Linear(config.hiddenSize, qDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        if config.useAttnOGate {
            self._oGateProj.wrappedValue = Linear(config.hiddenSize, qDim, bias: false)
        } else {
            self._oGateProj.wrappedValue = nil
        }
        self._oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: false)
        self._preQKVProjNorm.wrappedValue = SharpAIMuseSparkDsparkRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps,
            hasWeight: config.normAffine, zeroCenteredGamma: config.normZeroCenteredGamma)
    }

    /// Keys and values for an already-normed input [B, T, H], each [B, Hkv, T, D].
    private func keysAndValues(_ h: MLXArray, offset: Int) -> (MLXArray, MLXArray) {
        let batch = h.dim(0)
        let length = h.dim(1)
        var k = kProj(h).reshaped(batch, length, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(h).reshaped(batch, length, numKVHeads, headDim).transposed(0, 2, 1, 3)
        if useQkNorm {
            k = museWeightlessRMSNorm(k, eps: eps)
        }
        if let rope {
            k = applyRotaryPosition(rope, to: k, offset: .scalar(offset))
        }
        return (k, v)
    }

    /// Additive mask [1, 1, L, Lc + L]. Context keys come first, then draft keys. Draft query
    /// i sits at `draftOffset + i`, and each key carries its absolute position.
    private func draftMask(
        length: Int, contextLength: Int, contextOffset: Int, draftOffset: Int, dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        var keyPositions = (0 ..< contextLength).map { contextOffset + $0 }
        keyPositions += (0 ..< length).map { draftOffset + $0 }
        var values: [Float] = []
        values.reserveCapacity(length * keyPositions.count)
        for i in 0 ..< length {
            let queryPosition = draftOffset + i
            for keyPosition in keyPositions {
                var visible = keyPosition <= queryPosition
                if let window = slidingWindow, queryPosition - keyPosition >= window {
                    visible = false
                }
                values.append(visible ? 0 : -1e9)
            }
        }
        // Match the query dtype: fused SDPA rejects a float32 mask on bf16/fp16 inputs.
        let mask = MLXArray(values).reshaped(1, 1, length, keyPositions.count).asType(dtype)
        return .array(mask)
    }

    /// `x` is the pre-normed draft stream [B, L, H]. `context` is the target state [B, Lc, H]
    /// after hidden_norm. This layer applies its own pre_qkv_proj_norm to it.
    func callAsFunction(
        _ x: MLXArray, context: MLXArray?, contextOffset: Int, draftOffset: Int
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)

        var q = qProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        if useQkNorm {
            q = museWeightlessRMSNorm(q, eps: eps)
        }
        if let rope {
            q = applyRotaryPosition(rope, to: q, offset: .scalar(draftOffset))
        }

        var (keys, values) = keysAndValues(x, offset: draftOffset)
        var contextLength = 0
        if let context {
            let (contextKeys, contextValues) = keysAndValues(
                preQKVProjNorm(context), offset: contextOffset)
            keys = concatenated([contextKeys, keys], axis: 2)
            values = concatenated([contextValues, values], axis: 2)
            contextLength = context.dim(1)
        }
        if numHeads != numKVHeads {
            let repeats = numHeads / numKVHeads
            keys = repeated(keys, count: repeats, axis: 1)
            values = repeated(values, count: repeats, axis: 1)
        }

        let mask = draftMask(
            length: length, contextLength: contextLength,
            contextOffset: contextOffset, draftOffset: draftOffset, dtype: q.dtype)
        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale, mask: mask)

        if useONorm {
            out = museWeightlessRMSNorm(out, eps: eps)
        }
        if let oGateProj {
            let gate = oGateProj(x).reshaped(batch, length, numHeads, headDim)
                .transposed(0, 2, 1, 3)
            out = sigmoid(gate) * out
        }
        let merged = out.transposed(0, 2, 1, 3).reshaped(batch, length, numHeads * headDim)
        return oProj(merged)
    }
}

class SharpAIMuseSparkDsparkMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class SharpAIMuseSparkDsparkDecoderLayer: Module {
    let residualDType: DType
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMuseSparkDsparkAttention
    @ModuleInfo(key: "pre_ffn_norm") var preFFNNorm: SharpAIMuseSparkDsparkRMSNorm
    @ModuleInfo(key: "feed_forward") var feedForward: SharpAIMuseSparkDsparkMLP
    @ModuleInfo(key: "post_attn_norm") var postAttnNorm: SharpAIMuseSparkDsparkRMSNorm
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: SharpAIMuseSparkDsparkRMSNorm
    @ModuleInfo(key: "add_attn") var addAttn: SharpAIMuseSparkDsparkResidual
    @ModuleInfo(key: "add_ffn") var addFFN: SharpAIMuseSparkDsparkResidual

    init(_ config: SharpAIMuseSparkDsparkConfiguration, layerIndex: Int) {
        residualDType = config.residualDtype == "fp32" ? .float32 : .bfloat16

        // Post-norm gain centers use the residual index, as in vLLM. Validation rejects unknown types.
        let attnGain =
            museGainCenter(
                config.postNormGainCenterType, residualId: 2 * layerIndex + 1,
                numLayers: config.numHiddenLayers) ?? 1
        let ffnGain =
            museGainCenter(
                config.postNormGainCenterType, residualId: 2 * layerIndex + 2,
                numLayers: config.numHiddenLayers) ?? 1

        self._selfAttn.wrappedValue = SharpAIMuseSparkDsparkAttention(
            config, layerIndex: layerIndex)
        self._preFFNNorm.wrappedValue = SharpAIMuseSparkDsparkRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps,
            hasWeight: config.normAffine, zeroCenteredGamma: config.normZeroCenteredGamma)
        self._feedForward.wrappedValue = SharpAIMuseSparkDsparkMLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSizeMlp)
        self._postAttnNorm.wrappedValue = SharpAIMuseSparkDsparkRMSNorm(
            dimensions: config.hiddenSize, eps: config.postNormEps,
            hasWeight: config.postNormAffine, zeroCenteredGamma: config.normZeroCenteredGamma,
            gainCenter: attnGain)
        let learnableScales = config.useFfnLearnableScales
        self._postFFNNorm.wrappedValue = SharpAIMuseSparkDsparkRMSNorm(
            dimensions: config.hiddenSize, eps: config.postNormEps,
            hasWeight: learnableScales ? true : config.postNormAffine,
            zeroCenteredGamma: config.normZeroCenteredGamma, gainCenter: ffnGain,
            weightFirst: learnableScales)
        self._addAttn.wrappedValue = SharpAIMuseSparkDsparkResidual(
            dimensions: config.hiddenSize, useGate: config.useResidualGate)
        self._addFFN.wrappedValue = SharpAIMuseSparkDsparkResidual(
            dimensions: config.hiddenSize, useGate: config.useResidualGate)
    }

    func callAsFunction(
        _ hidden: MLXArray, context: MLXArray?, contextOffset: Int, draftOffset: Int
    ) -> MLXArray {
        var stream = hidden.asType(residualDType)
        let attn = selfAttn(
            selfAttn.preQKVProjNorm(stream), context: context,
            contextOffset: contextOffset, draftOffset: draftOffset)
        stream = addAttn(stream: stream, branch: postAttnNorm(attn))
        let ffn = feedForward(preFFNNorm(stream))
        return addFFN(stream: stream, branch: postFFNNorm(ffn))
    }
}

class SharpAIMuseSparkDsparkMarkovHead: Module {
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

class SharpAIMuseSparkDsparkConfidenceHead: Module {
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

class SharpAIMuseSparkDsparkBackbone: Module {
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: SharpAIMuseSparkDsparkRMSNorm
    var layers: [SharpAIMuseSparkDsparkDecoderLayer]
    @ModuleInfo(key: "norm") var norm: SharpAIMuseSparkDsparkRMSNorm
    @ModuleInfo(key: "markov_head") var markovHead: SharpAIMuseSparkDsparkMarkovHead
    @ModuleInfo(key: "confidence_head") var confidenceHead: SharpAIMuseSparkDsparkConfidenceHead?

    init(_ config: SharpAIMuseSparkDsparkConfiguration) {
        self._fc.wrappedValue = Linear(
            config.targetHiddenSize * config.targetLayerIds.count, config.hiddenSize,
            bias: false)
        self._hiddenNorm.wrappedValue = SharpAIMuseSparkDsparkRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps,
            hasWeight: config.normAffine, zeroCenteredGamma: config.normZeroCenteredGamma)
        self.layers = (0 ..< config.numHiddenLayers).map { index in
            SharpAIMuseSparkDsparkDecoderLayer(config, layerIndex: index)
        }
        self._norm.wrappedValue = SharpAIMuseSparkDsparkRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps,
            hasWeight: config.normAffine, zeroCenteredGamma: config.normZeroCenteredGamma)
        self._markovHead.wrappedValue = SharpAIMuseSparkDsparkMarkovHead(
            vocabSize: config.vocabSize, draftVocabSize: config.draftVocabSize,
            markovRank: config.markovRank)
        if config.enableConfidenceHead {
            let inputDim =
                config.hiddenSize + (config.confidenceHeadWithMarkov ? config.markovRank : 0)
            self._confidenceHead.wrappedValue = SharpAIMuseSparkDsparkConfidenceHead(
                inputDim: inputDim, withMarkov: config.confidenceHeadWithMarkov)
        } else {
            self._confidenceHead.wrappedValue = nil
        }
    }

    /// Target hidden states [B, Lc, targetHidden * numTargetLayers] to context [B, Lc, hidden].
    func combineHiddenStates(_ targetHidden: MLXArray) -> MLXArray {
        hiddenNorm(fc(targetHidden))
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray, targetHidden: MLXArray?, contextOffset: Int, draftOffset: Int
    ) -> MLXArray {
        let context = targetHidden.map { combineHiddenStates($0) }
        var h = inputsEmbeds
        for layer in layers {
            h = layer(h, context: context, contextOffset: contextOffset, draftOffset: draftOffset)
        }
        return norm(h)
    }
}

class SharpAIMuseSparkDsparkDraftModel: Module {
    let configuration: SharpAIMuseSparkDsparkConfiguration
    let outputSoftCapTemp: Float?
    let outputMultiplier: Float
    @ModuleInfo(key: "model") var model: SharpAIMuseSparkDsparkBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ParameterInfo(key: "draft_id_to_target_id") var draftIdToTargetId: MLXArray

    init(_ configuration: SharpAIMuseSparkDsparkConfiguration) {
        self.configuration = configuration
        self.outputSoftCapTemp = configuration.outputSoftCapTemp
        self.outputMultiplier = configuration.outputMultiplier
        self._model.wrappedValue = SharpAIMuseSparkDsparkBackbone(configuration)
        self._lmHead.wrappedValue = Linear(
            configuration.hiddenSize, configuration.draftVocabSize, bias: false)
        self._draftIdToTargetId.wrappedValue =
            MLXArray.zeros([configuration.draftVocabSize]).asType(.int32)
    }

    /// Draft block states [B, L, hidden]. Offsets give the absolute position of each block.
    func callAsFunction(
        _ inputsEmbeds: MLXArray, targetHidden: MLXArray? = nil,
        contextOffset: Int = 0, draftOffset: Int = 0
    ) -> MLXArray {
        model(
            inputsEmbeds, targetHidden: targetHidden,
            contextOffset: contextOffset, draftOffset: draftOffset)
    }

    /// Draft-vocab logits [B, L, draftVocab] from block states, before the output transform.
    func draftLogits(_ hidden: MLXArray) -> MLXArray {
        lmHead(hidden)
    }

    /// Scales by the Metap multiplier, then applies the soft cap when it is set.
    func applyOutputLogitTransform(_ logits: MLXArray) -> MLXArray {
        var x = logits.asType(.float32) * outputMultiplier
        if let temp = outputSoftCapTemp {
            x = temp * tanh(x / temp)
        }
        return x
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
