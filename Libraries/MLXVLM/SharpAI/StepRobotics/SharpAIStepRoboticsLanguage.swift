// SharpAI fork overlay: Step3 text decoder for StepVL. Not part of upstream mlx-swift-lm.
//
// Port of `Step3TextModel` in vLLM's `step3_text.py`. The decoder accepts input embeddings, so
// image features can be merged before the layers run. Module keys match the checkpoint after
// `SharpAIStepRoboticsSanitize`: `model.*` and `lm_head`, under `language_model`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Attention with a shared query slice and one KV head.
/// Path: `self_attn.qkv_proj` (q, k and v fused), `inter_norm`, `wq`, `o_proj`.
/// vLLM splits q from k and v, applies `inter_norm` to q, then projects q with `wq`.
final class SharpAIStepRoboticsTextAttention: Module {
    let numHeads: Int
    let headDim: Int
    let qDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "inter_norm") var interNorm: RMSNorm
    @ModuleInfo var wq: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIStepRoboticsConfiguration.TextConfiguration) {
        self.numHeads = config.attentionHeads
        self.headDim = config.headDim
        self.qDim = config.resolvedQDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.rope = initializeRope(
            dims: config.headDim, base: config.ropeTheta, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: nil)

        self._qkvProj.wrappedValue = Linear(
            config.hiddenSize, config.resolvedQDim + 2 * config.headDim, bias: false)
        self._interNorm.wrappedValue = RMSNorm(
            dimensions: config.resolvedQDim, eps: config.rmsNormEps)
        self._wq.wrappedValue = Linear(
            config.resolvedQDim, config.headDim * config.attentionHeads, bias: false)
        self._oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let qkv = qkvProj(x)

        let q = wq(interNorm(qkv[.ellipsis, 0 ..< qDim]))
        var queries = q.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = qkv[.ellipsis, qDim ..< (qDim + headDim)].reshaped(B, L, 1, headDim)
            .transposed(0, 2, 1, 3)
        let values = qkv[.ellipsis, (qDim + headDim)...].reshaped(B, L, 1, headDim)
            .transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return oProj(output)
    }
}

/// Dense SiLU MLP. Paths: `gate_proj`, `up_proj`, `down_proj`.
final class SharpAIStepRoboticsTextMLP: Module, UnaryLayer {
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

/// Routed experts plus a router. Path: `moe.gate`, `moe.switch_mlp.*`.
/// The checkpoint stores stacked expert weights `[experts, out, in]`, which `SwitchGLU` expects.
/// UNVERIFIED: softmax, then top-k, then optional renormalize. This matches the vLLM
/// `FusedMoEFactory` default (`scoring_func="softmax"`). The Step3 reference may differ.
final class SharpAIStepRoboticsMoE: Module {
    let topK: Int
    let renormalize: Bool

    @ModuleInfo var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ config: SharpAIStepRoboticsConfiguration.TextConfiguration) {
        self.topK = config.moeTopK
        self.renormalize = config.normExpertWeight
        self._gate.wrappedValue = Linear(config.hiddenSize, config.moeExperts, bias: false)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize,
            numExperts: config.moeExperts)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = gate(x)
        let probs = MLX.softmax(logits, axis: -1, precise: true)
        let inds = MLX.argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = MLX.takeAlong(probs, inds, axis: -1)
        if renormalize {
            scores = scores / MLX.sum(scores, axis: -1, keepDims: true)
        }
        return weightedExpertSum(switchMLP(x, inds), scores)
    }
}

/// Decoder layer. Layers listed in `moe_layers_enum` use `moe` plus `share_expert`.
/// The rest use a dense `mlp`.
final class SharpAIStepRoboticsDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIStepRoboticsTextAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo var moe: SharpAIStepRoboticsMoE?
    @ModuleInfo(key: "share_expert") var shareExpert: SharpAIStepRoboticsTextMLP?
    @ModuleInfo var mlp: SharpAIStepRoboticsTextMLP?

    init(_ config: SharpAIStepRoboticsConfiguration.TextConfiguration, layerIndex: Int) {
        self._selfAttn.wrappedValue = SharpAIStepRoboticsTextAttention(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.moeLayers.contains(layerIndex) {
            self._moe.wrappedValue = SharpAIStepRoboticsMoE(config)
            self._shareExpert.wrappedValue = SharpAIStepRoboticsTextMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.shareExpertDim)
        } else {
            self._mlp.wrappedValue = SharpAIStepRoboticsTextMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let normed = postAttentionLayerNorm(h)
        let ff: MLXArray
        if let moe, let shareExpert {
            ff = shareExpert(normed) + moe(normed)
        } else if let mlp {
            ff = mlp(normed)
        } else {
            preconditionFailure("step_robotics decoder layer has no feed-forward block")
        }
        return h + ff
    }
}

/// Decoder body. Module path `model.*`.
final class SharpAIStepRoboticsTextBody: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [SharpAIStepRoboticsDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: SharpAIStepRoboticsConfiguration.TextConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { index in
            SharpAIStepRoboticsDecoderLayer(config, layerIndex: index)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

/// Step3 text model with LM head. Module path `language_model.*` in the wrapper.
final class SharpAIStepRoboticsLanguageModel: Module {
    let kvHeads: [Int]

    @ModuleInfo var model: SharpAIStepRoboticsTextBody
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ config: SharpAIStepRoboticsConfiguration.TextConfiguration) {
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in 1 }
        self._model.wrappedValue = SharpAIStepRoboticsTextBody(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    func embed(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    func logits(_ hidden: MLXArray) -> MLXArray {
        lmHead(hidden)
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(model(embeddings: embed(tokens), cache: cache))
    }
}
