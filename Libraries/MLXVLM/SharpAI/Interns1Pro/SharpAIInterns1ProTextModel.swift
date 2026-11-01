// SharpAI fork overlay: interns1_pro text decoder. Not part of upstream mlx-swift-lm.
//
// Verified against: https://huggingface.co/internlm/Intern-S1-Pro/resolve/main/modeling_interns1_pro.py
// Classes ported: InternS1ProMoeTextAttention, InternS1ProMoeTextFourierEmbedding (FoPE),
// InternS1ProMoeTextGroupedRouter, InternS1ProMoeTextSparseMoeBlock, InternS1ProMoeTextDecoderLayer,
// InternS1ProTextModel, and the language_model part of InternS1ProForConditionalGeneration.
//
// Module paths mirror the checkpoint, so HF keys load without renaming:
//   model.language_model.embed_tokens / layers.N.* / norm / rotary_emb.{sin_coef,cos_coef}
//   lm_head.weight
//
// Scope and deviations:
// - Text decoder only. The vision tower (model.visual) and the time-series encoder
//   (model.time_series) are not implemented. Their keys are dropped in `sanitize`, so image and
//   time-series inputs are not handled.
// - FoPE coefficients start at identity. The reference draws Xavier noise and adds identity. The
//   checkpoint replaces the init, so this only affects a model built without weights.
// - FoPE is used when rope_scaling carries fope_* keys (the reference rule). Otherwise plain RoPE.
//   rope_type other than default, and num_inv_freq, are rejected by the config validator.
// - Attention scale is head_dim^-0.5 and attention_scaling is 1.0, as in the reference.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Rotation applied by the attention layers: cos and sin shaped [B, Hkv or 1, L, head_dim].
struct SharpAIInterns1ProRotation {
    let cos: MLXArray
    let sin: MLXArray
    /// True when the coefficients are per KV head (reference fope_sep_head).
    let sepHead: Bool
}

/// Rotate-half helper from the reference `rotate_half`.
func sharpAIInterns1ProRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

/// FoPE (reference InternS1ProMoeTextFourierEmbedding). The frequencies are the default RoPE
/// inverse frequencies, base rope_theta, length head_dim/2. The learned coefficient matrices mix
/// cos(freq) and sin(freq) into the head dimension, then the result is duplicated to head_dim.
class SharpAIInterns1ProFourierEmbedding: Module {
    let sepHead: Bool
    /// Plain Swift storage: a non-persistent buffer in the reference, so it is not a parameter.
    let invFreq: [Float]

    @ParameterInfo(key: "sin_coef") var sinCoef: MLXArray
    @ParameterInfo(key: "cos_coef") var cosCoef: MLXArray

    init(_ config: SharpAIInterns1ProTextConfiguration) {
        self.sepHead = config.fopeSepHead
        let half = config.headDim / 2
        self.invFreq = (0 ..< half).map { i in
            // Verified against: transformers modeling_rope_utils._compute_default_rope_parameters,
            // inv_freq = 1 / base^(arange(0, dim, 2) / dim), dim = head_dim.
            Float(1.0 / pow(Double(config.ropeTheta), Double(2 * i) / Double(config.headDim)))
        }
        let identity = MLXArray(
            (0 ..< half * half).map { $0 / half == $0 % half ? Float(1) : Float(0) }
        ).reshaped(half, half)
        let coef: MLXArray
        if config.fopeSepHead {
            coef = identity.expandedDimensions(axis: 0) + MLXArray.zeros([config.kvHeads, half, half])
        } else {
            coef = identity
        }
        self._sinCoef.wrappedValue = coef
        self._cosCoef.wrappedValue = coef
    }

    /// cos and sin for `length` positions. `offset` is the cache offset (scalar or per batch row).
    func rotation(length: Int, offset: RoPEOffset?, dtype: DType) -> SharpAIInterns1ProRotation {
        let base = MLXArray((0 ..< length).map { Float($0) }).reshaped(1, length)
        let positions: MLXArray
        switch offset {
        case .batch(let starts)?:
            positions = starts.asType(.float32).expandedDimensions(axis: 1) + base
        case .scalar(let start)?:
            positions = base + Float(start)
        case nil:
            positions = base
        }
        let inv = MLXArray(invFreq).reshaped(1, 1, invFreq.count)
        let freqs = positions.expandedDimensions(axis: -1) * inv  // [B, L, D/2]
        let posCos = MLX.cos(freqs)
        let posSin = MLX.sin(freqs)

        let cosOut: MLXArray
        let sinOut: MLXArray
        if sepHead {
            // einsum "bhtD,hDd->bhtd": [B,1,L,D/2] @ [Hkv,D/2,D/2] -> [B,Hkv,L,D/2]
            cosOut = posCos.expandedDimensions(axis: 1).matmul(cosCoef)
            sinOut = posSin.expandedDimensions(axis: 1).matmul(sinCoef)
        } else {
            // einsum "btD,Dd->btd", then a head axis of size 1 for broadcasting
            cosOut = posCos.matmul(cosCoef).expandedDimensions(axis: 1)
            sinOut = posSin.matmul(sinCoef).expandedDimensions(axis: 1)
        }
        // Reference pads to head_dim/2 with ones. The output already has head_dim/2 columns here,
        // because the coefficient matrices are square (num_inv_freq is rejected by the validator).
        let cosFull = concatenated([cosOut, cosOut], axis: -1)
        let sinFull = concatenated([sinOut, sinOut], axis: -1)
        return SharpAIInterns1ProRotation(
            cos: cosFull.asType(dtype), sin: sinFull.asType(dtype), sepHead: sepHead)
    }
}

/// Gated MLP (dense layers, and the non-MoE path when mlp_only_layers or sparsity says so).
class SharpAIInterns1ProMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

/// Grouped top-k router (reference InternS1ProMoeTextGroupedRouter, with router_n_groups).
/// Parameter: `weight` [num_experts, hidden_size], no bias.
class SharpAIInterns1ProRouter: Module {
    let topK: Int
    let numExperts: Int
    let nGroups: Int
    let normTopkProb: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(_ config: SharpAIInterns1ProTextConfiguration) {
        self.topK = config.numExpertsPerToken
        self.numExperts = config.numExperts
        self.nGroups = config.routerNGroups
        self.normTopkProb = config.normTopkProb
        self._weight.wrappedValue = MLXArray.zeros([config.numExperts, config.hiddenSize])
    }

    /// Returns (indices, scores), both shaped [..., topK].
    ///
    /// Reference steps: softmax over all experts in float32; split the experts into `nGroups`
    /// contiguous groups; take `topK / nGroups` experts from each group; map group-local indices
    /// to global ones; gather the probabilities; renormalise by their sum when norm_topk_prob.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let lead = Array(x.shape.dropLast())
        let flat = x.reshaped(-1, x.dim(-1))
        let logits = flat.matmul(weight.T)
        let probs = MLX.softmax(logits.asType(.float32), axis: -1)

        let groupSize = numExperts / nGroups
        let perGroup = topK / nGroups
        let grouped = probs.reshaped(-1, nGroups, groupSize)  // [N, G, S]
        let local = argPartition(-grouped, kth: perGroup - 1, axis: -1)[.ellipsis, ..<perGroup]
        let offsets = MLXArray((0 ..< nGroups).map { Int32($0 * groupSize) })
            .reshaped(1, nGroups, 1)
        let inds = (local.asType(.int32) + offsets).reshaped(-1, topK)  // [N, topK]

        var scores = takeAlong(probs, inds, axis: -1)
        if normTopkProb {
            scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (
            inds.reshaped(lead + [topK]),
            scores.asType(x.dtype).reshaped(lead + [topK])
        )
    }
}

/// Routed experts (SwitchGLU) combined with the router weights.
/// Keys: gate.weight, switch_mlp.{gate,up,down}_proj.weight (stacked by the sanitizer).
class SharpAIInterns1ProSparseMoeBlock: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: SharpAIInterns1ProRouter
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ config: SharpAIInterns1ProTextConfiguration) {
        self._gate.wrappedValue = SharpAIInterns1ProRouter(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            activation: silu
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        let y = switchMLP(x, inds)  // [B, L, topK, H]
        return (y * scores.expandedDimensions(axis: -1)).sum(axis: -2)
    }
}

/// Attention with per-head q/k RMSNorm before the rotation (reference InternS1ProMoeTextAttention).
class SharpAIInterns1ProAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    /// Plain RoPE, used only when FoPE is off. FoPE supplies the rotation otherwise.
    let rope: RoPELayer?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: SharpAIInterns1ProTextConfiguration, useFope: Bool) {
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.rope =
            useFope
            ? nil
            : initializeRope(
                dims: config.headDim,
                base: config.ropeTheta,
                traditional: false,
                scalingConfig: nil,
                maxPositionEmbeddings: nil)

        self._qProj.wrappedValue = Linear(
            config.hiddenSize, numHeads * headDim, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(
            config.hiddenSize, numKVHeads * headDim, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(
            config.hiddenSize, numKVHeads * headDim, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(
            numHeads * headDim, config.hiddenSize, bias: config.attentionBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        rotation: SharpAIInterns1ProRotation?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var q = qNorm(qProj(x).reshaped(B, L, numHeads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(B, L, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        if let rotation {
            // Verified against: apply_rotary_pos_emb_sep (sep head) and apply_rotary_pos_emb.
            // Sep head: the query cos/sin are repeated per KV group (repeat_kv order); keys use
            // their own KV-head cos/sin.
            let qCos: MLXArray
            let qSin: MLXArray
            if rotation.sepHead {
                let groups = numHeads / numKVHeads
                qCos = repeated(rotation.cos, count: groups, axis: 1)
                qSin = repeated(rotation.sin, count: groups, axis: 1)
            } else {
                qCos = rotation.cos
                qSin = rotation.sin
            }
            q = q * qCos + sharpAIInterns1ProRotateHalf(q) * qSin
            k = k * rotation.cos + sharpAIInterns1ProRotateHalf(k) * rotation.sin
        } else if let rope {
            let offset = cache?.ropeOffset
            q = applyRotaryPosition(rope, to: q, offset: offset)
            k = applyRotaryPosition(rope, to: k, offset: offset)
        }

        let output = attentionWithCacheUpdate(
            queries: q,
            keys: k,
            values: v,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

/// Pre-norm decoder layer (reference InternS1ProMoeTextDecoderLayer).
class SharpAIInterns1ProDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIInterns1ProAttention
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    fileprivate let mlp: UnaryLayer

    init(_ config: SharpAIInterns1ProTextConfiguration, layerIdx: Int, useFope: Bool) {
        self._selfAttn.wrappedValue = SharpAIInterns1ProAttention(config, useFope: useFope)
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.numExperts > 0, !config.mlpOnlyLayers.contains(layerIdx),
            (layerIdx + 1) % config.decoderSparseStep == 0
        {
            self.mlp = SharpAIInterns1ProSparseMoeBlock(config)
        } else {
            self.mlp = SharpAIInterns1ProMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        rotation: SharpAIInterns1ProRotation?
    ) -> MLXArray {
        var h = x + selfAttn(inputLayernorm(x), mask: mask, cache: cache, rotation: rotation)
        h = h + mlp(postAttentionLayernorm(h))
        return h
    }
}

/// The language model body (reference InternS1ProTextModel). Mirrors `model.language_model`.
class SharpAIInterns1ProTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    /// FoPE module. Nil when FoPE is off (plain RoPE is then used in attention).
    @ModuleInfo(key: "rotary_emb") var rotaryEmb: SharpAIInterns1ProFourierEmbedding?
    let layers: [SharpAIInterns1ProDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SharpAIInterns1ProTextConfiguration) {
        let useFope = config.useFope
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._rotaryEmb.wrappedValue =
            useFope ? SharpAIInterns1ProFourierEmbedding(config) : nil
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAIInterns1ProDecoderLayer(config, layerIdx: $0, useFope: useFope)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(x)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        var rotation: SharpAIInterns1ProRotation?
        if let rotaryEmb {
            rotation = rotaryEmb.rotation(
                length: h.dim(1), offset: cache?.first?.ropeOffset, dtype: h.dtype)
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], rotation: rotation)
        }
        return norm(h)
    }
}

/// Wrapper that mirrors the checkpoint prefix `model.`.
class SharpAIInterns1ProInner: Module {
    @ModuleInfo(key: "language_model") var languageModel: SharpAIInterns1ProTextModel

    init(_ config: SharpAIInterns1ProTextConfiguration) {
        self._languageModel.wrappedValue = SharpAIInterns1ProTextModel(config)
    }
}

/// Text-only interns1_pro language model. See the file header for the scope.
public class SharpAIInterns1ProModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int]

    let configuration: SharpAIInterns1ProTextConfiguration
    let tieWordEmbeddings: Bool

    @ModuleInfo(key: "model") var model: SharpAIInterns1ProInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ textConfiguration: SharpAIInterns1ProTextConfiguration, tieWordEmbeddings: Bool) {
        self.configuration = textConfiguration
        self.tieWordEmbeddings = tieWordEmbeddings
        self.kvHeads = Array(
            repeating: textConfiguration.kvHeads, count: textConfiguration.hiddenLayers)
        self._model.wrappedValue = SharpAIInterns1ProInner(textConfiguration)
        if tieWordEmbeddings {
            self._lmHead.wrappedValue = nil
        } else {
            self._lmHead.wrappedValue = Linear(
                textConfiguration.hiddenSize, textConfiguration.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model.languageModel(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.languageModel.embedTokens.asLinear(out)
    }

    /// Drops the vision and time-series keys (not implemented here), then applies
    /// `SharpAIInterns1ProSanitizer` (FP8 scales, rotary buffer, expert stacking).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter {
            !$0.key.hasPrefix("model.visual.") && !$0.key.hasPrefix("model.time_series.")
        }
        sanitized = SharpAIInterns1ProSanitizer.sanitize(
            sanitized,
            numExperts: configuration.numExperts > 0 ? configuration.numExperts : nil)
        if tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }
        return sanitized
    }

    public var loraLayers: [Module] {
        model.languageModel.layers
    }
}
