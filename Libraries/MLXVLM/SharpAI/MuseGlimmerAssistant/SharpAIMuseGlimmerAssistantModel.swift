// SharpAI fork overlay: muse_glimmer_assistant draft backbone (DFlash-style block drafter).
// Not part of upstream mlx-swift-lm.
//
// Port of transformers `modeling_muse_glimmer_assistant.py` (src/transformers/models/
// muse_glimmer_assistant, generated from modular_muse_glimmer_assistant.py). Weight keys were
// checked against the meta-models/Muse-Glimmer-30B-assistant safetensors header: encoder.fc,
// encoder.output_norm_enc, layers.N.{input_layernorm, post_attention_layernorm, self_attn.*
// (q/k/v/o_proj, q_norm, k_norm), mlp.*}, and norm. The checkpoint has no embed_tokens, no
// lm_head and no rotary buffers. The target supplies embeddings and the output head.
//
// Forward semantics:
// - Context hidden states are projected by encoder.fc and encoder.output_norm_enc. Every layer
//   reads the same projected context. The context is not normed per layer.
// - Attention is bidirectional. Queries come from the block (noise) states only. Keys and values
//   come from [context, block], so the key length is C + L.
// - Q and K get learned RMS norms over head_dim (q_norm, k_norm) before RoPE. RoPE is the default
//   rotate-half type with rope_theta, applied over the full head_dim. Query i sits at absolute
//   position C + i, and key j at position j.
// - Full layers apply no mask. Sliding layers keep keys with |q - k| <= sliding_window, as
//   transformers' create_bidirectional_sliding_window_mask does.
//
// UNVERIFIED: the sliding-window bound is taken from transformers' mask overlay. Its query index
// is inferred to be C + i, the absolute position, because the reference's cached path shifts the
// query offset by the previous accepted tokens. This matches the reference only when the window
// is smaller than the sequence. The hidden activation is assumed to be SiLU, as in the fixture's
// hidden_act. The config does not parse hidden_act.
// UNVERIFIED: this has not been compiled or run against a checkpoint. No reference output exists
// in this repo.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// RMS norm as in the reference. Variance is computed in fp32, the result is cast back to the
/// input dtype, and the learned weight multiplies afterwards.
final class SharpAIMuseGlimmerAssistantRMSNorm: Module {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let inputDtype = x.dtype
        let y = x.asType(.float32)
        let variance = sum(y * y, axes: [-1], keepDims: true) / Float(x.dim(-1))
        let normed = y * rsqrt(variance + eps)
        return weight * normed.asType(inputDtype)
    }
}

/// Cos and sin tables [count, headDim] for absolute positions in `positions`. Layout follows
/// the reference: freqs are concatenated with themselves along the last axis.
fileprivate func museAssistantRotaryTables(
    positions: Range<Int>, headDim: Int, base: Float
) -> (MLXArray, MLXArray) {
    let half = headDim / 2
    let invFreq: [Float] = (0 ..< half).map { index in
        1 / powf(base, Float(2 * index) / Float(headDim))
    }
    var cosValues: [Float] = []
    var sinValues: [Float] = []
    cosValues.reserveCapacity(positions.count * headDim)
    sinValues.reserveCapacity(positions.count * headDim)
    for position in positions {
        let angles = invFreq.map { Float(position) * $0 }
        let emb = angles + angles
        cosValues += emb.map { cosf($0) }
        sinValues += emb.map { sinf($0) }
    }
    let count = positions.count
    return (
        MLXArray(cosValues).reshaped(count, headDim),
        MLXArray(sinValues).reshaped(count, headDim)
    )
}

/// rotate_half from the reference: [-x2, x1] over the last axis.
fileprivate func museAssistantRotateHalf(_ x: MLXArray) -> MLXArray {
    let parts = split(x, indices: [x.dim(-1) / 2], axis: -1)
    return concatenated([-parts[1], parts[0]], axis: -1)
}

/// Additive bidirectional sliding-window mask [1, 1, L, Lk]. Query i sits at absolute position
/// `queryStart + i`. Key j is visible when |query - j| <= window.
fileprivate func museAssistantSlidingMask(
    queryStart: Int, length: Int, keyCount: Int, window: Int, dtype: DType
) -> MLXArray {
    var values: [Float] = []
    values.reserveCapacity(length * keyCount)
    for i in 0 ..< length {
        let query = queryStart + i
        for key in 0 ..< keyCount {
            values.append(abs(query - key) <= window ? 0 : -1e9)
        }
    }
    return MLXArray(values).reshaped(1, 1, length, keyCount).asType(dtype)
}

final class SharpAIMuseGlimmerAssistantAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    /// nil for full_attention layers.
    let slidingWindow: Int?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIMuseGlimmerAssistantRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIMuseGlimmerAssistantRMSNorm

    init(_ config: SharpAIMuseGlimmerAssistantConfiguration, layerIndex: Int) {
        numHeads = config.attentionHeads
        numKVHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)

        if config.layerTypes[layerIndex] == "sliding_attention" {
            guard let window = config.slidingWindow, window > 0 else {
                preconditionFailure("muse_glimmer_assistant: sliding layers need sliding_window")
            }
            slidingWindow = window
        } else {
            slidingWindow = nil
        }

        let qDim = config.attentionHeads * config.headDim
        let kvDim = config.kvHeads * config.headDim
        self._qProj.wrappedValue = Linear(config.hiddenSize, qDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        self._oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = SharpAIMuseGlimmerAssistantRMSNorm(
            dimensions: config.headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = SharpAIMuseGlimmerAssistantRMSNorm(
            dimensions: config.headDim, eps: config.rmsNormEps)
    }

    /// `hidden` is the normed block stream [B, L, H]. `context` is the projected target context
    /// [B, C, H]. `ropeCos` and `ropeSin` cover all C + L positions, shape [C + L, headDim].
    func callAsFunction(
        _ hidden: MLXArray, context: MLXArray, ropeCos: MLXArray, ropeSin: MLXArray
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let contextLength = context.dim(1)
        let total = contextLength + length

        let keyStates = concatenated([context, hidden], axis: 1)
        var q = qProj(hidden).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(keyStates).reshaped(batch, total, numKVHeads, headDim)
            .transposed(0, 2, 1, 3)
        var v = vProj(keyStates).reshaped(batch, total, numKVHeads, headDim)
            .transposed(0, 2, 1, 3)

        q = qNorm(q)
        k = kNorm(k)

        // Queries take the last L positions. Keys take all C + L positions.
        let qCos = ropeCos[contextLength ..< total].asType(q.dtype)
        let qSin = ropeSin[contextLength ..< total].asType(q.dtype)
        let kCos = ropeCos.asType(k.dtype)
        let kSin = ropeSin.asType(k.dtype)
        q = q * qCos + museAssistantRotateHalf(q) * qSin
        k = k * kCos + museAssistantRotateHalf(k) * kSin

        if numHeads != numKVHeads {
            let repeats = numHeads / numKVHeads
            k = repeated(k, count: repeats, axis: 1)
            v = repeated(v, count: repeats, axis: 1)
        }

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let slidingWindow {
            mask = .array(
                museAssistantSlidingMask(
                    queryStart: contextLength, length: length, keyCount: total,
                    window: slidingWindow, dtype: q.dtype))
        } else {
            mask = .none
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, numHeads * headDim))
    }
}

final class SharpAIMuseGlimmerAssistantMLP: Module {
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

final class SharpAIMuseGlimmerAssistantDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMuseGlimmerAssistantAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIMuseGlimmerAssistantMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: SharpAIMuseGlimmerAssistantRMSNorm
    @ModuleInfo(key: "post_attention_layernorm")
    var postAttentionLayernorm: SharpAIMuseGlimmerAssistantRMSNorm

    init(_ config: SharpAIMuseGlimmerAssistantConfiguration, layerIndex: Int) {
        self._selfAttn.wrappedValue = SharpAIMuseGlimmerAssistantAttention(
            config, layerIndex: layerIndex)
        self._mlp.wrappedValue = SharpAIMuseGlimmerAssistantMLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        self._inputLayernorm.wrappedValue = SharpAIMuseGlimmerAssistantRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = SharpAIMuseGlimmerAssistantRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ hidden: MLXArray, context: MLXArray, ropeCos: MLXArray, ropeSin: MLXArray
    ) -> MLXArray {
        var h =
            hidden
            + selfAttn(
                inputLayernorm(hidden), context: context, ropeCos: ropeCos, ropeSin: ropeSin)
        h = h + mlp(postAttentionLayernorm(h))
        return h
    }
}

/// `encoder` in the checkpoint. Projects the concatenated target hidden states to hidden_size.
final class SharpAIMuseGlimmerAssistantContextProjection: Module {
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "output_norm_enc") var outputNormEnc: SharpAIMuseGlimmerAssistantRMSNorm

    init(inputDimensions: Int, hiddenSize: Int, eps: Float) {
        self._fc.wrappedValue = Linear(inputDimensions, hiddenSize, bias: false)
        self._outputNormEnc.wrappedValue = SharpAIMuseGlimmerAssistantRMSNorm(
            dimensions: hiddenSize, eps: eps)
    }

    func callAsFunction(_ contextHiddenStates: MLXArray) -> MLXArray {
        outputNormEnc(fc(contextHiddenStates))
    }
}

/// The drafter backbone. It owns no embedding and no lm_head. The caller supplies noise
/// embeddings from the target and reads logits through the target head.
final class SharpAIMuseGlimmerAssistantModel: Module {
    let headDim: Int
    let ropeTheta: Float

    @ModuleInfo(key: "encoder") var encoder: SharpAIMuseGlimmerAssistantContextProjection
    var layers: [SharpAIMuseGlimmerAssistantDecoderLayer]
    @ModuleInfo(key: "norm") var norm: SharpAIMuseGlimmerAssistantRMSNorm

    init(_ config: SharpAIMuseGlimmerAssistantConfiguration) {
        precondition(
            config.layerTypes.count == config.hiddenLayers,
            "muse_glimmer_assistant: layer_types must have one entry per layer")
        headDim = config.headDim
        ropeTheta = config.ropeTheta

        self._encoder.wrappedValue = SharpAIMuseGlimmerAssistantContextProjection(
            inputDimensions: config.targetLayerIds.count * config.hiddenSize,
            hiddenSize: config.hiddenSize, eps: config.rmsNormEps)
        self.layers = (0 ..< config.hiddenLayers).map { index in
            SharpAIMuseGlimmerAssistantDecoderLayer(config, layerIndex: index)
        }
        self._norm.wrappedValue = SharpAIMuseGlimmerAssistantRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// `noiseEmbeds` is [B, L, H], the block input. `contextHiddenStates` is
    /// [B, C, numTargetLayers * H], the target hidden states concatenated on the last axis.
    /// `positionOffset` is the absolute position of the first context token.
    /// Returns [B, L, H] block states after the final norm.
    func callAsFunction(
        _ noiseEmbeds: MLXArray, contextHiddenStates: MLXArray, positionOffset: Int = 0
    ) -> MLXArray {
        let context = encoder(contextHiddenStates)
        let total = context.dim(1) + noiseEmbeds.dim(1)
        let (ropeCos, ropeSin) = museAssistantRotaryTables(
            positions: positionOffset ..< (positionOffset + total), headDim: headDim,
            base: ropeTheta)

        var h = noiseEmbeds
        for layer in layers {
            h = layer(h, context: context, ropeCos: ropeCos, ropeSin: ropeSin)
        }
        return norm(h)
    }
}
