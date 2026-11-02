// SharpAI fork overlay: cosmos3_omni generation tower. Not part of upstream mlx-swift-lm.
//
// The `_moe_gen` weights of Cosmos3 are NOT a routed mixture of experts. They are the second tower
// of a Mixture-of-Transformers: every layer has a full second set of attention projections, QK
// norms, a dense SwiGLU MLP and two layernorms, plus a second final norm. Token routing is static
// by token type (understanding tokens use the first tower, generation tokens use this one).
// Verified against: https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_cosmos3.py
// (Cosmos3PackedMoTAttention, Cosmos3VLTextMoTDecoderLayer, Cosmos3OmniTransformer._run_decoder_stack).
//
// Parameter names follow the post-sanitize key contract, not the checkpoint names:
//   gen.layers.N.self_attn.{q,k,v,o}_proj.weight   <- self_attn.{add_q_proj,add_k_proj,add_v_proj,to_add_out}
//   gen.layers.N.self_attn.{q,k}_norm.weight       <- self_attn.{norm_added_q,norm_added_k}
//   gen.layers.N.mlp.{gate,up,down}_proj.weight    <- mlp_moe_gen.*
//   gen.layers.N.{input,post_attention}_layernorm  <- *_layernorm_moe_gen
//   gen.norm.weight                                <- norm_moe_gen
// when the wrapper stores the tower as the property `gen`.
//
// The two towers only meet inside attention; see SharpAICosmos3OmniJointAttention.swift.

import Foundation
import MLX
import MLXNN

/// Sizes of the generation tower. They mirror the understanding text config of the same checkpoint
/// (Cosmos3-Nano: hidden 4096, intermediate 12288, 36 layers, 32 heads, 8 KV heads, head_dim 128).
public struct SharpAICosmos3OmniGenTowerConfiguration: Sendable, Equatable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float

    public init(
        hiddenSize: Int, intermediateSize: Int, numHiddenLayers: Int, numAttentionHeads: Int,
        numKeyValueHeads: Int, headDim: Int, rmsNormEps: Float = 1e-6
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
    }
}

/// Generation-side projections of one layer. Linear layers carry no bias: the checkpoint has no
/// bias key for them and `attention_bias` is false in the Cosmos3-Nano transformer config.
public final class SharpAICosmos3OmniGenAttention: Module {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    public init(_ config: SharpAICosmos3OmniGenTowerConfiguration) {
        let dim = config.hiddenSize
        self.heads = config.numAttentionHeads
        self.kvHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    /// Projects and QK-normalizes (per head, over head_dim). Returns q [B,H,N,D], k and v [B,Hkv,N,D];
    /// rotary is applied by the caller. V is not normalized.
    func project(_ x: MLXArray) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let (batch, length) = (x.dim(0), x.dim(1))
        var queries = wq(x).reshaped(batch, length, heads, headDim)
        var keys = wk(x).reshaped(batch, length, kvHeads, headDim)
        let values = wv(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys).transposed(0, 2, 1, 3)
        return (queries, keys, values)
    }
}

/// Dense SwiGLU MLP of the generation tower (`silu(gate(x)) * up(x)` then `down`).
public final class SharpAICosmos3OmniGenMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

/// One generation layer. It cannot run alone: attention needs the understanding tower's keys and
/// values, so the layer is driven by `SharpAICosmos3OmniGenTower.jointForward`.
public final class SharpAICosmos3OmniGenLayer: Module {

    @ModuleInfo(key: "self_attn") var attention: SharpAICosmos3OmniGenAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAICosmos3OmniGenMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ config: SharpAICosmos3OmniGenTowerConfiguration) {
        _attention.wrappedValue = SharpAICosmos3OmniGenAttention(config)
        _mlp.wrappedValue = SharpAICosmos3OmniGenMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }
}

/// The generation tower: `layers` and the final `norm`.
public final class SharpAICosmos3OmniGenTower: Module {

    @ModuleInfo(key: "layers") var layers: [SharpAICosmos3OmniGenLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public let configuration: SharpAICosmos3OmniGenTowerConfiguration

    public init(_ config: SharpAICosmos3OmniGenTowerConfiguration) {
        self.configuration = config
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            SharpAICosmos3OmniGenLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// Runs the whole two-tower stack.
    ///
    /// - Parameters:
    ///   - undLayers: the understanding decoder layers, one per generation layer (see
    ///     `SharpAICosmos3OmniJointAttention.understandingLayers(of:)`).
    ///   - undSeq: understanding hidden states [B, Nu, H] (text embeddings).
    ///   - genSeq: generation hidden states [B, Ng, H] (projected latents plus timestep embeddings).
    ///   - undPositions, genPositions: mRoPE position ids [3, B, N] for each tower.
    /// - Returns: the understanding hidden states BEFORE the final understanding norm (the caller
    ///   owns that norm and `lm_head`) and the generation hidden states AFTER `gen.norm`.
    func jointForward(
        undLayers: [Qwen3VLLanguage.DecoderLayer],
        undSeq: MLXArray,
        genSeq: MLXArray,
        undPositions: MLXArray,
        genPositions: MLXArray
    ) -> (und: MLXArray, gen: MLXArray) {
        precondition(
            undLayers.count == layers.count,
            "Cosmos3Omni: \(undLayers.count) understanding layers vs \(layers.count) generation layers")
        var und = undSeq
        var gen = genSeq
        for (undLayer, genLayer) in zip(undLayers, layers) {
            (und, gen) = SharpAICosmos3OmniJointAttention.layerForward(
                undLayer: undLayer, genLayer: genLayer, und: und, gen: gen,
                undPositions: undPositions, genPositions: genPositions)
        }
        return (und, norm(gen))
    }

    /// Convenience over `jointForward(undLayers:...)` that reads the layers out of an upstream Qwen3VL.
    func jointForward(
        understanding model: Qwen3VL,
        undSeq: MLXArray,
        genSeq: MLXArray,
        undPositions: MLXArray,
        genPositions: MLXArray
    ) -> (und: MLXArray, gen: MLXArray) {
        jointForward(
            undLayers: SharpAICosmos3OmniJointAttention.understandingLayers(of: model),
            undSeq: undSeq, genSeq: genSeq,
            undPositions: undPositions, genPositions: genPositions)
    }
}
