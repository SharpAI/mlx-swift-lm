// SharpAI fork overlay: hyperclovax text model. Not part of upstream mlx-swift-lm.
//
// Llama-style decoder with Granite-style multipliers and optional post norms.
// UNVERIFIED: post norm key names and placement are guesses. The HF modeling file is not local.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIHyperclovaxAttention: Module {
    let args: SharpAIHyperclovaxConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ args: SharpAIHyperclovaxConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads
        let headDim = args.headDim

        self.scale = args.attentionMultiplier

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)

        self.rope = initializeRope(
            dims: headDim, base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class SharpAIHyperclovaxMLP: Module {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: SharpAIHyperclovaxConfiguration) {
        self._gate.wrappedValue = Linear(
            args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        self._down.wrappedValue = Linear(
            args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
        self._up.wrappedValue = Linear(
            args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class SharpAIHyperclovaxDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIHyperclovaxAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIHyperclovaxMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    // UNVERIFIED: key names are guesses, not checked against a checkpoint index. A strict load of
    // the 14B checkpoint may fail on missing or unexpected keys. Present only when use_post_norm is true.
    @ModuleInfo(key: "post_attn_layernorm") var postAttnLayerNorm: RMSNorm?
    @ModuleInfo(key: "post_mlp_layernorm") var postMLPLayerNorm: RMSNorm?

    let residualMultiplier: Float

    init(_ args: SharpAIHyperclovaxConfiguration) {
        self._attention.wrappedValue = SharpAIHyperclovaxAttention(args)
        self._mlp.wrappedValue = SharpAIHyperclovaxMLP(args)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        if args.usePostNorm {
            self._postAttnLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postMLPLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }
        self.residualMultiplier = args.residualMultiplier
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        if let postAttnLayerNorm {
            r = postAttnLayerNorm(r)
        }
        let h = x + r * residualMultiplier

        r = mlp(postAttentionLayerNorm(h))
        if let postMLPLayerNorm {
            r = postMLPLayerNorm(r)
        }
        return h + r * residualMultiplier
    }
}

public class SharpAIHyperclovaxModelInner: Module, LayerPartitionable {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [SharpAIHyperclovaxDecoderLayer]
    let norm: RMSNorm
    let embeddingMultiplier: Float

    // LayerPartitionable
    public var gpuLayerCount: Int?
    public var totalLayerCount: Int { layers.count }

    init(_ args: SharpAIHyperclovaxConfiguration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in
            SharpAIHyperclovaxDecoderLayer(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.embeddingMultiplier = args.embeddingMultiplier
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs) * embeddingMultiplier

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount) {
                layer(h, mask: mask, cache: cache?[i])
            }
        }

        return norm(h)
    }
}

/// Text model for the hyperclovax model type.
public class SharpAIHyperclovaxModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: SharpAIHyperclovaxModelInner
    let logitsScaling: Float
    let configuration: SharpAIHyperclovaxConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: SharpAIHyperclovaxConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = SharpAIHyperclovaxModelInner(args)
        self.logitsScaling = args.logitsScaling

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        // muP: logits are multiplied by logits_scaling (Granite divides).
        return out * MLXArray(logitsScaling)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIHyperclovaxSanitizer.sanitize(weights, configuration: configuration)
    }
}

// MARK: - LoRA

extension SharpAIHyperclovaxModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
