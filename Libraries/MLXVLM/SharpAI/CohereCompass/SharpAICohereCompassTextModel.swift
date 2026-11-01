// SharpAI fork overlay: cohere_compass text decoder (language model path). Not part of upstream mlx-swift-lm.
//
// Port of the text path in transformers `modeling_cohere_compass.py` (PR #47878), cross-checked
// against vLLM `cohere_compass.py`. Facts taken from those sources:
// - LayerNorm with a learned scale and no bias, eps = layer_norm_eps.
// - Single pre-norm block with a parallel residual: x + attn(norm(x)) + mlp(norm(x)).
// - Full-attention layers have no RoPE (NoPE). Sliding-attention layers use RoPE, base from
//   rope_parameters["sliding_attention"].rope_theta, neox-style rotation.
// - Sliding-attention layers use a sliding window mask. Full-attention layers are causal.
// - Logits are multiplied by logit_scale. Embeddings are tied to the output projection.
//
// Limits: this is text-only. Plain RoPE is used. That matches M-RoPE only if, for text-only input,
// all three axes carry the same position. UNVERIFIED: the reference's position-id builder was not
// read. M-RoPE with image or video positions is not implemented. The vision tower is not
// implemented. The model is not verified against a reference numeric run.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// LayerNorm with a weight only, matching `CohereCompassLayerNorm` (no bias term).
public class SharpAICohereCompassLayerNorm: Module {

    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: nil, eps: eps)
    }
}

class SharpAICohereCompassAttention: Module {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    /// nil on full-attention layers (NoPE).
    let rope: RoPE?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAICohereCompassTextConfiguration, layerIndex: Int) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)

        let bias = config.attentionBias
        self._qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDim, bias: bias)
        self._kProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: bias)
        self._vProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: bias)
        self._oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: bias)

        if let theta = config.layerRopeTheta[layerIndex] {
            self.rope = RoPE(dimensions: config.headDim, traditional: false, base: theta)
        } else {
            self.rope = nil
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

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

        return oProj(output)
    }
}

class SharpAICohereCompassMLP: Module {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class SharpAICohereCompassDecoderLayer: Module {

    @ModuleInfo(key: "self_attn") var attention: SharpAICohereCompassAttention
    let mlp: SharpAICohereCompassMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: SharpAICohereCompassLayerNorm

    /// True when this layer uses the sliding-window mask.
    let isSliding: Bool
    let slidingWindow: Int

    init(_ config: SharpAICohereCompassTextConfiguration, layerIndex: Int) {
        self._attention.wrappedValue = SharpAICohereCompassAttention(
            config, layerIndex: layerIndex)
        self.mlp = SharpAICohereCompassMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        self._inputLayerNorm.wrappedValue = SharpAICohereCompassLayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.isSliding = config.layerTypes[layerIndex] == "sliding_attention"
        self.slidingWindow = config.slidingWindow
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // Parallel block: attention and MLP read the same normed input, and both add to the residual.
        let h = inputLayerNorm(x)
        let attnOut = attention(h, mask: mask, cache: cache)
        let ffOut = mlp(h)
        return attnOut + ffOut + x
    }
}

public class SharpAICohereCompassTextInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [SharpAICohereCompassDecoderLayer]
    let norm: SharpAICohereCompassLayerNorm

    public init(_ config: SharpAICohereCompassTextConfiguration) {
        precondition(config.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { index in
            SharpAICohereCompassDecoderLayer(config, layerIndex: index)
        }
        self.norm = SharpAICohereCompassLayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)

        for (i, layer) in layers.enumerated() {
            // Each layer builds its mask from its own cache. Sliding layers get the window.
            let mask = createAttentionMask(
                h: h, cache: cache?[i],
                windowSize: layer.isSliding ? layer.slidingWindow : nil)
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

/// Text-only cohere_compass causal LM. Weight keys: `model.embed_tokens`, `model.layers.N.*`, `model.norm`.
/// `lm_head` is tied to the embedding table, so no separate key exists.
public class SharpAICohereCompassTextModel: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: SharpAICohereCompassTextInner
    let logitScale: Float

    public init(_ config: SharpAICohereCompassTextConfiguration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self.model = SharpAICohereCompassTextInner(config)
        self.logitScale = config.logitScale
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        out = model.embedTokens.asLinear(out)
        return out * logitScale
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
