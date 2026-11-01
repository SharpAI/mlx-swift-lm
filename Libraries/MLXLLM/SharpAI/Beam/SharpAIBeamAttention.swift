import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Ported from vLLM `vllm/models/beam/nvidia/attention.py` and `attention_ops.py`.

/// Depth-scaled residual coefficients `(x, r)` for one sublayer.
///
/// `L` is the 1-based layer index. The attention add is add `2L`, the MLP add is add `2L + 1`.
/// The sublayer output is scaled by `x = 1 / sqrt(n)` and the residual by `r = sqrt(n - 1) / sqrt(n)`.
func sharpAIBeamDepthScales(layerIndex: Int, isAttention: Bool) -> (x: Float, r: Float) {
    let oneBased = layerIndex + 1
    let numAdds = isAttention ? 2 * oneBased : 2 * oneBased + 1
    let denominator = Double(numAdds).squareRoot()
    let xScale = 1 / denominator
    let rScale = Double(numAdds - 1).squareRoot() / denominator
    return (Float(xScale), Float(rScale))
}

/// SSMax query scale for NoPE layers: `(1 + (position + 1) / 2048) ** 0.1`.
///
/// `queries` is `[B, H, L, D]` and `offset` is the cache length before this call.
func sharpAIBeamSSMaxScaled(_ queries: MLXArray, offset: Int) -> MLXArray {
    let length = queries.dim(2)
    let positions = (0 ..< length).map { Float(offset + $0 + 1) }
    let positionArray = MLXArray(positions, [length])
    let scale = exp(log1p(positionArray / 2048) * 0.1)
    return queries * scale.reshaped(1, 1, length, 1).asType(queries.dtype)
}

/// Beam attention.
///
/// Per-head Q/K RMSNorm runs first. Sliding-window layers apply RoPE. Full-attention layers are
/// NoPE and scale queries with SSMax. A per-head sigmoid gate scales the output before `o_proj`.
final class SharpAIBeamAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo var rope: RoPELayer

    /// Per-head output gate bias, shape `[num_heads]`.
    var gate_bias: MLXArray

    init(_ config: SharpAIBeamConfiguration, layerIndex: Int) {
        let hidden = config.hiddenSize
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let dim = config.headDim

        self.numHeads = heads
        self.numKVHeads = kvHeads
        self.headDim = dim
        self.scale = 1 / Float(dim).squareRoot()
        self.isSliding = config.layerTypes[layerIndex] == "sliding_attention"
        self.slidingWindow = config.slidingWindow

        self._qProj.wrappedValue = Linear(hidden, heads * dim, bias: false)
        self._kProj.wrappedValue = Linear(hidden, kvHeads * dim, bias: false)
        self._vProj.wrappedValue = Linear(hidden, kvHeads * dim, bias: false)
        self._oProj.wrappedValue = Linear(heads * dim, hidden, bias: false)
        self._gateProj.wrappedValue = Linear(hidden, heads * dim, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: dim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: dim, eps: config.rmsNormEps)
        // NoPE layers never apply this RoPE. It is built so the module tree is uniform.
        self.rope = initializeRope(
            dims: dim, base: config.ropeTheta, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: nil)
        self.gate_bias = zeros([heads])

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(B, L, numHeads, headDim)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim)
        let values = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let gate = gateProj(x).reshaped(B, L, numHeads, headDim)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys).transposed(0, 2, 1, 3)

        if isSliding {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        } else {
            queries = sharpAIBeamSSMaxScaled(queries, offset: cache?.offset ?? 0)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )

        let gateBias = gate_bias.reshaped(1, 1, numHeads, 1).asType(gate.dtype)
        let gated = output.transposed(0, 2, 1, 3) * sigmoid(gate + gateBias)
        return oProj(gated.reshaped(B, L, -1))
    }
}
