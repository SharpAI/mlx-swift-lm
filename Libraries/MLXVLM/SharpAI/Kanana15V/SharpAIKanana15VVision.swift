// SharpAI fork overlay: kanana-1.5-v vision encoder (`qwen2-vl-ve`). Not part of upstream mlx-swift-lm.
//
// Qwen2-VL style ViT: Conv3d patch embed, 2D rotary positions, full attention, quick GELU MLP.
// The Qwen2-VL patch merger is left out, because the C-Abstractor projector replaces it.
// UNVERIFIED: weight names (patch_embed, blocks, attn.qkv, mlp.fc1) and rotary theta 10_000 follow Qwen2-VL.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum SharpAIKanana15VRotary {
    /// Applies rotary embedding to `[L, heads, headDim]` with `freqs` shaped `[L, headDim / 2]`.
    static func apply(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cosines = cos(freqs)
        var sines = sin(freqs)

        cosines = expandedDimensions(cosines, axis: 1)
        cosines = tiled(cosines, repetitions: [1, 1, 2])
        cosines = expandedDimensions(cosines, axis: 0)

        sines = expandedDimensions(sines, axis: 1)
        sines = tiled(sines, repetitions: [1, 1, 2])
        sines = expandedDimensions(sines, axis: 0)

        let output = (tensor * cosines) + (QwenVL.rotateHalf(tensor) * sines)
        return output.asType(tensor.dtype)
    }
}

final class SharpAIKanana15VAttention: Module {

    let numHeads: Int
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    init(dims: Int, numHeads: Int) {
        self.numHeads = numHeads
        let headDim = dims / numHeads
        self.scale = pow(Float(headDim), -0.5)

        self._qkv.wrappedValue = Linear(dims, 3 * dims, bias: true)
        self._proj.wrappedValue = Linear(dims, dims)
    }

    func callAsFunction(
        _ x: MLXArray, attentionMask: MLXArray, rotaryPositionEmbedding: MLXArray
    ) -> MLXArray {
        let sequenceLength = x.dim(0)

        let packed = qkv(x)
        let parts = split(packed, parts: 3, axis: -1)
        var q = parts[0]
        var k = parts[1]
        var v = parts[2]

        q = q.reshaped(sequenceLength, numHeads, -1)
        k = k.reshaped(sequenceLength, numHeads, -1)
        v = v.reshaped(sequenceLength, numHeads, -1)

        q = SharpAIKanana15VRotary.apply(q, freqs: rotaryPositionEmbedding)
        k = SharpAIKanana15VRotary.apply(k, freqs: rotaryPositionEmbedding)

        q = q.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
        k = k.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
        v = v.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .array(attentionMask)
        )
        .transposed(0, 2, 1, 3)
        .reshaped(sequenceLength, -1)

        return proj(output)
    }
}

final class SharpAIKanana15VMLP: Module {

    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._fc1.wrappedValue = Linear(dimensions, hiddenDimensions)
        self._fc2.wrappedValue = Linear(hiddenDimensions, dimensions)
    }

    /// `geluFastApproximate` is `x * sigmoid(1.702 * x)`, the quick GELU named by `hidden_act`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluFastApproximate(fc1(x)))
    }
}

final class SharpAIKanana15VBlock: Module {

    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attention: SharpAIKanana15VAttention
    @ModuleInfo var mlp: SharpAIKanana15VMLP

    init(_ config: SharpAIKanana15VVisionConfiguration) {
        self.norm1 = LayerNorm(dimensions: config.embedDimensions, eps: 1e-6)
        self.norm2 = LayerNorm(dimensions: config.embedDimensions, eps: 1e-6)

        self._attention.wrappedValue = SharpAIKanana15VAttention(
            dims: config.embedDimensions, numHeads: config.attentionHeads)

        let mlpHiddenDimensions = Int(Float(config.embedDimensions) * config.mlpRatio)
        self.mlp = SharpAIKanana15VMLP(
            dimensions: config.embedDimensions, hiddenDimensions: mlpHiddenDimensions)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray, attentionMask: MLXArray, rotaryPositionEmbedding: MLXArray
    ) -> MLXArray {
        var h =
            hiddenStates
            + attention(
                norm1(hiddenStates), attentionMask: attentionMask,
                rotaryPositionEmbedding: rotaryPositionEmbedding)
        h = h + mlp(norm2(h))
        return h
    }
}

/// Vision encoder without the merger. Output is `[patches, embedDimensions]`.
final class SharpAIKanana15VVisionModel: Module {

    @ModuleInfo(key: "patch_embed") var patchEmbed: QwenVL.PatchEmbed
    @ModuleInfo(key: "rotary_pos_emb") var rotaryPositionEmbedding: QwenVL.VisionRotaryEmbedding
    @ModuleInfo(key: "blocks") var blocks: [SharpAIKanana15VBlock]

    let spatialMergeSize: Int

    init(_ config: SharpAIKanana15VVisionConfiguration) {
        self.spatialMergeSize = config.spatialMergeSize

        self._patchEmbed.wrappedValue = QwenVL.PatchEmbed(
            patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize,
            inChannels: config.inChannels,
            embedDimensions: config.embedDimensions)

        let headDimensions = config.embedDimensions / config.attentionHeads
        self._rotaryPositionEmbedding.wrappedValue = QwenVL.VisionRotaryEmbedding(
            dimensions: headDimensions / 2, theta: 10_000)

        self._blocks.wrappedValue = (0 ..< config.depth).map { _ in
            SharpAIKanana15VBlock(config)
        }
    }

    /// Rotary angles per patch: row and column positions, each merged in `spatialMergeSize` tiles.
    func rotaryEmbedding(_ frames: [THW]) -> MLXArray {
        var positionIds = [MLXArray]()

        for frame in frames {
            let (t, h, w) = frame.values

            var hposIds = expandedDimensions(MLXArray(0 ..< h), axis: 1)
            hposIds = repeated(hposIds, count: w, axis: 1)
            hposIds =
                hposIds
                .reshaped(
                    h / spatialMergeSize, spatialMergeSize,
                    w / spatialMergeSize, spatialMergeSize
                )
                .transposed(0, 2, 1, 3)
                .flattened()

            var wposIds = expandedDimensions(MLXArray(0 ..< w), axis: 0)
            wposIds = repeated(wposIds, count: h, axis: 0)
            wposIds =
                wposIds
                .reshaped(
                    h / spatialMergeSize, spatialMergeSize,
                    w / spatialMergeSize, spatialMergeSize
                )
                .transposed(0, 2, 1, 3)
                .flattened()

            let stackedPosIds = stacked([hposIds, wposIds], axis: -1)
            positionIds.append(tiled(stackedPosIds, repetitions: [t, 1]))
        }

        let indices = concatenated(positionIds, axis: 0)
        let maxFrameSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 0
        let full = rotaryPositionEmbedding(sequenceLength: maxFrameSize)[indices]

        return full.reshaped(indices.dim(0), -1)
    }

    /// Block-diagonal mask `[1, patches, patches]`: a patch attends only within its own temporal
    /// slice, so each slice of `h * w` patches is one segment (`cu_seqlens` in the reference).
    func attentionMask(_ frames: [THW]) -> MLXArray {
        var segmentIds = [Int32]()
        var segment: Int32 = 0
        for frame in frames {
            for _ in 0 ..< frame.t {
                segmentIds.append(contentsOf: [Int32](repeating: segment, count: frame.h * frame.w))
                segment += 1
            }
        }
        let ids = MLXArray(segmentIds)
        return (ids.expandedDimensions(axis: 1) .== ids.expandedDimensions(axis: 0))
            .expandedDimensions(axis: 0)
    }

    /// `pixels` is `[patches, inChannels * temporal * patch * patch]`, one row per patch.
    func callAsFunction(_ pixels: MLXArray, frames: [THW]) -> MLXArray {
        var hiddenStates = patchEmbed(pixels)
        let rotary = rotaryEmbedding(frames)
        let mask = attentionMask(frames)

        for block in blocks {
            hiddenStates = block(hiddenStates, attentionMask: mask, rotaryPositionEmbedding: rotary)
        }

        return hiddenStates
    }
}
