// SharpAI fork overlay: nemotron_parse vision encoder (C-RADIOv2-H ViT with CPE) and neck.
// Not part of upstream mlx-swift-lm.
//
// Reference (Verified against):
//   - https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1/resolve/main/hf_nemotron_parse_modeling.py
//     (`RadioWithNeck`: conv1 -> layer_norm1 -> 2D reshape -> conv2 (1x4, stride 1x4) -> layer_norm2,
//     summary -> sum_proj -> layer_norm3, summary appended as the last token)
//   - https://huggingface.co/nvidia/C-RADIOv2-H/resolve/main/radio_model.py (`_extract_final`,
//     `create_model_from_args`)
//   - https://huggingface.co/nvidia/C-RADIOv2-H/resolve/main/vit_patch_generator.py (`ViTPatchGenerator`:
//     channel-major patch flattening, learned position table, align-corners bilinear resize, window crop)
//   - https://huggingface.co/nvidia/C-RADIOv2-H/resolve/main/enable_cpe_support.py (`_forward_cpe`:
//     patch generator -> blocks -> norm, where `norm` is Identity for this checkpoint)
//   - https://huggingface.co/nvidia/C-RADIOv2-H/resolve/main/cls_token.py (class and register tokens)
//   - LayerNorm eps 1e-6 for the ViT blocks: timm `LayerNorm` default, and vLLM `radio.py` layer_norm_eps default.
//
// Limits (inference only):
//   - No training-time position jitter (`pos_dropout`, random CPE grid sampling). The eval branch of
//     `_get_pos_embeddings` is ported.
//   - No input conditioner. The checkpoint sets `image_processor_normalizes: true`, so the caller
//     normalizes pixels before this module and `norm_mean`/`norm_std` are dropped by sanitize.
//   - Tensors are NCHW for pixels and NLC for tokens, matching the reference.

import Foundation
import MLX
import MLXNN

enum SharpAINemotronParseVision {

    // MARK: - Helpers

    /// Dense matrix `[outputSize, inputSize]` of align-corners bilinear weights. Applying it along
    /// one axis matches `F.interpolate(..., mode="bilinear", align_corners=True)`.
    static func alignCornersMatrix(outputSize: Int, inputSize: Int) -> MLXArray {
        var weights = [Float](repeating: 0, count: outputSize * inputSize)
        for out in 0 ..< outputSize {
            let source =
                outputSize > 1
                ? Double(out) * Double(inputSize - 1) / Double(outputSize - 1) : 0
            let low = min(Int(source.rounded(.down)), inputSize - 1)
            let high = min(low + 1, inputSize - 1)
            let frac = Float(source - Double(low))
            weights[out * inputSize + low] += 1 - frac
            weights[out * inputSize + high] += frac
        }
        return MLXArray(weights, [outputSize, inputSize])
    }

    // MARK: - ViT pieces (timm layout)

    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        init(dim: Int, heads: Int) {
            self.heads = heads
            self.headDim = dim / heads
            self.scale = pow(Float(dim / heads), -0.5)
            _qkv.wrappedValue = Linear(dim, 3 * dim, bias: true)
            _proj.wrappedValue = Linear(dim, dim, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))
            // timm: qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4), with q, k, v in that order.
            let parts = qkv(x).split(parts: 3, axis: -1)
            let queries = parts[0].reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let keys = parts[1].reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let values = parts[2].reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)
            return proj(output)
        }
    }

    final class MLP: Module {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(dim: Int, hiddenDim: Int) {
            _fc1.wrappedValue = Linear(dim, hiddenDim, bias: true)
            _fc2.wrappedValue = Linear(hiddenDim, dim, bias: true)
        }

        /// timm's default activation is `nn.GELU` (exact erf form).
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(gelu(fc1(x)))
        }
    }

    final class Block: Module {
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo var norm2: LayerNorm
        @ModuleInfo var mlp: MLP

        init(dim: Int, heads: Int, mlpDim: Int) {
            _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
            _attn.wrappedValue = Attention(dim: dim, heads: heads)
            _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
            _mlp.wrappedValue = MLP(dim: dim, hiddenDim: mlpDim)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = x + attn(norm1(x))
            h = h + mlp(norm2(h))
            return h
        }
    }

    /// `cls_token.token`: `[numClsTokens + numRegisters, dim]`.
    final class ClsTokens: Module {
        @ParameterInfo(key: "token") var token: MLXArray

        init(count: Int, dim: Int) {
            self._token.wrappedValue = MLXRandom.normal([count, dim]) * pow(Float(dim), -0.5)
        }
    }

    /// `patch_generator`: patch embed, learned position table, and class/register tokens.
    final class PatchGenerator: Module {
        let patchSize: Int
        let gridSide: Int
        let numSkipTokens: Int

        @ModuleInfo var embedder: Linear
        @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
        @ModuleInfo(key: "cls_token") var clsToken: ClsTokens

        init(_ shape: SharpAINemotronParseEncoderShape) {
            self.patchSize = shape.patchSize
            self.gridSide = shape.cpeGridSide
            self.numSkipTokens = shape.numSkipTokens
            _embedder.wrappedValue = Linear(3 * shape.patchSize * shape.patchSize, shape.hiddenSize, bias: false)
            let positions = shape.cpeGridSide * shape.cpeGridSide
            self._posEmbed.wrappedValue =
                MLXRandom.normal([1, positions, shape.hiddenSize]) * pow(Float(shape.hiddenSize), -0.5)
            _clsToken.wrappedValue = ClsTokens(count: shape.numSkipTokens, dim: shape.hiddenSize)
        }

        /// Channel-major patch flattening: `b c (py yy) (px xx) -> b (py px) (c yy xx)`.
        func patchify(_ pixels: MLXArray) -> MLXArray {
            let (batch, channels) = (pixels.dim(0), pixels.dim(1))
            let py = pixels.dim(2) / patchSize
            let px = pixels.dim(3) / patchSize
            return pixels
                .reshaped(batch, channels, py, patchSize, px, patchSize)
                .transposed(0, 2, 4, 1, 3, 5)
                .reshaped(batch, py * px, channels * patchSize * patchSize)
        }

        /// `ViTPatchGenerator._get_pos_embeddings` in eval mode, returned as `[py * px, dim]`.
        /// Exact grid reuse when the input equals the table. Otherwise the table is resized to
        /// `M x M` (M = max(py, px)) with align-corners bilinear, then cropped to `py x px`.
        func positionEmbedding(rows py: Int, cols px: Int) throws -> MLXArray {
            let dim = posEmbed.dim(2)
            if py == gridSide && px == gridSide {
                return posEmbed.reshaped(py * px, dim)
            }
            guard py <= gridSide, px <= gridSide else {
                throw SharpAINemotronParseError.cpeGridTooSmall(
                    rows: py, cols: px, maxRows: gridSide, maxCols: gridSide)
            }
            let side = max(py, px)
            // [dim, rows, cols] layout for the separable resize.
            let grid = posEmbed.reshaped(gridSide, gridSide, dim).transposed(2, 0, 1)
            let weights = SharpAINemotronParseVision.alignCornersMatrix(
                outputSize: side, inputSize: gridSide)
            var resized = matmul(weights, grid)  // [dim, side, gridSide]
            resized = matmul(resized, weights.transposed())  // [dim, side, side]
            let cropped = resized[0..., 0 ..< py, 0 ..< px]  // [dim, py, px]
            return cropped.transposed(1, 2, 0).reshaped(py * px, dim)
        }

        /// Returns `[batch, numSkipTokens + py * px, dim]` (class and register tokens first).
        func callAsFunction(_ pixels: MLXArray) throws -> MLXArray {
            let batch = pixels.dim(0)
            let py = pixels.dim(2) / patchSize
            let px = pixels.dim(3) / patchSize
            var patches = embedder(patchify(pixels))
            patches = patches + (try positionEmbedding(rows: py, cols: px)).expandedDimensions(axis: 0)
            let cls = broadcast(
                clsToken.token.expandedDimensions(axis: 0),
                to: [batch, numSkipTokens, patches.dim(2)])
            return concatenated([cls, patches], axis: 1)
        }
    }

    /// `radio_model.model`: the ViT with the patch generator. The final norm is Identity.
    final class VisionTransformer: Module {
        @ModuleInfo(key: "patch_generator") var patchGenerator: PatchGenerator
        let blocks: [Block]

        init(_ shape: SharpAINemotronParseEncoderShape) {
            _patchGenerator.wrappedValue = PatchGenerator(shape)
            self.blocks = (0 ..< shape.layers).map { _ in
                Block(dim: shape.hiddenSize, heads: shape.attentionHeads, mlpDim: shape.mlpDim)
            }
        }

        func callAsFunction(_ pixels: MLXArray) throws -> MLXArray {
            var h = try patchGenerator(pixels)
            for block in blocks {
                h = block(h)
            }
            return h
        }
    }

    /// `encoder.model_encoder.radio_model`.
    final class RadioBackbone: Module {
        @ModuleInfo var model: VisionTransformer

        init(_ shape: SharpAINemotronParseEncoderShape) {
            _model.wrappedValue = VisionTransformer(shape)
        }
    }

    /// `encoder.model_encoder` (RADIO wrapper).
    final class RadioEncoder: Module {
        @ModuleInfo(key: "radio_model") var radioModel: RadioBackbone

        init(_ shape: SharpAINemotronParseEncoderShape) {
            _radioModel.wrappedValue = RadioBackbone(shape)
        }
    }

    // MARK: - Neck (`RadioWithNeck`)

    /// Returns `[batch, tokens + 1, width]`: the pooled feature grid followed by the summary token.
    final class Neck: Module {
        let shape: SharpAINemotronParseEncoderShape
        let width: Int

        @ModuleInfo(key: "model_encoder") var modelEncoder: RadioEncoder
        @ModuleInfo var conv1: Conv1d
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var conv2: Conv2d
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
        @ModuleInfo(key: "sum_proj") var sumProj: Linear
        @ModuleInfo(key: "layer_norm3") var layerNorm3: LayerNorm

        /// `width` is the decoder d_model. The reference hard-codes 1024, which equals this
        /// checkpoint's d_model. Tying the two keeps reduced-size test configs consistent.
        init(_ shape: SharpAINemotronParseEncoderShape, width: Int) {
            self.shape = shape
            self.width = width
            _modelEncoder.wrappedValue = RadioEncoder(shape)
            _conv1.wrappedValue = Conv1d(
                inputChannels: shape.hiddenSize, outputChannels: width, kernelSize: 1, bias: true)
            _layerNorm1.wrappedValue = LayerNorm(dimensions: width, eps: 1e-6)
            _conv2.wrappedValue = Conv2d(
                inputChannels: width, outputChannels: width,
                kernelSize: IntOrPair((1, 4)), stride: IntOrPair((1, 4)), bias: false)
            _layerNorm2.wrappedValue = LayerNorm(dimensions: width, eps: 1e-6)
            _sumProj.wrappedValue = Linear(shape.summaryDim, width, bias: true)
            _layerNorm3.wrappedValue = LayerNorm(dimensions: width, eps: 1e-6)
        }

        /// `pixels` is NCHW, already normalized. Height and width must be multiples of the patch size.
        func callAsFunction(_ pixels: MLXArray) throws -> MLXArray {
            let (imageHeight, imageWidth) = (pixels.dim(2), pixels.dim(3))
            guard imageHeight % shape.patchSize == 0, imageWidth % shape.patchSize == 0 else {
                throw SharpAINemotronParseError.unsupportedResolution(
                    height: imageHeight, width: imageWidth, patchSize: shape.patchSize)
            }
            let batch = pixels.dim(0)
            let rows = imageHeight / shape.patchSize
            let cols = imageWidth / shape.patchSize

            let y = try modelEncoder.radioModel.model(pixels)
            // all_summary = y[:, :num_cls_tokens]; bb_summary = all_summary[:, summary_idxs].
            let summaries = shape.summaryIndices.map { index in
                y[0..., index ..< index + 1, 0...]
            }
            let summary = concatenated(summaries, axis: 1).reshaped(batch, shape.summaryDim)
            // all_feat = y[:, num_skip:].
            let features = y[0..., shape.numSkipTokens..., 0...]

            var tokens = conv1(features)
            tokens = layerNorm1(tokens)
            // 'b (h w) d -> b h w d' (NHWC for MLX conv), then conv2 with a 1x4 kernel and stride.
            tokens = tokens.reshaped(batch, rows, cols, width)
            tokens = conv2(tokens)
            tokens = tokens.reshaped(batch, -1, width)
            tokens = layerNorm2(tokens)

            let pooled = layerNorm3(sumProj(summary)).expandedDimensions(axis: 1)
            return concatenated([tokens, pooled], axis: 1)
        }
    }
}
