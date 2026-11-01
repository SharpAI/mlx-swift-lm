// SharpAI fork overlay: patch and normalize helpers for isaac images. Not part of upstream.
//
// Only the tensor math is here. Image decoding, resizing and the chat-template image token
// expansion are not implemented.
// UNVERIFIED: the 0.5 mean and std and the (patch_h, patch_w, channel) flatten order follow
// the Siglip2 image processor. Isaac's own processor was not read.

import Foundation
import MLX

public enum SharpAIIsaacImageProcessor {

    /// Maps 0...255 pixels to -1...1, the Siglip2 normalization.
    public static func normalize(_ pixels: MLXArray) -> MLXArray {
        ((pixels / Float(255)) - Float(0.5)) / Float(0.5)
    }

    /// Splits channels-last `[N, H, W, C]` images into `[N, (H/p) * (W/p), p * p * C]` patches.
    /// Each patch flattens as (row, column, channel). H and W must be multiples of `patchSize`.
    public static func patchify(_ pixels: MLXArray, patchSize: Int) -> MLXArray {
        let n = pixels.dim(0)
        let h = pixels.dim(1)
        let w = pixels.dim(2)
        let c = pixels.dim(3)
        let gridH = h / patchSize
        let gridW = w / patchSize
        return
            pixels
            .reshaped(n, gridH, patchSize, gridW, patchSize, c)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(n, gridH * gridW, patchSize * patchSize * c)
    }
}
