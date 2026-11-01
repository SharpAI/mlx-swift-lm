// SharpAI fork overlay tests for unlimited_ocr key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIUnlimitedOcrSanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    func testKeepsTextKeysUnchanged() {
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": scalar(),
            "model.layers.0.self_attn.q_proj.weight": scalar(),
            "lm_head.weight": scalar(),
        ]
        let sanitized = SharpAIUnlimitedOcrSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
    }

    func testKeepsVisionEncoderProjectorAndImageTokenWeights() {
        // The SAM, CLIP-L, projector and image token weights are loaded by the vision stack, so
        // none of them is dropped.
        let weights: [String: MLXArray] = [
            "model.sam_model.blocks.0.attn.qkv.weight": scalar(),
            "model.sam_model.pos_embed": scalar(),
            "model.vision_model.embeddings.class_embedding": scalar(),
            "model.vision_model.transformer.layers.0.mlp.fc1.weight": scalar(),
            "model.projector.layers.weight": scalar(),
            "model.projector.layers.bias": scalar(),
            "model.image_newline": scalar(),
            "model.view_seperator": scalar(),
            "lm_head.weight": scalar(),
        ]
        let sanitized = SharpAIUnlimitedOcrSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
    }

    func testDropsRotaryInvFreqBuffers() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": scalar(),
            "model.norm.weight": scalar(),
        ]
        let sanitized = SharpAIUnlimitedOcrSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["model.norm.weight"])
    }

    func testConvWeightsMoveToMLXLayout() {
        // PyTorch (out, in, kH, kW) becomes MLX (out, kH, kW, in) for the vision convolutions.
        let patch = zeros([1024, 3, 14, 14])
        let neck = zeros([256, 768, 1, 1])
        let weights: [String: MLXArray] = [
            "model.vision_model.embeddings.patch_embedding.weight": patch,
            "model.sam_model.neck.0.weight": neck,
        ]
        let sanitized = SharpAIUnlimitedOcrSanitizer.sanitize(weights)

        XCTAssertEqual(sanitized["model.vision_model.embeddings.patch_embedding.weight"]?.shape,
            [1024, 14, 14, 3])
        XCTAssertEqual(sanitized["model.sam_model.neck.0.weight"]?.shape, [256, 1, 1, 768])
    }

    func testDoesNotDropTextKeysThatOnlyResembleVisionPrefixes() {
        // "model.layers" must not match "model.projector" or "model.vision_model" prefixes.
        let weights: [String: MLXArray] = [
            "model.layers.0.mlp.gate.weight": scalar(),
            "model.norm.weight": scalar(),
        ]
        let sanitized = SharpAIUnlimitedOcrSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
    }
}
