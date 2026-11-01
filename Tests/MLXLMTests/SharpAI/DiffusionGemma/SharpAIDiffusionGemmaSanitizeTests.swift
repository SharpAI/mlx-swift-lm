// SharpAI fork overlay tests for diffusion_gemma key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIDiffusionGemmaSanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    func testKeepsAllKeysUnchanged() {
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": scalar(),
            "model.layers.0.self_attn.q_proj.weight": scalar(),
            "lm_head.weight": scalar(),
        ]
        let sanitized = SharpAIDiffusionGemmaSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
    }

    func testEmptyInputGivesEmptyOutput() {
        XCTAssertTrue(SharpAIDiffusionGemmaSanitizer.sanitize([:]).isEmpty)
    }

    /// Key names copied verbatim from the `weight_map` of
    /// google/diffusiongemma-26B-A4B-it `model.safetensors.index.json` (fetched 2026-10-30).
    func testRealCheckpointKeyNamesPassThrough() {
        let checkpointKeys = [
            "model.decoder.embed_tokens.weight",
            "model.decoder.norm.weight",
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.encoder.embed_vision.embedding_projection.weight",
            "model.encoder.language_model.layers.0.layer_scalar",
            "model.encoder.vision_tower.std_scale",
        ]
        let weights = Dictionary(uniqueKeysWithValues: checkpointKeys.map { ($0, scalar()) })

        let sanitized = SharpAIDiffusionGemmaSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), Set(checkpointKeys))
    }
}
