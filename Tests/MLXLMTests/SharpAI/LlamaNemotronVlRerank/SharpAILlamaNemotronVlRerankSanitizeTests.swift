// SharpAI fork overlay tests for llama_nemotron_vl_rerank weight sanitize. Not part of upstream.

import MLX
import MLXVLM
import XCTest

final class SharpAILlamaNemotronVlRerankSanitizeTests: XCTestCase {

    /// Tiny shapes that keep the reranker's structure: a SigLIP tower on 32x32 pixels (2x2 patches),
    /// one 2x2 pixel-shuffle step into one image token, and a bidirectional Llama decoder.
    private let tinyJSON = """
        {
            "model_type": "llama_nemotron_vl_rerank",
            "architectures": ["LlamaNemotronVLForSequenceClassification"],
            "llm_config": {
                "model_type": "llama_bidirec",
                "vocab_size": 64,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 64,
                "rms_norm_eps": 1e-5,
                "rope_theta": 500000.0
            },
            "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 32,
                "patch_size": 16,
                "image_size": 32,
                "num_channels": 3,
                "layer_norm_eps": 1e-6
            },
            "downsample_ratio": 0.5,
            "img_context_token_id": 60,
            "pooling": "avg"
        }
        """

    func testSanitizePassesKeysThroughUnchanged() {
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": MLXArray.zeros([4, 2]),
            "score.weight": MLXArray.zeros([1, 2]),
        ]

        let result = SharpAILlamaNemotronVlRerankSanitize.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(weights.keys))
    }

    /// Random-weight forward pass on tiny shapes. The reranker returns one score per sequence.
    func testTinyForwardPass() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAILlamaNemotronVlRerankConfiguration.self, from: Data(tinyJSON.utf8))
        let model = try SharpAILlamaNemotronVlRerankModel(config)

        // Text only: no image rows, so no pixels.
        let textOnly = try model.rerankScores(MLXArray([5, 6, 7, 8] as [Int32], [1, 4]))
        XCTAssertEqual(textOnly.shape, [1, 1])

        // One image with one image token (id 60) in the sequence.
        let pixels = MLXArray.zeros([1, 3, 32, 32])
        let withImage = try model.rerankScores(
            MLXArray([5, 60, 7, 8] as [Int32], [1, 4]), pixelValues: pixels)
        XCTAssertEqual(withImage.shape, [1, 1])
        XCTAssertTrue(withImage.asArray(Float.self).allSatisfy { $0.isFinite })

        // Pixels with no matching image token must fail, not silently drop the image.
        XCTAssertThrowsError(
            try model.rerankScores(MLXArray([5, 6, 7, 8] as [Int32], [1, 4]), pixelValues: pixels))
    }
}
