// SharpAI fork overlay tests for cohere_asr key remapping and conv layout. Not part of upstream.
// No network and no real weights: tensors are zeros with HF (PyTorch) shapes.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAICohereAsrSanitizeTests: XCTestCase {

    static let tinyJSON = """
        {
          "model_type": "cohere_asr",
          "vocab_size": 32,
          "hidden_size": 16,
          "num_hidden_layers": 2,
          "num_attention_heads": 2,
          "intermediate_size": 32,
          "max_position_embeddings": 64,
          "tie_word_embeddings": false,
          "encoder_config": {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "conv_kernel_size": 3,
            "subsampling_conv_channels": 8,
            "num_mel_bins": 16,
            "max_position_embeddings": 128
          }
        }
        """

    private func tinyConfiguration(tied: Bool = false) throws -> SharpAICohereAsrConfiguration {
        var config = try JSONDecoder.json5().decode(
            SharpAICohereAsrConfiguration.self, from: Data(Self.tinyJSON.utf8))
        config.tieWordEmbeddings = tied
        return config
    }

    func testSubsamplingKeysAreRemappedAndConvWeightsTransposed() throws {
        let config = try tinyConfiguration()
        let weights: [String: MLXArray] = [
            "model.encoder.subsampling.layers.0.weight": MLXArray.zeros([8, 1, 3, 3]),
            "model.encoder.subsampling.layers.0.bias": MLXArray.zeros([8]),
            "model.encoder.subsampling.layers.2.weight": MLXArray.zeros([8, 1, 3, 3]),
            "model.encoder.subsampling.layers.3.weight": MLXArray.zeros([8, 8, 1, 1]),
            "model.encoder.subsampling.layers.5.weight": MLXArray.zeros([8, 1, 3, 3]),
            "model.encoder.subsampling.layers.6.weight": MLXArray.zeros([8, 8, 1, 1]),
        ]

        let out = try SharpAICohereAsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(out["model.encoder.subsampling.layers.0.weight"]?.shape, [8, 3, 3, 1])
        XCTAssertEqual(out["model.encoder.subsampling.layers.0.bias"]?.shape, [8])
        XCTAssertEqual(out["model.encoder.subsampling.layers.1.weight"]?.shape, [8, 3, 3, 1])
        XCTAssertEqual(out["model.encoder.subsampling.layers.2.weight"]?.shape, [8, 1, 1, 8])
        XCTAssertEqual(out["model.encoder.subsampling.layers.3.weight"]?.shape, [8, 3, 3, 1])
        XCTAssertEqual(out["model.encoder.subsampling.layers.4.weight"]?.shape, [8, 1, 1, 8])
        XCTAssertNil(out["model.encoder.subsampling.layers.2.bias"])
        XCTAssertNil(out["model.encoder.subsampling.layers.5.weight"])
        XCTAssertNil(out["model.encoder.subsampling.layers.6.weight"])
    }

    func testMLXLayoutCheckpointIsNotTransposedAgain() throws {
        let config = try tinyConfiguration()
        let weights: [String: MLXArray] = [
            "model.encoder.subsampling.layers.0.weight": MLXArray.zeros([8, 3, 3, 1]),
            "model.encoder.subsampling.layers.3.weight": MLXArray.zeros([8, 1, 1, 8]),
            "model.encoder.layers.0.conv.depthwise_conv.weight": MLXArray.zeros([16, 3, 1]),
        ]

        let out = try SharpAICohereAsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(out["model.encoder.subsampling.layers.0.weight"]?.shape, [8, 3, 3, 1])
        XCTAssertEqual(out["model.encoder.subsampling.layers.2.weight"]?.shape, [8, 1, 1, 8])
        XCTAssertEqual(
            out["model.encoder.layers.0.conv.depthwise_conv.weight"]?.shape, [16, 3, 1])
    }

    func testConv1dWeightsAreTransposedAndBufferKeysDropped() throws {
        let config = try tinyConfiguration()
        let weights: [String: MLXArray] = [
            "model.encoder.layers.0.conv.depthwise_conv.weight": MLXArray.zeros([16, 1, 3]),
            "model.encoder.layers.0.conv.pointwise_conv1.weight": MLXArray.zeros([32, 16, 1]),
            "model.encoder.layers.0.conv.norm.running_mean": MLXArray.zeros([16]),
            "model.encoder.layers.0.conv.norm.num_batches_tracked": MLXArray.zeros([1]),
            "preprocessor.featurizer.window": MLXArray.zeros([4]),
            "model.encoder.encode_positions.inv_freq": MLXArray.zeros([8]),
            "model.decoder.embed_tokens.weight": MLXArray.zeros([32, 16]),
        ]

        let out = try SharpAICohereAsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(
            out["model.encoder.layers.0.conv.depthwise_conv.weight"]?.shape, [16, 3, 1])
        XCTAssertEqual(
            out["model.encoder.layers.0.conv.pointwise_conv1.weight"]?.shape, [32, 1, 16])
        XCTAssertEqual(out["model.encoder.layers.0.conv.norm.running_mean"]?.shape, [16])
        XCTAssertEqual(out["model.decoder.embed_tokens.weight"]?.shape, [32, 16])
        XCTAssertNil(out["model.encoder.layers.0.conv.norm.num_batches_tracked"])
        XCTAssertNil(out["preprocessor.featurizer.window"])
        XCTAssertNil(out["model.encoder.encode_positions.inv_freq"])
    }

    func testTiedOutputProjectionFallsBackToEmbeddings() throws {
        let config = try tinyConfiguration(tied: true)
        let weights: [String: MLXArray] = [
            "model.decoder.embed_tokens.weight": MLXArray.zeros([32, 16]),
            "proj_out.bias": MLXArray.zeros([32]),
        ]

        let out = try SharpAICohereAsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(out["proj_out.weight"]?.shape, [32, 16])
    }

    func testUntiedMissingOutputProjectionIsNotInvented() throws {
        let config = try tinyConfiguration(tied: false)
        let weights: [String: MLXArray] = [
            "model.decoder.embed_tokens.weight": MLXArray.zeros([32, 16]),
        ]

        let out = try SharpAICohereAsrSanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(out["proj_out.weight"])
    }

    /// Builds HF-named, PyTorch-shaped zeros from the Swift parameter tree, then checks that
    /// sanitize returns exactly the Swift parameter names and MLX shapes. This does not prove
    /// the HF names are right, only that the remapping matches this Swift tree.
    func testSanitizedKeysMatchSwiftParameterTree() throws {
        let config = try tinyConfiguration()
        let model = try SharpAICohereAsrForConditionalGeneration(config)
        let swiftParameters = model.parameters().flattened()

        let swiftToHFSubsampling = [0: 0, 1: 2, 2: 3, 3: 5, 4: 6]
        let prefix = "model.encoder.subsampling.layers."
        var hfWeights: [String: MLXArray] = [:]
        for (key, value) in swiftParameters {
            var hfKey = key
            if key.hasPrefix(prefix) {
                let rest = key.dropFirst(prefix.count)
                if let dot = rest.firstIndex(of: "."),
                    let index = Int(rest[..<dot]),
                    let hfIndex = swiftToHFSubsampling[index]
                {
                    hfKey = prefix + String(hfIndex) + String(rest[dot...])
                }
            }
            hfWeights[hfKey] = MLXArray.zeros(Self.pytorchShape(value.shape))
        }

        let sanitized = try SharpAICohereAsrSanitizer.sanitize(hfWeights, configuration: config)

        XCTAssertEqual(Set(sanitized.keys), Set(swiftParameters.map(\.0)))
        for (key, value) in swiftParameters {
            XCTAssertEqual(sanitized[key]?.shape, value.shape, key)
        }
    }

    /// MLX conv layout to PyTorch layout, the inverse of the sanitizer's transpose.
    private static func pytorchShape(_ shape: [Int]) -> [Int] {
        switch shape.count {
        case 4:
            return [shape[0], shape[3], shape[1], shape[2]]
        case 3:
            return [shape[0], shape[2], shape[1]]
        default:
            return shape
        }
    }
}
