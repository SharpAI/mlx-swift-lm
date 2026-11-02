// SharpAI fork overlay tests for cosmos3_omni subfolder weight selection. Not part of upstream.

import Foundation
@testable import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAICosmos3OmniWeightFilesTests: XCTestCase {

    func testSelectsOnlyOneFolderDeepSafetensors() {
        let paths = [
            "model.safetensors",
            "transformer/model-00002-of-00002.safetensors",
            "transformer/model-00001-of-00002.safetensors",
            "transformer/config.json",
            "nested/deep/model.safetensors",
        ]

        XCTAssertEqual(
            SharpAICosmos3OmniWeightFiles.subfolderSafetensors(in: paths),
            [
                "transformer/model-00001-of-00002.safetensors",
                "transformer/model-00002-of-00002.safetensors",
            ])
    }

    func testEmptyInputReturnsEmpty() {
        XCTAssertTrue(SharpAICosmos3OmniWeightFiles.subfolderSafetensors(in: []).isEmpty)
    }

    func testSplitsTransformerAndVisionEncoderFiles() {
        let paths = [
            "model.safetensors.index.json",
            "transformer/diffusion_pytorch_model-00002-of-00007.safetensors",
            "transformer/diffusion_pytorch_model-00001-of-00007.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vision_encoder/model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "sound_tokenizer/diffusion_pytorch_model.safetensors",
        ]

        let files = SharpAICosmos3OmniWeightFiles.modelSafetensors(in: paths)

        XCTAssertEqual(
            files.transformer,
            [
                "transformer/diffusion_pytorch_model-00001-of-00007.safetensors",
                "transformer/diffusion_pytorch_model-00002-of-00007.safetensors",
            ])
        XCTAssertEqual(files.vision, ["vision_encoder/model.safetensors"])
    }

    /// The root index of nvidia/Cosmos3-Nano names files in `transformer/` and `vision_encoder/`.
    /// The common loader follows an index into subdirectories when every named file exists, so the
    /// real layout selects both folders' shards without extra wiring.
    func testRootIndexSelectsSubfolderShards() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("cosmos3-index-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let fm = FileManager.default
        for sub in ["transformer", "vision_encoder", "vae"] {
            try fm.createDirectory(at: dir.appendingPathComponent(sub), withIntermediateDirectories: true)
        }
        let shards = [
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "vision_encoder/model.safetensors",
        ]
        for name in shards + ["vae/diffusion_pytorch_model.safetensors"] {
            try Data().write(to: dir.appendingPathComponent(name))
        }
        let map = [
            "layers.0.self_attn.to_q.weight": shards[0],
            "layers.1.self_attn.to_q.weight": shards[1],
            "blocks.0.attn.qkv.weight": shards[2],
        ]
        let index = try JSONSerialization.data(withJSONObject: ["metadata": [:], "weight_map": map])
        try index.write(to: dir.appendingPathComponent("model.safetensors.index.json"))

        let urls = try safetensorWeightURLs(in: dir)
        XCTAssertEqual(
            urls.map { $0.path.replacingOccurrences(of: dir.path + "/", with: "") }.sorted(),
            shards.sorted())
    }
}
