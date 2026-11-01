// SharpAI fork overlay: cosmos3_omni subfolder weight selection. Not part of upstream.
//
// Verified against the nvidia/Cosmos3-Nano file listing and root `model.safetensors.index.json`
// (https://huggingface.co/nvidia/Cosmos3-Nano/raw/main/model.safetensors.index.json): the 814
// transformer keys live in `transformer/diffusion_pytorch_model-0000N-of-00007.safetensors` and the
// 351 vision keys in `vision_encoder/model.safetensors`. `vae/` and `sound_tokenizer/` hold single
// files that are not in the index and are out of scope for the model forward.
// UNVERIFIED: no repo source sets allow_patterns for cosmos3_omni (the checkpoint config only has
// `allow_patterns_overrides: ["*/*.safetensors"]`); nothing calls these functions yet.

import Foundation

public enum SharpAICosmos3OmniWeightFiles {

    public static let transformerFolder = "transformer"
    public static let visionEncoderFolder = "vision_encoder"

    /// Returns the transformer shards and the vision encoder files, each sorted, from the one-folder
    /// deep `.safetensors` paths. Other folders (`vae/`, `sound_tokenizer/`) are left out.
    public static func modelSafetensors(in relativePaths: [String]) -> (
        transformer: [String], vision: [String]
    ) {
        let files = subfolderSafetensors(in: relativePaths)
        return (
            files.filter { $0.hasPrefix(transformerFolder + "/") },
            files.filter { $0.hasPrefix(visionEncoderFolder + "/") }
        )
    }

    /// Returns `<folder>/<name>.safetensors` paths, sorted. Top-level and deeper files are skipped.
    public static func subfolderSafetensors(in relativePaths: [String]) -> [String] {
        relativePaths
            .filter { path in
                let parts = path.split(separator: "/", omittingEmptySubsequences: false)
                return parts.count == 2 && !parts[0].isEmpty
                    && parts[1].hasSuffix(".safetensors")
            }
            .sorted()
    }
}
