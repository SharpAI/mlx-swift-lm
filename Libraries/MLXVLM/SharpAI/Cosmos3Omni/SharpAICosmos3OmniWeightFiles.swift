// SharpAI fork overlay: cosmos3_omni subfolder weight selection. Not part of upstream.
//
// UNVERIFIED: no repo source sets allow_patterns for cosmos3_omni. The one-folder depth is an
// assumption, and nothing calls this function yet.
// UNVERIFIED: the folder name is not fixed in the repo, so this only matches the depth.

import Foundation

public enum SharpAICosmos3OmniWeightFiles {

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
