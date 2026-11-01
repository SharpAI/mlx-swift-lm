// SharpAI fork overlay: alpamayo2_super configuration. Not part of upstream mlx-swift-lm.
//
// Verified against: https://github.com/NVlabs/alpamayo2/blob/main/src/alpamayo2_super/config.py
// and https://huggingface.co/nvidia/Alpamayo2-Super/blob/main/config.json (history_vocab_size 1000,
// future_vocab_size 3000, traj_vocab_size 4000, traj_ids, tokens_per_history_traj 45).
// Keys inside `vlm_config` are read first, then top-level keys.

import Foundation
import MLXLMCommon

public enum SharpAIAlpamayo2SuperError: LocalizedError {
    case forwardNotImplemented(String)
    case invalidTrajectoryRange(String)
    case duplicateWeight(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass is not implemented yet "
                + "(architecture UNVERIFIED)."
        case .invalidTrajectoryRange(let detail):
            return "Invalid alpamayo2_super trajectory token range: \(detail)"
        case .duplicateWeight(let key):
            return "Two alpamayo2_super checkpoint keys map to the same name: \(key)"
        }
    }
}

public struct SharpAIAlpamayo2SuperConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let vocabularySize: Int?

    /// True when the checkpoint nests its model keys under `vlm_config`.
    public let isWrapped: Bool

    // Checked against the released config.json: it has no `trajectory_token_start` or
    // `trajectory_token_count`. It has a `traj_ids` object and `traj_vocab_size` 4000. The keys
    // below are still decoded, but they are absent from the released checkpoint.
    public let trajectoryTokenStart: Int?
    public let trajectoryTokenCount: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vlmConfig = "vlm_config"
        case vocabularySize = "vocab_size"
        // Not present in the released config.json (verified). See the note above.
        case trajectoryTokenStart = "trajectory_token_start"
        case trajectoryTokenCount = "trajectory_token_count"
    }

    public init(from decoder: Decoder) throws {
        let outer = try decoder.container(keyedBy: CodingKeys.self)

        var inner: KeyedDecodingContainer<CodingKeys>?
        if outer.contains(.vlmConfig), try !outer.decodeNil(forKey: .vlmConfig) {
            inner = try outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .vlmConfig)
        }
        isWrapped = inner != nil

        modelType =
            try Self.value(String.self, .modelType, inner, outer) ?? "alpamayo2_super"
        vocabularySize = try Self.value(Int.self, .vocabularySize, inner, outer)
        trajectoryTokenStart =
            try Self.value(Int.self, .trajectoryTokenStart, inner, outer)
        trajectoryTokenCount =
            try Self.value(Int.self, .trajectoryTokenCount, inner, outer)
    }

    /// Reads `key` from the unwrapped `vlm_config` first, then from the top level.
    private static func value<T: Decodable>(
        _ type: T.Type,
        _ key: CodingKeys,
        _ inner: KeyedDecodingContainer<CodingKeys>?,
        _ outer: KeyedDecodingContainer<CodingKeys>
    ) throws -> T? {
        if let inner, let value = try inner.decodeIfPresent(T.self, forKey: key) {
            return value
        }
        return try outer.decodeIfPresent(T.self, forKey: key)
    }

    /// Trajectory token ids, when both range keys are present and valid.
    public var trajectoryTokenRange: Range<Int>? {
        guard let start = trajectoryTokenStart, let count = trajectoryTokenCount,
            start >= 0, count > 0
        else { return nil }
        return start ..< start + count
    }

    public func validateModelConfiguration() throws {
        if let vocabularySize, vocabularySize <= 0 {
            throw ModelFactoryError.invalidConfiguration(
                "alpamayo2_super: vocab_size must be positive")
        }
        if trajectoryTokenStart != nil || trajectoryTokenCount != nil {
            guard let start = trajectoryTokenStart, let count = trajectoryTokenCount,
                start >= 0, count > 0
            else {
                throw SharpAIAlpamayo2SuperError.invalidTrajectoryRange(
                    "start must be >= 0 and count must be > 0")
            }
            if let vocabularySize, start + count > vocabularySize {
                throw SharpAIAlpamayo2SuperError.invalidTrajectoryRange(
                    "range exceeds vocab_size \(vocabularySize)")
            }
        }
    }
}
