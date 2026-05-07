import Foundation

/// Global configuration for Multi-Token Prediction (MTP) Speculative Decoding
public struct MTPConfig: Sendable {
    /// Indicates whether models should retain their `mtp.*` weights during initialization.
    /// By default, these weights are aggressively stripped to save memory unless the user
    /// specifically enables MTP speculative decoding.
    public static var retainMTPWeights: Bool {
        ProcessInfo.processInfo.environment["SWIFTLM_MTP_ENABLE"] == "1"
    }
}
