// SharpAI fork overlay helpers. New file; not part of upstream mlx-swift-lm.
//
// Upstream watch: if upstream ever adds a type or function named `SharpAIOverlay`
// (vanishingly unlikely: the name is fork-specific), this file breaks the build and must be
// renamed. See scripts/conformance/README.md, rules 3 and 4.

import Foundation

/// Namespace for fork-only (SharpAI) additions that must never collide with upstream names.
public enum SharpAIOverlay {

    /// Registers `creator` for `type` in `registry` only if `type` is not already registered.
    ///
    /// An existing registration (for example one added by an upstream sync) always wins and is
    /// never overwritten. This is the only sanctioned way for overlay code to register a
    /// `model_type`; a lint test (`SharpAIOverlayTests`) rejects other direct
    /// `registerModelType` calls in `Libraries/`.
    ///
    /// The check and the registration are two actor hops, not one atomic step. Overlay
    /// registration runs once at startup, so this is not a concern in practice; do not call it
    /// concurrently for the same `type` from several tasks.
    ///
    /// - Returns: `true` if this call registered the creator, `false` if `type` was already present.
    @discardableResult
    public static func registerIfAbsent<T>(
        registry: ModelTypeRegistry<T>,
        type: String,
        creator: sending @escaping (Data) throws -> T
    ) async -> Bool {
        if await registry.contains(type) { return false }
        await registry.registerModelType(type, creator: creator)
        return true
    }
}
