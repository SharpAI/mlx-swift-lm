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
    /// The check and the registration run inside one actor-isolated method
    /// (`ModelTypeRegistry.sharpAIRegisterIfAbsent`) with no suspension point between them, so
    /// the pair is atomic: concurrent callers for the same `type` get exactly one `true`.
    /// (An earlier draft did `await contains` then `await registerModelType`, two actor hops;
    /// `OverlayRegistrationTests.concurrentRegistrationIsAtomic` pins the fix.)
    ///
    /// - Returns: `true` if this call registered the creator, `false` if `type` was already present.
    @discardableResult
    public static func registerIfAbsent<T>(
        registry: ModelTypeRegistry<T>,
        type: String,
        creator: sending @escaping (Data) throws -> T
    ) async -> Bool {
        await registry.sharpAIRegisterIfAbsent(type, creator: creator)
    }
}

extension ModelTypeRegistry {
    /// Atomic check-then-register. Actor-isolated and synchronous, so no other task can register
    /// `type` between the `contains` check and the insertion. Use through
    /// ``SharpAIOverlay/registerIfAbsent(registry:type:creator:)``.
    ///
    /// upstream-watch: if upstream adds a member with this exact name to `ModelTypeRegistry`,
    /// the build breaks; rename ours (see scripts/conformance/upstream/UPSTREAM_WATCH.md).
    func sharpAIRegisterIfAbsent(
        _ type: String, creator: @escaping (Data) throws -> T
    ) -> Bool {
        if contains(type) { return false }
        registerModelType(type, creator: creator)
        return true
    }
}
