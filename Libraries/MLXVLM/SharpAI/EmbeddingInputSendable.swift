// SharpAI fork only. Swift 6.2 (Xcode 26.3, which this fork's CI uses) rejects an actor-isolated
// `embed(_ inputs: sending [EmbeddingInput], task:)` when `EmbeddingInput` is not Sendable
// ("non-Sendable parameter type cannot be sent from caller of protocol requirement ... into
// actor-isolated implementation"); upstream builds with Xcode 27 (Swift 6.4), where it compiles.
// `EmbeddingInput` is an immutable value (`let` fields only), so crossing an isolation boundary is
// safe. Remove this file once the fork's CI builds with a toolchain that accepts the upstream code.

extension EmbeddingInput: @unchecked Sendable {}
