// Copyright © 2026 Apple Inc.

import Dispatch

/// Runs one generation's blocking work outside Swift's cooperative pool.
actor GenerationWorker {
    private nonisolated let queue = DispatchSerialQueue(
        label: "ml-explore.mlx-swift-lm.generation")

    nonisolated var unownedExecutor: UnownedSerialExecutor {
        queue.asUnownedSerialExecutor()
    }

    // Keep the body synchronous so all evaluation and cleanup stay on this executor.
    func run<R: Sendable>(_ body: @Sendable () -> R) -> R {
        body()
    }
}
