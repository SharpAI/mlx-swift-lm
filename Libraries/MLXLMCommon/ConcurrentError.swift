import Foundation
import MLX

/// Error thrown when SSD expert streaming encounters a corrupted, truncated,
/// or incomplete safetensors file during pread I/O.
public struct SSDStreamingError: Error, LocalizedError {
    public let underlyingError: Error

    public var errorDescription: String? {
        "MLX SSD Streaming Error: \(underlyingError.localizedDescription). The model safetensors file may be corrupted, truncated, or incomplete. Try re-downloading the model."
    }
}

/// Global error latch for SSD streaming errors that occur inside non-throwing
/// `callAsFunction` paths. Set by `ThreadSafeError.check()`, cleared and
/// inspected by the generation loop after each token.
public final class SSDStreamingErrorLatch: @unchecked Sendable {
    public static let shared = SSDStreamingErrorLatch()
    private let lock = NSLock()
    private var _error: Error?

    /// Record an error (first-wins semantics).
    public func set(_ error: Error) {
        lock.withLock {
            if _error == nil { _error = error }
        }
    }

    /// Consume and return the recorded error, resetting the latch.
    /// Returns nil if no error was recorded.
    public func consume() -> Error? {
        lock.withLock {
            let e = _error
            _error = nil
            return e
        }
    }

    /// Throw the recorded error if one exists, then clear it.
    public func throwIfSet() throws {
        if let error = consume() {
            throw error
        }
    }
}

package final class ThreadSafeError: @unchecked Sendable {
    package let lock = NSLock()
    package var error: Swift.Error?
    
    package init() {}

    package func catchError(_ block: () throws -> Void) {
        do {
            try MLX.withError {
                try block()
            }
        } catch {
            lock.withLock {
                if self.error == nil { self.error = error }
            }
        }
    }
    
    /// Check if any error was recorded during concurrent I/O.
    ///
    /// Instead of calling `fatalError` (which crashes the entire app), this
    /// posts the error to the global `SSDStreamingErrorLatch` so the generation
    /// loop can detect it after the current token and surface it gracefully
    /// in the UI (e.g., prompting a re-download).
    package func check() {
        if let error = error {
            SSDStreamingErrorLatch.shared.set(
                SSDStreamingError(underlyingError: error)
            )
        }
    }
}
