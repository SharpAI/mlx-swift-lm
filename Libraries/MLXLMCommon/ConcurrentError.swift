import Foundation
import MLX

/// Error thrown when SSD expert streaming encounters a corrupted, truncated,
/// or incomplete safetensors file during pread I/O.
public struct SSDStreamingError: Error, LocalizedError {
    public let underlyingError: Error

    public init(underlyingError: Error) {
        self.underlyingError = underlyingError
    }

    public var errorDescription: String? {
        "MLX SSD Streaming Error: \(underlyingError.localizedDescription). The model safetensors file may be corrupted, truncated, or incomplete. Try re-downloading the model."
    }
}

private enum SSDStreamingErrorLatchContext {
    static let threadDictionaryKey = "MLXLMCommon.SSDStreamingErrorLatch.active"
}

/// Error latch for SSD streaming errors that occur inside non-throwing
/// `callAsFunction` paths. A generation installs its own active latch around
/// model execution so concurrent sessions do not cross-contaminate each other.
public final class SSDStreamingErrorLatch: @unchecked Sendable {
    private let lock = NSLock()
    private var _error: Error?

    public init() {}

    package static func withActive<T>(_ latch: SSDStreamingErrorLatch, _ body: () throws -> T) rethrows -> T {
        let key = SSDStreamingErrorLatchContext.threadDictionaryKey as NSString
        let threadDictionary = Thread.current.threadDictionary
        let previous = threadDictionary[key]
        threadDictionary[key] = latch
        defer {
            if let previous {
                threadDictionary[key] = previous
            } else {
                threadDictionary.removeObject(forKey: key)
            }
        }
        return try body()
    }

    package static var active: SSDStreamingErrorLatch? {
        let key = SSDStreamingErrorLatchContext.threadDictionaryKey as NSString
        return Thread.current.threadDictionary[key] as? SSDStreamingErrorLatch
    }

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
    private let latch: SSDStreamingErrorLatch?
    
    package init(latch: SSDStreamingErrorLatch? = SSDStreamingErrorLatch.active) {
        self.latch = latch
    }

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
    @discardableResult
    package func check() -> SSDStreamingError? {
        if let error = error {
            let streamingError = SSDStreamingError(underlyingError: error)
            latch?.set(streamingError)
            return streamingError
        }
        return nil
    }
}
