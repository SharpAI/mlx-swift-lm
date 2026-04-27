import Foundation
import MLX

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
    
    package func check() {
        if let error = error {
            fatalError("MLX SSD Streaming Error: \(error.localizedDescription). (The model safetensors file may be corrupted, truncated, or incomplete).")
        }
    }
}
