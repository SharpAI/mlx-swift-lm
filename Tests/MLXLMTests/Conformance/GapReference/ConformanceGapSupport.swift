// Known-gap wrapper shared by the ConformanceGap* suites.
//
// Normal run: a gap is `withKnownIssue(isIntermittent: true)` -- green while the gap exists AND
// green once it is fixed, so an upstream sync that implements it is never blocked. Unlike a bare
// wrapper, the OUTCOME is tracked: the wrapper observes whether any assertion in the body failed.
//   * assertions failed  -> `[gap][OPEN] <name>`   (gap still exists)
//   * assertions passed  -> `[gap][FIXED] <name>`  (gap closed: remove the wrapper / retire the overlay)
// A running `[gap][SUMMARY] fixed N, open M` line follows each report (Swift Testing has no
// suite-end hook, so the last SUMMARY line of the log is the final count).
// Strict run (`CONFORMANCE_STRICT=1`, same switch as ConformanceRegistryTests' LANDED rule): a FIXED
// gap is a test FAILURE ("remove the knownIssue wrapper / retire the overlay"); by default it is a
// warning line only.
// Raw run (`CONFORMANCE_GAP_RAW=1`): the wrapper is skipped and the same assertions run bare, so
// the red state can be inspected (and re-demonstrated) at any time without editing code.
// upstream-watch: none (SharpAI-only helper).

import Foundation
import Testing

/// Outcome book-keeping for known gaps. A class so the shared instance can be mutated from
/// parallel tests; tests of the support itself use their own instance.
final class ConformanceGapTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var fixed: [String: Bool] = [:]

    func record(_ name: String, fixed isFixed: Bool) {
        lock.lock(); defer { lock.unlock() }
        // A name seen twice (e.g. a parameterised test) is FIXED only if every occurrence passed.
        fixed[name] = (fixed[name] ?? true) && isFixed
    }

    var counts: (fixed: Int, open: Int) {
        lock.lock(); defer { lock.unlock() }
        let f = fixed.values.filter { $0 }.count
        return (f, fixed.count - f)
    }

    func isFixed(_ name: String) -> Bool? {
        lock.lock(); defer { lock.unlock() }
        return fixed[name]
    }
}

final class IssueFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    func set() { lock.lock(); flag = true; lock.unlock() }
    var value: Bool { lock.lock(); defer { lock.unlock() }; return flag }
}

enum ConformanceGap {
    static var raw: Bool { ProcessInfo.processInfo.environment["CONFORMANCE_GAP_RAW"] == "1" }
    static var strict: Bool { ProcessInfo.processInfo.environment["CONFORMANCE_STRICT"] == "1" }

    static let shared = ConformanceGapTracker()

    static let strictMessage = "remove the knownIssue wrapper / retire the overlay"

    static func knownIssue(
        _ comment: Comment, sourceLocation: SourceLocation = #_sourceLocation,
        _ body: () throws -> Void
    ) rethrows {
        if raw {
            try body()
        } else {
            track(
                "\(comment)", tracker: shared, strict: strict, sourceLocation: sourceLocation,
                print: { Swift.print($0) },
                onStrictViolation: { Issue.record(Comment(rawValue: $0), sourceLocation: sourceLocation) },
                body)
        }
    }

    /// Core of the wrapper with every side channel injected, so it can be tested without touching
    /// the process environment or failing the surrounding test. Returns true when the gap is FIXED.
    @discardableResult
    static func track(
        _ name: String, tracker: ConformanceGapTracker, strict: Bool,
        sourceLocation: SourceLocation = #_sourceLocation,
        print: (String) -> Void, onStrictViolation: (String) -> Void,
        _ body: () throws -> Void
    ) -> Bool {
        let saw = IssueFlag()
        do {
            try withKnownIssue(
                Comment(rawValue: name), isIntermittent: true, sourceLocation: sourceLocation,
                { try body() },
                matching: { _ in saw.set(); return true })
        } catch {
            // Unreachable: `matching` accepts every issue, including a thrown error.
            saw.set()
        }
        let isFixed = !saw.value
        tracker.record(name, fixed: isFixed)
        print("[gap][\(isFixed ? "FIXED" : "OPEN")] \(name)")
        if isFixed {
            if strict {
                onStrictViolation("gap FIXED: \(name) -- \(strictMessage)")
            } else {
                print("[gap][WARNING] \(name) is FIXED -- \(strictMessage) (failure under CONFORMANCE_STRICT=1)")
            }
        }
        let c = tracker.counts
        print("[gap][SUMMARY] fixed \(c.fixed), open \(c.open)")
        return isFixed
    }
}
