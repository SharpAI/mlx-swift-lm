// Known-gap wrapper shared by the ConformanceGap* suites.
//
// Normal run: a gap is `withKnownIssue(isIntermittent: true)` -- green while the gap exists AND
// green once it is fixed, so an upstream sync that implements it is never blocked. Unlike a bare
// wrapper, the OUTCOME is tracked: the wrapper observes whether any assertion in the body failed.
//   * assertions failed  -> `[gap][OPEN] <name>`   (gap still exists)
//   * assertions passed  -> `[gap][FIXED] <name>`  (gap closed: remove the wrapper / retire the overlay)
//   * no issue AND no counted assertion -> `[gap][UNKNOWN] <name>` (never FIXED: an empty body or bare
//     `#expect`, which Swift Testing cannot observe when it passes; assert via `ConformanceGap.expect`)
// Only failures raised by `ConformanceGap.expect` (tagged `[gap-expect]`) are absorbed; any other issue
// in the body (a throw, `Issue.record`, a bare `#expect`) is NOT a gap signature and fails the test.
// A `[gap][SUMMARY] fixed N, open M, unknown K` line follows each report, printed inside the tracker
// lock from the final state, so the LAST SUMMARY line of the log is authoritative (Swift Testing has
// no suite-end hook, so there is no separate end-of-run print).
// Strict run (`CONFORMANCE_STRICT=1`, same switch as ConformanceRegistryTests' LANDED rule): a FIXED
// gap is a test FAILURE ("remove the knownIssue wrapper / retire the overlay"); by default it is a
// warning line only.
// Raw run (`CONFORMANCE_GAP_RAW=1`): the wrapper is skipped and the same assertions run bare, so
// the red state can be inspected (and re-demonstrated) at any time without editing code.
// upstream-watch: none (SharpAI-only helper).

import Foundation
import Testing

/// Outcome of one known-gap body.
enum ConformanceGapOutcome: Sendable, Equatable {
    case open     // an assertion failed: the gap still exists
    case fixed    // at least one counted assertion ran and none failed
    case unknown  // no issue recorded, but no counted assertion ran either (empty body / bare #expect)
}

/// Outcome book-keeping for known gaps. A class so the shared instance can be mutated from
/// parallel tests; tests of the support itself use their own instance.
final class ConformanceGapTracker: @unchecked Sendable {
    private let lock = NSLock()
    private var outcomes: [String: ConformanceGapOutcome] = [:]

    /// Records the outcome and calls `emit` for each report line WHILE HOLDING the lock, so lines of
    /// concurrent gaps never interleave and the last SUMMARY line printed is computed from the final
    /// state (it is authoritative). `emit` must not call back into the tracker.
    func record(
        _ name: String, outcome: ConformanceGapOutcome,
        emit: (_ summary: String) -> Void = { _ in }
    ) {
        lock.lock(); defer { lock.unlock() }
        // A name seen twice (parameterised test): OPEN beats UNKNOWN beats FIXED.
        switch (outcomes[name], outcome) {
        case (.open, _), (_, .open): outcomes[name] = .open
        case (.unknown, _), (_, .unknown): outcomes[name] = .unknown
        default: outcomes[name] = .fixed
        }
        emit(Self.summary(outcomes))
    }

    private static func summary(_ o: [String: ConformanceGapOutcome]) -> String {
        let (f, op, u) = count(o)
        return "[gap][SUMMARY] fixed \(f), open \(op), unknown \(u)"
    }

    private static func count(_ o: [String: ConformanceGapOutcome]) -> (Int, Int, Int) {
        var f = 0, op = 0, u = 0
        for v in o.values { switch v { case .fixed: f += 1; case .open: op += 1; case .unknown: u += 1 } }
        return (f, op, u)
    }

    var counts: (fixed: Int, open: Int, unknown: Int) {
        lock.lock(); defer { lock.unlock() }
        let c = Self.count(outcomes)
        return (c.0, c.1, c.2)
    }

    func outcome(_ name: String) -> ConformanceGapOutcome? {
        lock.lock(); defer { lock.unlock() }
        return outcomes[name]
    }

    func isFixed(_ name: String) -> Bool? {
        guard let o = outcome(name) else { return nil }
        if case .fixed = o { return true }
        return false
    }
}

enum ConformanceGap {
    static var raw: Bool { ProcessInfo.processInfo.environment["CONFORMANCE_GAP_RAW"] == "1" }
    static var strict: Bool { ProcessInfo.processInfo.environment["CONFORMANCE_STRICT"] == "1" }

    final class IssueFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var flag = false
        func set() { lock.lock(); flag = true; lock.unlock() }
        var value: Bool { lock.lock(); defer { lock.unlock() }; return flag }
    }

    /// Counts assertions made through `ConformanceGap.expect` inside one gap body.
    final class AssertionCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var n = 0
        func bump() { lock.lock(); n += 1; lock.unlock() }
        var count: Int { lock.lock(); defer { lock.unlock() }; return n }
    }

    @TaskLocal static var currentCounter: AssertionCounter?

    /// Swift Testing cannot observe a PASSING `#expect`, so "no issue recorded" alone cannot tell a
    /// fixed gap from an empty body. Assertions inside a gap body should go through this helper (or
    /// call `noteAssertion()`); a body that counted none is reported UNKNOWN, not FIXED.
    static func expect(
        _ condition: @autoclosure () -> Bool, _ comment: @autoclosure () -> Comment? = nil,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        noteAssertion()
        #expect(condition(), Comment(rawValue: gapTag + (comment()?.rawValue ?? "")), sourceLocation: sourceLocation)
    }

    /// Marker carried by every failure recorded through `expect`. The known-issue matcher accepts ONLY
    /// issues that are an expectation failure carrying this marker, so a gap absorbs the assertion it
    /// declares and nothing else: a thrown error, an `Issue.record`, a bare `#expect` or a crash-adjacent
    /// failure inside the body is a real failure, not an OPEN gap.
    static let gapTag = "[gap-expect] "

    static func isGapIssue(_ issue: Issue) -> Bool {
        guard case .expectationFailed = issue.kind else { return false }
        return issue.comments.contains { $0.rawValue.hasPrefix(gapTag) }
    }

    static func noteAssertion() { currentCounter?.bump() }

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
        let counter = AssertionCounter()
        do {
            try $currentCounter.withValue(counter) {
                try withKnownIssue(
                    Comment(rawValue: name), isIntermittent: true, sourceLocation: sourceLocation,
                    { try body() },
                    matching: { issue in
                        guard isGapIssue(issue) else { return false }
                        saw.set(); return true
                    })
            }
        } catch {
            // A thrown error is not a gap signature: `matching` rejected it, so Swift Testing has already
            // recorded it as a real failure. Do not count it as OPEN.
        }
        let outcome: ConformanceGapOutcome =
            saw.value ? .open : (counter.count > 0 ? .fixed : .unknown)
        // All report lines are printed inside the tracker lock (see `record`).
        tracker.record(name, outcome: outcome) { summary in
            switch outcome {
            case .open: print("[gap][OPEN] \(name)")
            case .unknown:
                print("[gap][UNKNOWN] \(name)")
                print("[gap][WARNING] \(name) recorded no counted assertion (empty body or bare #expect): not reported FIXED; use ConformanceGap.expect")
            case .fixed:
                print("[gap][FIXED] \(name)")
                if !strict {
                    print("[gap][WARNING] \(name) is FIXED -- \(strictMessage) (failure under CONFORMANCE_STRICT=1)")
                }
            }
            print(summary)
        }
        if case .fixed = outcome, strict {
            onStrictViolation("gap FIXED: \(name) -- \(strictMessage)")
        }
        return { if case .fixed = outcome { return true } else { return false } }()
    }
}
