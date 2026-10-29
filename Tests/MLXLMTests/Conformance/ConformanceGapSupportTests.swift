// Self-tests for GapReference/ConformanceGapSupport.swift (the outcome-tracking known-gap wrapper).
// Strict mode is driven by an injected flag, never by mutating the process environment.
// upstream-watch: none (SharpAI-only).

import Foundation
import Testing

struct ConformanceGapSupportTests {
    final class Log: @unchecked Sendable {
        var lines: [String] = []
        var violations: [String] = []
    }

    func run(_ name: String, strict: Bool, passing: Bool, tracker: ConformanceGapTracker = .init())
        -> (fixed: Bool, log: Log, tracker: ConformanceGapTracker)
    {
        let log = Log()
        let fixed = ConformanceGap.track(
            name, tracker: tracker, strict: strict,
            print: { log.lines.append($0) }, onStrictViolation: { log.violations.append($0) }
        ) {
            if passing { ConformanceGap.expect(1 + 1 == 2) } else { ConformanceGap.expect(1 + 1 == 3, "synthetic gap") }
        }
        return (fixed, log, tracker)
    }

    @Test("a passing body is reported FIXED (warning only, not strict)")
    func passingIsFixed() {
        let r = run("synthetic-fixed", strict: false, passing: true)
        #expect(r.fixed)
        #expect(r.tracker.isFixed("synthetic-fixed") == true)
        #expect(r.log.lines.contains("[gap][FIXED] synthetic-fixed"))
        #expect(r.log.lines.contains { $0.hasPrefix("[gap][WARNING]") })
        #expect(r.log.lines.last == "[gap][SUMMARY] fixed 1, open 0, unknown 0")
        #expect(r.log.violations.isEmpty)
    }

    @Test("a failing body is reported OPEN and swallowed")
    func failingIsOpen() {
        for strict in [false, true] {
            let r = run("synthetic-open", strict: strict, passing: false)
            #expect(!r.fixed)
            #expect(r.tracker.isFixed("synthetic-open") == false)
            #expect(r.log.lines.contains("[gap][OPEN] synthetic-open"))
            #expect(r.log.lines.last == "[gap][SUMMARY] fixed 0, open 1, unknown 0")
            #expect(r.log.violations.isEmpty, "an OPEN gap never fails, even in strict mode")
        }
    }

    @Test("strict mode flips a FIXED gap into a failure with the retire message")
    func strictFlipsFixed() {
        let r = run("synthetic-strict", strict: true, passing: true)
        #expect(r.fixed)
        #expect(r.log.violations.count == 1)
        #expect(r.log.violations.first?.contains("remove the knownIssue wrapper / retire the overlay") == true)
        #expect(r.log.violations.first?.contains("synthetic-strict") == true)
        #expect(!r.log.lines.contains { $0.hasPrefix("[gap][WARNING]") })
    }

    @Test("a throwing body counts as OPEN")
    func throwingIsOpen() {
        struct Boom: Error {}
        let log = Log()
        let fixed = ConformanceGap.track(
            "synthetic-throw", tracker: .init(), strict: true,
            print: { log.lines.append($0) }, onStrictViolation: { log.violations.append($0) }
        ) { throw Boom() }
        #expect(!fixed)
        #expect(log.violations.isEmpty)
    }

    @Test("summary counts accumulate across gaps; a name is FIXED only if every occurrence passed")
    func summaryAccumulates() {
        let t = ConformanceGapTracker()
        _ = run("a", strict: false, passing: true, tracker: t)
        _ = run("b", strict: false, passing: false, tracker: t)
        let r = run("a", strict: false, passing: false, tracker: t)  // same name now fails -> OPEN
        #expect(r.log.lines.last == "[gap][SUMMARY] fixed 0, open 2, unknown 0")
        #expect(t.counts.fixed == 0 && t.counts.open == 2 && t.counts.unknown == 0)
    }

    @Test("an empty body (zero assertions) is UNKNOWN, never FIXED, and never trips strict mode")
    func emptyBodyIsUnknown() {
        let log = Log()
        let t = ConformanceGapTracker()
        let fixed = ConformanceGap.track(
            "synthetic-empty", tracker: t, strict: true,
            print: { log.lines.append($0) }, onStrictViolation: { log.violations.append($0) }
        ) {}
        #expect(!fixed)
        #expect(log.lines.contains("[gap][UNKNOWN] synthetic-empty"))
        #expect(!log.lines.contains("[gap][FIXED] synthetic-empty"))
        #expect(log.violations.isEmpty)
    }

    @Test("the last SUMMARY line is authoritative under concurrent gaps")
    func summaryAuthoritativeConcurrent() async {
        let t = ConformanceGapTracker()
        let log = Log()
        let lock = NSLock()
        await withTaskGroup(of: Void.self) { g in
            for i in 0..<40 {
                g.addTask {
                    t.record("g\(i)", outcome: i % 2 == 0 ? .open : .fixed) { s in
                        lock.lock(); log.lines.append(s); lock.unlock()
                    }
                }
            }
        }
        #expect(log.lines.count == 40)
        #expect(log.lines.last == "[gap][SUMMARY] fixed 20, open 20, unknown 0")
    }

    @Test("an empty-body occurrence downgrades a FIXED name to UNKNOWN; OPEN still wins")
    func unknownMerging() {
        let t = ConformanceGapTracker()
        t.record("n", outcome: .fixed); t.record("n", outcome: .unknown)
        #expect(t.outcome("n") == .unknown)
        t.record("n", outcome: .open); t.record("n", outcome: .fixed)
        #expect(t.outcome("n") == .open)
    }
}
