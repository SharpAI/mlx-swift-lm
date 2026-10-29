// Known-gap wrapper shared by the ConformanceGap* suites.
//
// Normal run: a gap is `withKnownIssue(isIntermittent: true)` -- green while the gap exists AND
// green once it is fixed, so an upstream sync that implements it is never blocked.
// Raw run (`CONFORMANCE_GAP_RAW=1`): the wrapper is skipped and the same assertions run bare, so
// the red state can be inspected (and re-demonstrated) at any time without editing code.
// upstream-watch: none (SharpAI-only helper).

import Foundation
import Testing

enum ConformanceGap {
    static var raw: Bool { ProcessInfo.processInfo.environment["CONFORMANCE_GAP_RAW"] == "1" }

    static func knownIssue(
        _ comment: Comment, sourceLocation: SourceLocation = #_sourceLocation,
        _ body: () throws -> Void
    ) rethrows {
        if raw {
            try body()
        } else {
            withKnownIssue(comment, isIntermittent: true, sourceLocation: sourceLocation) { try body() }
        }
    }
}
