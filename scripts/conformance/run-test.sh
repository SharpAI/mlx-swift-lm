#!/usr/bin/env bash
# run-test.sh <TestFilterOrFile> [extra swift-test args...]
#
# Runs `swift test --filter <name>` for ONE named test (or test file/suite) and nothing else,
# then prints a final line  EXIT=<exit code of the swift test process itself>.
# The exit code is captured from swift, not from tee/tail/grep, so a pipe cannot mask a failure.
#
# <TestFilterOrFile> may be a swift-test filter (ConformanceRegistryTests, or
# MLXLMTests.ConformanceRegistryTests/registriesMatchExpectations) or a path such as
# Tests/MLXLMTests/Conformance/ConformanceRegistryTests.swift (reduced to its basename).
set -u

if [ $# -lt 1 ] || [ -z "$1" ]; then
    echo "usage: $0 <TestFilterOrFile> [extra swift-test args...]" >&2
    echo "EXIT=64"
    exit 64
fi

filter=$1
shift
case $filter in
    *.swift)
        filter=$(basename "$filter" .swift)
        ;;
esac

# Run from the repository root regardless of the caller's cwd.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || { echo "EXIT=65"; exit 65; }

# No pipe: swift's own status is what we report.
swift test --filter "$filter" "$@"
status=$?
echo "EXIT=${status}"
exit "$status"
