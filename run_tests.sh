#!/bin/bash
# Run MLX Swift LM tests with Metal library properly located.
# Swift test runner needs mlx.metallib next to the test binary.

set -e

FILTER="${1:-}"
METALLIB="${METALLIB_PATH:-/Users/simba/SwiftLM/default.metallib}"

echo "Building..."
swift build

# Find the test binary and copy metallib next to it
TEST_BINARY_DIR=$(swift build --show-bin-path 2>/dev/null)
if [ -z "$TEST_BINARY_DIR" ]; then
    TEST_BINARY_DIR=".build/arm64-apple-macosx/debug"
fi

XCTEST_DIR="${TEST_BINARY_DIR}/mlx-swift-lmPackageTests.xctest/Contents/MacOS"
if [ -d "$XCTEST_DIR" ] && [ -f "$METALLIB" ]; then
    cp "$METALLIB" "$XCTEST_DIR/mlx.metallib"
    echo "Copied metallib to $XCTEST_DIR"
fi

if [ -n "$FILTER" ]; then
    swift test --filter "$FILTER"
else
    swift test
fi
