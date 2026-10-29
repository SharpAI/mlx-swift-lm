#!/usr/bin/env bash
# Self-test for run-test.sh and upstream/sentinel.sh safety (no swift build, no network, no GPU).
# usage: scripts/conformance/test-scripts.sh      prints PASS/FAIL per check, exits non-zero on any FAIL.
set -u
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fail=0
check() { # check <name> <condition-exit-status>
    if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi
}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/conf-scripts-test.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

# 1. run-test.sh: no argument -> usage error, EXIT=64.
out=$("$here/run-test.sh" 2>/dev/null); s=$?
[ "$s" -eq 64 ] && [ "$(echo "$out" | tail -1)" = "EXIT=64" ]; check "run-test: no-arg gives EXIT=64" $?

# 2. run-test.sh reports swift's own exit code (stub swift exits 7), as the LAST line, and shows the filter.
mkdir -p "$tmp/bin"
cat >"$tmp/bin/swift" <<'STUB'
#!/usr/bin/env bash
echo "stub swift $*"
exit 7
STUB
chmod +x "$tmp/bin/swift"
out=$(PATH="$tmp/bin:$PATH" "$here/run-test.sh" Tests/MLXLMTests/Conformance/ConformanceRegistryTests.swift 2>&1); s=$?
[ "$s" -eq 7 ]; check "run-test: process exit status is swift's (7)" $?
[ "$(echo "$out" | tail -1)" = "EXIT=7" ]; check "run-test: last line is EXIT=7 (not masked by a pipe)" $?
echo "$out" | grep -q "stub swift test --filter ConformanceRegistryTests$"; check "run-test: file path reduced to --filter basename" $?

# 3. sentinel refuses outside a scratch root.
mkdir -p "$tmp/s" && git init -q "$tmp/s/repo" && git -C "$tmp/s/repo" remote add origin https://example.invalid/x.git
out=$(SENTINEL_SCRATCH_ROOTS=/nonexistent-root "$here/upstream/sentinel.sh" "$tmp/s/repo" 2>&1); s=$?
{ [ "$s" -eq 2 ] && echo "$out" | grep -q "not under a scratch root"; }; check "sentinel: refuses non-scratch clone" $?

# 4. sentinel refuses a dirty tree.
touch "$tmp/s/repo/untracked"
out=$(SENTINEL_SCRATCH_ROOTS="$tmp:$(cd "$tmp" && pwd -P)" "$here/upstream/sentinel.sh" "$tmp/s/repo" 2>&1); s=$?
{ [ "$s" -eq 2 ] && echo "$out" | grep -q "uncommitted or untracked"; }; check "sentinel: refuses dirty tree" $?
rm "$tmp/s/repo/untracked"

# 5. sentinel refuses an origin that is a local non-scratch path.
mkdir -p "$tmp/real" && git -C "$tmp/s/repo" remote set-url origin "$tmp/real"
out=$(SENTINEL_SCRATCH_ROOTS="$(cd "$tmp/s" && pwd -P)" "$here/upstream/sentinel.sh" "$tmp/s/repo" 2>&1); s=$?
{ [ "$s" -eq 2 ] && echo "$out" | grep -q "local, non-scratch"; }; check "sentinel: refuses local non-scratch origin" $?

# 6. a failed refusal must not have touched remotes (no push URL rewritten before the checks).
[ "$(git -C "$tmp/s/repo" remote get-url --push origin)" = "$tmp/real" ]; check "sentinel: refusals leave remotes untouched" $?

exit "$fail"
