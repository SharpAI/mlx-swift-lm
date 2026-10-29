#!/usr/bin/env bash
# run-blocked.sh [--self-test-only]
#
# SL-28 A4: prove that a CLEAN build plus the offline conformance tests work with outbound network BLOCKED.
# macOS only: uses sandbox-exec with a profile that denies network*. Prints a final  EXIT=<n>  line taken from the
# failing stage's own process status (not from a pipe).
#
# Stages (every stage after the self-test runs INSIDE the sandbox):
#   0. self-test   the sandbox really blocks: a local TCP listener is reachable outside the profile and NOT inside
#                  it; curl to 1.1.1.1 and `git ls-remote` of the SwiftPM remotes fail inside. If any of this does
#                  not hold the run stops with EXIT=70, so a later pass can never be vacuous.
#   1. resolve     `swift package resolve` into a fresh scratch dir. Only works if the SwiftPM cache
#                  (~/Library/Caches/org.swift.swiftpm/repositories) already holds the remotes: that is the
#                  "prefetch" a sealed CI leg does BEFORE the seal (see offline-build/README.md).
#   2. build       `swift build --build-tests` (clean: the scratch dir is removed first unless KEEP=1).
#   3. test        `swift test --skip-build --filter <f>` for each filter, with SWIFTLM_OFFLINE=1.
#
# Env:
#   SCRATCH       build dir (default: $TMPDIR/swiftlm-blocked-build). Removed first => clean build.
#   FILTERS       space-separated test filters (default: "ConformanceOfflineGuard ConformanceOfflineBundleTests")
#   LOCK_BUILD    optional command prefix for the build stages, e.g. "python3 with-lock.py build 3 --"
#   LOCK_GPU      optional command prefix for the test stage
#   KEEP=1        do not remove SCRATCH first (incremental)
#
# upstream-watch: if Package.swift gains a dependency that is not in the SwiftPM cache, stage 1 fails and names
# the URL it tried to fetch.
set -u

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd) || { echo "EXIT=65"; exit 65; }
SCRATCH=${SCRATCH:-${TMPDIR:-/tmp}/swiftlm-blocked-build}
FILTERS=${FILTERS:-"ConformanceOfflineGuard ConformanceOfflineBundleTests"}
LOCK_BUILD=${LOCK_BUILD:-}
LOCK_GPU=${LOCK_GPU:-}
PROFILE='(version 1)(allow default)(deny network*)'

finish() { echo "EXIT=$1"; exit "$1"; }

[ "$(uname -s)" = Darwin ] && command -v sandbox-exec >/dev/null || {
    echo "run-blocked: needs macOS sandbox-exec" >&2; finish 69; }

sb() { sandbox-exec -p "$PROFILE" "$@"; }

# ---- stage 0: self-test -------------------------------------------------------------------------------------
self_test() {
    local dir port pid ok=0
    dir=$(mktemp -d) || return 1
    # local listener: reachable from outside the sandbox, must be unreachable from inside
    python3 -I - "$dir" <<'PY' &
import socket, sys, threading
s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(8)
open(sys.argv[1] + "/port", "w").write(str(s.getsockname()[1]))
while True:
    c, _ = s.accept(); c.close()
PY
    pid=$!
    for _ in 1 2 3 4 5 6 7 8 9 10; do [ -s "$dir/port" ] && break; sleep 0.3; done
    port=$(cat "$dir/port" 2>/dev/null)
    probe='import socket,sys; s=socket.socket(); s.settimeout(3); s.connect(("127.0.0.1", int(sys.argv[1])))'
    if [ -z "$port" ]; then echo "[self-test] FAIL: listener did not start"; ok=1
    else
        if python3 -I -c "$probe" "$port" 2>/dev/null; then echo "[self-test] control: local connect OUTSIDE sandbox succeeds"
        else echo "[self-test] FAIL: control connect outside sandbox failed (test is vacuous)"; ok=1; fi
        if sb python3 -I -c "$probe" "$port" 2>/dev/null; then echo "[self-test] FAIL: connect INSIDE sandbox succeeded"; ok=1
        else echo "[self-test] local connect INSIDE sandbox is blocked (good)"; fi
    fi
    if sb curl -sS --max-time 5 -o /dev/null https://1.1.1.1 2>/dev/null; then
        echo "[self-test] FAIL: curl 1.1.1.1 inside sandbox succeeded"; ok=1
    else echo "[self-test] curl 1.1.1.1 inside sandbox fails (good)"; fi
    if GIT_TERMINAL_PROMPT=0 sb git ls-remote https://github.com/apple/swift-numerics >/dev/null 2>&1; then
        echo "[self-test] FAIL: git ls-remote inside sandbox succeeded"; ok=1
    else echo "[self-test] git ls-remote inside sandbox fails (good)"; fi
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -rf "$dir"
    return $ok
}

self_test || { echo "[run-blocked] self-test failed: refusing to claim anything"; finish 70; }
[ "${1:-}" = "--self-test-only" ] && finish 0

cd "$REPO" || finish 65
[ "${KEEP:-0}" = 1 ] || rm -rf "$SCRATCH"
mkdir -p "$SCRATCH" || finish 65
# --disable-sandbox: SwiftPM's own sandbox-exec cannot nest inside ours.
SW=(--scratch-path "$SCRATCH" --disable-sandbox)

echo "[run-blocked] stage 1: resolve (network blocked)"
sb swift package "${SW[@]}" resolve > "$SCRATCH/resolve.log" 2>&1; st=$?
cat "$SCRATCH/resolve.log"
[ $st -eq 0 ] || { echo "[run-blocked] resolve failed: dependency not in the SwiftPM cache (see log above)"; finish $st; }
if grep -E "^(Fetching|Updating) https" "$SCRATCH/resolve.log" | grep -v "from cache" ; then
    echo "[run-blocked] a dependency was fetched from the network, not the cache"; finish 71; fi

echo "[run-blocked] stage 2: clean build (network blocked)"
# shellcheck disable=SC2086
$LOCK_BUILD sandbox-exec -p "$PROFILE" swift build --build-tests "${SW[@]}" > "$SCRATCH/build.log" 2>&1; st=$?
tail -5 "$SCRATCH/build.log" | cut -c1-300
[ $st -eq 0 ] || { echo "[run-blocked] build failed, log: $SCRATCH/build.log"; finish $st; }

rc=0
for f in $FILTERS; do
    echo "[run-blocked] stage 3: test --filter $f (network blocked, SWIFTLM_OFFLINE=1)"
    # shellcheck disable=SC2086
    SWIFTLM_OFFLINE=1 HF_HUB_OFFLINE=1 $LOCK_GPU sandbox-exec -p "$PROFILE" \
        swift test --skip-build "${SW[@]}" --filter "$f" > "$SCRATCH/test-$f.log" 2>&1; st=$?
    grep -E "Test run with|Executed [0-9]+ tests" "$SCRATCH/test-$f.log" | tail -2
    echo "[run-blocked] $f exit=$st"
    [ $st -eq 0 ] || rc=$st
done
finish $rc
