#!/usr/bin/env bash
# run-blocked.sh [--self-test-only | --validate-scratch]
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
#   SCRATCH       build dir (default: ${RUNNER_TEMP:-$TMPDIR}/swiftlm-blocked-build). Removed first => clean build.
#   FILTERS       space-separated test filters (default: "ConformanceOfflineGuard ConformanceOfflineBundleTests")
#   LOCK_BUILD    optional command prefix for the build stages, e.g. "python3 with-lock.py build 3 --"
#   LOCK_GPU      optional command prefix for the test stage
#   KEEP=1        do not remove SCRATCH first (incremental)
#   POST_BUILD    optional shell snippet run INSIDE the sandbox after the build and before the tests, with SCRATCH
#                 exported (CI uses it to copy mlx.metallib next to the test binaries)
#   TEST_ARGS     extra arguments for `swift test` (CI: --no-parallel, GPU tests are not safe to overlap)
#
# upstream-watch: if Package.swift gains a dependency that is not in the SwiftPM cache, stage 1 fails and names
# the URL it tried to fetch.
set -u

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd) || { echo "EXIT=65"; exit 65; }
TMP_BASE=${RUNNER_TEMP:-${TMPDIR:-/tmp}}
SCRATCH=${SCRATCH:-$TMP_BASE/swiftlm-blocked-build}
FILTERS=${FILTERS:-"ConformanceOfflineGuard ConformanceOfflineBundleTests"}
LOCK_BUILD=${LOCK_BUILD:-}
LOCK_GPU=${LOCK_GPU:-}
POST_BUILD=${POST_BUILD:-}
TEST_ARGS=${TEST_ARGS:-}
PROFILE='(version 1)(allow default)(deny network*)'

finish() { echo "EXIT=$1"; exit "$1"; }

# SCRATCH is rm -rf'd below, so it must be a path we can prove is disposable: a swiftlm-blocked* dir directly under the
# runner/system temp dir (RUNNER_TEMP on GitHub Actions, else TMPDIR, else /tmp; the default SCRATCH is one), or strictly
# inside an explicit SCRATCH_ROOT the caller names (empty by default: no personal path is baked in). Never /, $HOME,
# the repo or any parent of them.
SCRATCH_ROOT=${SCRATCH_ROOT:-}
rp() { python3 -I -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"; }
validate_scratch() {
    local p=$1 real root_real repo_real home_real tmp_real
    case "$p" in /*) ;; *) return 1 ;; esac
    case "$p" in *..*) return 1 ;; esac
    real=$(rp "$p") || return 1          # resolves symlinks, so a link cannot smuggle us out
    repo_real=$(rp "$REPO"); home_real=$(rp "$HOME"); tmp_real=$(rp "$TMP_BASE")
    [ "$real" = / ] && return 1
    [ "$real" = "$home_real" ] && return 1
    [ "$real" = "$repo_real" ] && return 1
    case "$real/" in "$repo_real"/*) return 1 ;; esac      # inside the repo
    case "$repo_real/" in "$real"/*) return 1 ;; esac      # a parent of the repo
    if [ -n "$SCRATCH_ROOT" ]; then                         # explicit root: strictly inside it
        root_real=$(rp "$SCRATCH_ROOT") || return 1
        case "$real" in "$root_real"/?*) return 0 ;; esac
    fi
    case "$real" in "$tmp_real"/swiftlm-blocked*|/private/tmp/swiftlm-blocked*) return 0 ;; esac   # mktemp-style dir
    return 1
}

if [ "${1:-}" = "--validate-scratch" ]; then      # test hook: only the rm -rf safety check, no sandbox, no build
    validate_scratch "$SCRATCH" && finish 0 || finish 64
fi

[ "$(uname -s)" = Darwin ] && command -v sandbox-exec >/dev/null || {
    echo "run-blocked: needs macOS sandbox-exec" >&2; finish 69; }

sb() { sandbox-exec -p "$PROFILE" "$@"; }

# ---- stage 0: self-test -------------------------------------------------------------------------------------
# Every inside-the-sandbox probe has a POSITIVE CONTROL: the same probe outside the sandbox must succeed (so the
# tool exists and the target is reachable), and inside it must fail with the sandbox's own error, not just "any
# failure" (a missing tool or a dead uplink would also fail). If the host has no uplink the external probes
# (curl 1.1.1.1, git ls-remote) cannot have a control; that is printed as INCONCLUSIVE and the loopback probes
# (python connect, curl to the loopback HTTP listener) carry the proof alone. Nothing is claimed beyond that.
self_test() {
    local dir port pid ok=0 err rc
    dir=$(mktemp -d) || return 1
    # local listener answering HTTP: reachable from outside the sandbox, must be unreachable from inside
    python3 -I - "$dir" <<'PY' &
import socket, sys
s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(8)
open(sys.argv[1] + "/port", "w").write(str(s.getsockname()[1]))
while True:
    c, _ = s.accept()
    try:
        c.settimeout(1); c.recv(4096); c.sendall(b"HTTP/1.0 200 OK\r\nContent-Length: 0\r\n\r\n")
    except Exception:
        pass
    c.close()
PY
    pid=$!
    for _ in 1 2 3 4 5 6 7 8 9 10; do [ -s "$dir/port" ] && break; sleep 0.3; done
    port=$(cat "$dir/port" 2>/dev/null)
    probe='import socket,sys; s=socket.socket(); s.settimeout(3); s.connect(("127.0.0.1", int(sys.argv[1])))'
    for t in python3 curl git; do
        command -v "$t" >/dev/null || { echo "[self-test] FAIL: $t not installed (probes would be vacuous)"; ok=1; }
    done
    if [ -z "$port" ]; then echo "[self-test] FAIL: listener did not start"; ok=1
    else
        # python connect: control outside, then inside must raise PermissionError / EPERM specifically
        if python3 -I -c "$probe" "$port" 2>/dev/null; then echo "[self-test] control: python connect OUTSIDE sandbox succeeds"
        else echo "[self-test] FAIL: control connect outside sandbox failed (test is vacuous)"; ok=1; fi
        err=$(sb python3 -I -c "$probe" "$port" 2>&1); rc=$?
        if [ $rc -eq 0 ]; then echo "[self-test] FAIL: connect INSIDE sandbox succeeded"; ok=1
        elif printf '%s' "$err" | grep -q "PermissionError.*Operation not permitted"; then
            echo "[self-test] python connect INSIDE sandbox: blocked by sandbox (EPERM) (good)"
        else echo "[self-test] FAIL: inside connect failed but NOT with the sandbox's EPERM: $(printf '%s' "$err" | tail -1)"; ok=1; fi
        # curl to the loopback HTTP listener: control outside (exit 0), inside must be exit 7 (connect refused/denied)
        if curl -sS --max-time 5 -o /dev/null "http://127.0.0.1:$port/" 2>/dev/null; then
            echo "[self-test] control: curl to loopback OUTSIDE sandbox succeeds"
            sb curl -sS --max-time 5 -o /dev/null "http://127.0.0.1:$port/" >/dev/null 2>&1; rc=$?
            if [ $rc -eq 7 ]; then echo "[self-test] curl loopback INSIDE sandbox: connect denied, exit 7 (good)"
            else echo "[self-test] FAIL: curl loopback inside sandbox exit=$rc (expected 7)"; ok=1; fi
        else echo "[self-test] FAIL: control curl to loopback outside sandbox failed (test is vacuous)"; ok=1; fi
    fi
    # external probes: only meaningful when the host has an uplink (outside control succeeds)
    if curl -sS --max-time 5 -o /dev/null https://1.1.1.1 2>/dev/null; then
        sb curl -sS --max-time 5 -o /dev/null https://1.1.1.1 >/dev/null 2>&1; rc=$?
        if [ $rc -eq 7 ] || [ $rc -eq 6 ]; then echo "[self-test] control ok (uplink up); curl 1.1.1.1 inside sandbox fails with exit $rc (good)"
        else echo "[self-test] FAIL: curl 1.1.1.1 inside sandbox exit=$rc (0 = reached the net, else tool problem)"; ok=1; fi
    else echo "[self-test] INCONCLUSIVE: no uplink outside the sandbox; curl 1.1.1.1 gives no evidence, loopback probes only"; fi
    if GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/apple/swift-numerics >/dev/null 2>&1; then
        err=$(GIT_TERMINAL_PROMPT=0 sb git ls-remote https://github.com/apple/swift-numerics 2>&1 >/dev/null); rc=$?
        if [ $rc -ne 0 ] && printf '%s' "$err" | grep -qE "Could not resolve host|unable to access|Operation not permitted"; then
            echo "[self-test] control ok (git works outside); git ls-remote inside sandbox fails with a network error (good)"
        else echo "[self-test] FAIL: git ls-remote inside sandbox: rc=$rc err=$(printf '%s' "$err" | tail -1)"; ok=1; fi
    else echo "[self-test] INCONCLUSIVE: git ls-remote also fails outside the sandbox (no uplink); no evidence from git"; fi
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; rm -rf "$dir"
    return $ok
}

self_test || { echo "[run-blocked] self-test failed: refusing to claim anything"; finish 70; }
[ "${1:-}" = "--self-test-only" ] && finish 0

cd "$REPO" || finish 65
validate_scratch "$SCRATCH" || { echo "run-blocked: refusing SCRATCH=$SCRATCH (must be a swiftlm-blocked* dir under $TMP_BASE, or strictly inside SCRATCH_ROOT if set; never /, HOME or the repo)" >&2; finish 64; }
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

if [ -n "$POST_BUILD" ]; then
    echo "[run-blocked] post-build hook (network blocked)"
    SCRATCH="$SCRATCH" sandbox-exec -p "$PROFILE" bash -c "$POST_BUILD" || { echo "[run-blocked] post-build hook failed"; finish 73; }
fi

rc=0
for f in $FILTERS; do
    echo "[run-blocked] stage 3: test --filter $f (network blocked, SWIFTLM_OFFLINE=1)"
    # shellcheck disable=SC2086
    SWIFTLM_OFFLINE=1 HF_HUB_OFFLINE=1 $LOCK_GPU sandbox-exec -p "$PROFILE" \
        swift test --skip-build "${SW[@]}" $TEST_ARGS --filter "$f" > "$SCRATCH/test-$f.log" 2>&1; st=$?
    grep -E "Test run with|Executed [0-9]+ tests" "$SCRATCH/test-$f.log" | tail -2
    echo "[run-blocked] $f exit=$st"
    [ $st -eq 0 ] || rc=$st
    # A filter that matches nothing makes swift test exit 0 ("Test run with 0 tests"): that is a vacuous pass.
    # Require N>0 from the swift-testing summary line, or XCTest "Executed N tests" with N>0.
    n=$(grep -E "Test run with [0-9]+ test" "$SCRATCH/test-$f.log" | tail -1 | sed -E 's/.*Test run with ([0-9]+) test.*/\1/')
    x=$(grep -E "Executed [0-9]+ tests?," "$SCRATCH/test-$f.log" | tail -1 | sed -E 's/.*Executed ([0-9]+) tests?,.*/\1/')
    if [ "${n:-0}" -gt 0 ] 2>/dev/null || [ "${x:-0}" -gt 0 ] 2>/dev/null; then
        echo "[run-blocked] $f ran tests (swift-testing=${n:-none} xctest=${x:-none})"
    else
        echo "[run-blocked] $f matched ZERO tests: vacuous pass refused"; [ $rc -eq 0 ] && rc=72
    fi
done
finish $rc
