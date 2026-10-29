#!/usr/bin/env bash
# Self-test for run-periodic.sh (no swift, no network, no GPU): the sentinel is replaced by a stub
# (RUN_PERIODIC_SENTINEL) and "ml-explore/mlx-swift" by a local git repo (RUN_PERIODIC_MLX_URL).
# usage: scripts/conformance/upstream/test-run-periodic.sh   prints PASS/FAIL per check, exits non-zero on any FAIL.
set -u
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rp="${RUN_PERIODIC_SCRIPT:-$here/run-periodic.sh}"
fail=0
check() { if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }
tmp=$(cd "$(mktemp -d "${TMPDIR:-/tmp}/run-periodic-test.XXXXXX")" && pwd -P); trap 'rm -rf "$tmp"' EXIT

# fork clone fixture (needs Libraries/ for mlx-swift-shadow) and a local "mlx-swift" upstream.
mkdir -p "$tmp/lm/Libraries/MLXLMCommon" "$tmp/lm/scripts/conformance/upstream" "$tmp/mlxsrc/Source/MLX"
cp "$here/mlx-swift-shadow.sh" "$tmp/lm/scripts/conformance/upstream/"; : >"$tmp/lm/scripts/conformance/upstream/mlx-swift-shadow-allowlist.txt"
printf 'public func helperFree(x: Int) {}\n' >"$tmp/lm/Libraries/MLXLMCommon/Ours.swift"
printf 'public func other(x: Int) {}\n' >"$tmp/mlxsrc/Source/MLX/T.swift"
git init -q -b main "$tmp/mlxsrc" && git -C "$tmp/mlxsrc" add -A && git -C "$tmp/mlxsrc" -c user.name=t -c user.email=t@t commit -q -m m

stub() { # stub <exit> <extra stdout lines...>
    local code=$1; shift
    { echo '#!/usr/bin/env bash'; echo 'echo "== sentinel: throwaway branch x (from b @ 1111111), upstream @ abc1234"'
      for l in "$@"; do printf 'printf "%%s\\n" %q\n' "$l"; done
      echo "echo EXIT=$code"; echo "exit $code"; } >"$tmp/stub.sh"; chmod +x "$tmp/stub.sh"
}
export SENTINEL_SCRATCH_ROOTS="$tmp"
run() { RUN_PERIODIC_SENTINEL="$tmp/stub.sh" RUN_PERIODIC_MLX_URL="$tmp/mlxsrc" "$rp" --lm-clone "$tmp/lm" --mlx-dir "$tmp/mlxclone" 2>&1; }

stub 0 '== LANDED (registry grew vs committed expectations):' '  LANDED foo -> .llm' '== REGRESSION (registry shrank vs committed expectations):' '  (none)' 'SENTINEL_RESULT=PASS'
out=$(run); s=$?
{ [ "$s" -eq 0 ] && echo "$out" | grep -q "PERIODIC_RESULT=PASS" && echo "$out" | grep -q "landed=1" && echo "$out" | grep -q "regression=0"; }; check "run-periodic: clean run exits 0 (LANDED alone is not a failure)" $?
[ "$(echo "$out" | tail -1)" = "EXIT=0" ]; check "run-periodic: last line is EXIT=0" $?
! printf '%s' "$out" | LC_ALL=C grep -q '[^[:print:][:space:]]'; check "run-periodic: report is plain ASCII" $?

stub 1 '== REGRESSION (registry shrank vs committed expectations):' '  REGRESSION bar was .llm' 'SENTINEL_RESULT=FAIL'
out=$(run); s=$?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "PERIODIC_RESULT=FAIL" && echo "$out" | grep -q "regression=1"; }; check "run-periodic: REGRESSION line exits 1" $?

stub 0 '  REGRESSION baz was .llm' 'SENTINEL_RESULT=PASS'
out=$(run); s=$?
[ "$s" -eq 1 ]; check "run-periodic: REGRESSION line fails even if the stub exits 0" $?

stub 4 '  COLLISION SharpAIOverlay  (ours: a; upstream: b)' 'SENTINEL_RESULT=COLLISION'
out=$(run); s=$?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "sentinel_exit=4"; }; check "run-periodic: sentinel collision (EXIT=4) exits 1" $?

stub 3 '  CONFLICT Package.swift' 'SENTINEL_RESULT=CONFLICT'
out=$(run); s=$?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "CONFLICT Package.swift"; }; check "run-periodic: merge conflict exits 1 and lists the file" $?

# mlx-swift side: planted collision in the fork makes it fail although the sentinel is clean.
stub 0 'SENTINEL_RESULT=PASS'
printf 'public func other(x: Int) {}\n' >>"$tmp/lm/Libraries/MLXLMCommon/Ours.swift"
out=$(run); s=$?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "COLLISION .other(x:)" && echo "$out" | grep -q "mlx_swift_shadow_exit=1"; }; check "run-periodic: mlx-swift collision exits 1" $?

# environment error: fork clone missing.
out=$(RUN_PERIODIC_SENTINEL="$tmp/stub.sh" RUN_PERIODIC_MLX_URL="$tmp/mlxsrc" "$rp" --lm-clone "$tmp/nope" --mlx-dir "$tmp/mlxclone" 2>&1); s=$?
[ "$s" -eq 2 ]; check "run-periodic: missing clone exits 2" $?

# gap fixed under CONFORMANCE_STRICT=1 is reported and fails.
stub 1 '     [gap][FIXED] softcap' 'SENTINEL_RESULT=FAIL'
out=$(run); s=$?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "gap_fixed=1"; }; check "run-periodic: [gap][FIXED] is counted and exits 1" $?

# sentinel exit 2 (REFUSED / environment error) maps to run-periodic exit 2, not 1.
stub 2 'sentinel: REFUSED: dirty tree' 'SENTINEL_RESULT=REFUSED'
out=$(run); s=$?
{ [ "$s" -eq 2 ] && echo "$out" | grep -q "PERIODIC_RESULT=ERROR" && [ "$(echo "$out" | tail -1)" = "EXIT=2" ]; }; check "run-periodic: sentinel exit 2 (refused) exits 2" $?
# ... but a swift process that failed with status 2 after printing a verdict is a failure to act on (1).
stub 2 'SENTINEL_RESULT=BUILD_FAILED'
out=$(run); s=$?
[ "$s" -eq 1 ]; check "run-periodic: build failure that happens to exit 2 still exits 1" $?

# scratch rule: refuse non-scratch --mlx-dir / --lm-clone BEFORE any fetch, checkout --force or set-url.
stub 0 'SENTINEL_RESULT=PASS'
mkdir -p "$tmp/real-mlx" && git init -q -b main "$tmp/real-mlx" && echo keep >"$tmp/real-mlx/mine.txt" \
    && git -C "$tmp/real-mlx" add -A && git -C "$tmp/real-mlx" -c user.name=t -c user.email=t@t commit -q -m mine \
    && git -C "$tmp/real-mlx" remote add origin https://example.invalid/real.git && echo dirty >>"$tmp/real-mlx/mine.txt"
mkdir -p "$tmp/scratch-only"
out=$(SENTINEL_SCRATCH_ROOTS="$tmp/scratch-only" RUN_PERIODIC_SENTINEL="$tmp/stub.sh" RUN_PERIODIC_MLX_URL="$tmp/mlxsrc" "$rp" --lm-clone "$tmp/scratch-only" --mlx-dir "$tmp/real-mlx" 2>&1); s=$?
{ [ "$s" -eq 2 ] && echo "$out" | grep -q "not under a scratch root" && [ "$(echo "$out" | tail -1)" = "EXIT=2" ]; }; check "run-periodic: refuses a non-scratch --mlx-dir (exit 2)" $?
{ [ "$(git -C "$tmp/real-mlx" remote get-url origin)" = "https://example.invalid/real.git" ] && [ "$(tail -1 "$tmp/real-mlx/mine.txt")" = "dirty" ] \
  && ! git -C "$tmp/real-mlx" remote get-url --push origin | grep -q DISABLED; }; check "run-periodic: refused --mlx-dir was not fetched into, forced or re-pointed" $?
out=$(SENTINEL_SCRATCH_ROOTS="$tmp/scratch-only" RUN_PERIODIC_SENTINEL="$tmp/stub.sh" RUN_PERIODIC_MLX_URL="$tmp/mlxsrc" "$rp" --lm-clone "$tmp/lm" --mlx-dir "$tmp/scratch-only/mlx" 2>&1); s=$?
{ [ "$s" -eq 2 ] && echo "$out" | grep -q "fork clone" && [ ! -e "$tmp/scratch-only/mlx" ]; }; check "run-periodic: refuses a non-scratch --lm-clone before touching mlx-dir" $?
# a symlink into a non-scratch place does not smuggle a path through.
mkdir -p "$tmp/scratch-only/ln" && ln -s "$tmp/real-mlx" "$tmp/scratch-only/ln/mlx"
out=$(SENTINEL_SCRATCH_ROOTS="$tmp/scratch-only" RUN_PERIODIC_SENTINEL="$tmp/stub.sh" RUN_PERIODIC_MLX_URL="$tmp/mlxsrc" "$rp" --lm-clone "$tmp/scratch-only" --mlx-dir "$tmp/scratch-only/ln/mlx" 2>&1); s=$?
[ "$s" -eq 2 ]; check "run-periodic: --mlx-dir symlinked out of scratch is refused" $?

# it must not schedule anything.
! grep -n -E 'crontab|launchctl|CronCreate|schedule_task|\bat -f' "$here/run-periodic.sh" | grep -v '^[0-9]*:#'; check "run-periodic: creates no scheduled task" $?
exit "$fail"
