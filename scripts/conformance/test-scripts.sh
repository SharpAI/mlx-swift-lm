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

# 7. sentinel runs shadow-check against the fetched upstream tree (SL-30): a planted collision stops it with
#    SENTINEL_RESULT=COLLISION / EXIT=4 BEFORE any merge; a clean upstream lets it proceed to the merge step.
#    The "upstream" is a local git repo (SENTINEL_UPSTREAM_URL), so no network is used.
# The fixture keeps RELATED histories (repo is a clone of up, then diverges), so a clean upstream really merges.
#   mk_fixture <name> <upstream swift line> [after-expectations-text]  -> $tmp/<name>/{repo,up}
# With an after-expectations text the repo also gets stub gen_expectations.py / run-test.sh (recording what the
# sentinel runs, and with which CONFORMANCE_STRICT) so the whole pipeline after the merge can be observed.
exp_rel=Tests/MLXLMTests/Conformance/ConformanceExpectations.swift
mk_fixture() {
    local d="$tmp/$1"; mkdir -p "$d/up/Libraries/MLXLMCommon"
    printf 'public struct Base {}\n' >"$d/up/Libraries/MLXLMCommon/Base.swift"
    git init -q -b main "$d/up" && git -C "$d/up" add -A && git -C "$d/up" -c user.name=t -c user.email=t@t commit -q -m base
    git clone -q "$d/up" "$d/repo" && git -C "$d/repo" remote set-url origin https://example.invalid/x.git
    mkdir -p "$d/repo/Libraries/MLXLMCommon/SharpAIOverlay" "$d/repo/scripts/conformance/upstream" "$d/repo/Tests/MLXLMTests/Conformance"
    printf 'public enum SharpAIOverlay {}\n' >"$d/repo/Libraries/MLXLMCommon/SharpAIOverlay/H.swift"
    cp "$here/upstream/shadow-check.sh" "$d/repo/scripts/conformance/upstream/"
    : >"$d/repo/scripts/conformance/upstream/shadow-allowlist.txt"
    if [ -n "${3:-}" ]; then
        printf '%s\n' '        .init(modelType: "foo", registries: [.llm], archs: [], surveyStatus: "x"),' \
            '        .init(modelType: "baz", registries: [.llm], archs: [], surveyStatus: "x"),' >"$d/repo/$exp_rel"
        printf '%s\n' "$3" >"$d/repo/scripts/conformance/after.txt"
        printf '%s\n' '#!/usr/bin/env python3' 'import shutil,sys' \
            'shutil.copy(sys.argv[1] + "/scripts/conformance/after.txt", sys.argv[1] + "/'"$exp_rel"'")' >"$d/repo/scripts/conformance/gen_expectations.py"
        printf '%s\n' '#!/usr/bin/env bash' 'echo "$1 STRICT=${CONFORMANCE_STRICT:-unset}" >>"$RT_LOG"' \
            '[ "$1" = ConformanceGap ] && { echo "[gap][FIXED] demo"; echo "[gap][OPEN] other"; exit 1; }' 'exit 0' >"$d/repo/scripts/conformance/run-test.sh"
        chmod +x "$d/repo/scripts/conformance/gen_expectations.py" "$d/repo/scripts/conformance/run-test.sh"
    fi
    git -C "$d/repo" add -A && git -C "$d/repo" -c user.name=t -c user.email=t@t commit -q -m ours
    printf '%s\n' "$2" >"$d/up/Libraries/MLXLMCommon/Up.swift"
    git -C "$d/up" add -A && git -C "$d/up" -c user.name=t -c user.email=t@t commit -q -m up
}
tmpr=$(cd "$tmp" && pwd -P)
mk_fixture coll 'public enum SharpAIOverlay {}'
out=$(SENTINEL_SCRATCH_ROOTS="$tmpr" SENTINEL_UPSTREAM_URL="$tmp/coll/up" "$here/upstream/sentinel.sh" "$tmp/coll/repo" 2>&1); s=$?
{ [ "$s" -eq 4 ] && echo "$out" | grep -q "COLLISION SharpAIOverlay" && echo "$out" | grep -q "SENTINEL_RESULT=COLLISION" && [ "$(echo "$out" | tail -1)" = "EXIT=4" ]; }; check "sentinel: upstream name collision exits 4 (shadow-check wired in)" $?
git -C "$tmp/coll/repo" branch --list 'sentinel/*' | grep -q .; [ $? -ne 0 ]; check "sentinel: collision stops before creating the merge branch" $?
mk_fixture clean 'public struct Unrelated {}'
out=$(SENTINEL_SCRATCH_ROOTS="$tmpr" SENTINEL_UPSTREAM_URL="$tmp/clean/up" "$here/upstream/sentinel.sh" "$tmp/clean/repo" 2>&1); s=$?
{ echo "$out" | grep -q "SHADOW_CHECK=PASS" && ! echo "$out" | grep -q "SENTINEL_RESULT=COLLISION" && [ "$s" -ne 4 ]; }; check "sentinel: clean upstream passes shadow-check" $?
{ echo "$out" | grep -q "^== merge: clean" && git -C "$tmp/clean/repo" log --all --oneline | grep -q "sentinel: merge upstream main" \
  && git -C "$tmp/clean/repo" ls-tree -r --name-only "$(git -C "$tmp/clean/repo" branch --list 'sentinel/*' | head -1 | tr -d ' *')" 2>/dev/null | grep -q 'Up.swift'; }; check "sentinel: clean upstream really reaches and completes the merge (related histories)" $?

# registry classification (registry-diff.py): a registry that is only GAINED is LANDED, never REGRESSION.
rd="$here/upstream/registry-diff.py"
mk_exp() { printf '        .init(modelType: "%s", registries: [%s], archs: [], surveyStatus: "x"),\n' "$2" "$3" >>"$1"; }
: >"$tmp/rd-before.txt"; : >"$tmp/rd-after.txt"
mk_exp "$tmp/rd-before.txt" gain ".llm";        mk_exp "$tmp/rd-after.txt" gain ".llm, .vlm"
mk_exp "$tmp/rd-before.txt" swap ".llm";        mk_exp "$tmp/rd-after.txt" swap ".vlm"
mk_exp "$tmp/rd-before.txt" shrink ".llm, .vlm"; mk_exp "$tmp/rd-after.txt" shrink ".llm"
mk_exp "$tmp/rd-before.txt" gone ".llm"
mk_exp "$tmp/rd-before.txt" same ".llm";        mk_exp "$tmp/rd-after.txt" same ".llm"
mk_exp "$tmp/rd-after.txt" fresh ".llm";        mk_exp "$tmp/rd-before.txt" empty "";   mk_exp "$tmp/rd-after.txt" empty ".embedder"
out=$(python3 -I "$rd" "$tmp/rd-before.txt" "$tmp/rd-after.txt" 2>&1); s=$?
{ [ "$s" -eq 0 ] && echo "$out" | grep -q "^LANDED gain -> .llm, .vlm" && ! echo "$out" | grep -q "REGRESSION gain"; }; check "registry-diff: llm -> llm,vlm is LANDED, not REGRESSION" $?
{ echo "$out" | grep -q "^REGRESSION swap" && echo "$out" | grep -q "^REGRESSION shrink" && echo "$out" | grep -q "^REGRESSION gone"; }; check "registry-diff: swapped, shrunk and vanished registries are REGRESSION" $?
{ echo "$out" | grep -q "^LANDED fresh" && echo "$out" | grep -q "^LANDED empty" && ! echo "$out" | grep -q " same"; }; check "registry-diff: new/filled entries LANDED, unchanged silent" $?

# whole pipeline after the merge, with stub gen_expectations.py / run-test.sh / swift.
mkdir -p "$tmp/fbin"; printf '#!/usr/bin/env bash\nexit 0\n' >"$tmp/fbin/swift"; chmod +x "$tmp/fbin/swift"
mk_fixture gain 'public struct Unrelated2 {}' '        .init(modelType: "foo", registries: [.llm, .vlm], archs: [], surveyStatus: "x"),
        .init(modelType: "baz", registries: [.llm], archs: [], surveyStatus: "x"),'
out=$(PATH="$tmp/fbin:$PATH" RT_LOG="$tmp/gain.rtlog" SENTINEL_SCRATCH_ROOTS="$tmpr" SENTINEL_UPSTREAM_URL="$tmp/gain/up" "$here/upstream/sentinel.sh" "$tmp/gain/repo" 2>&1); s=$?
{ echo "$out" | grep -q "^  LANDED foo -> .llm, .vlm" && ! echo "$out" | grep -q "^ *REGRESSION foo"; }; check "sentinel: a gained registry is LANDED and not a REGRESSION line" $?
{ grep -q "^ConformanceRegistryTests STRICT=1$" "$tmp/gain.rtlog" && grep -q "^ConformanceForwardTests STRICT=1$" "$tmp/gain.rtlog" && grep -q "^ConformanceGap STRICT=1$" "$tmp/gain.rtlog"; }; check "sentinel: Registry, Forward and Gap suites all run with CONFORMANCE_STRICT=1" $?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "\[gap\]\[FIXED\] demo" && echo "$out" | grep -q "gaps_open=1" && echo "$out" | grep -q "SENTINEL_RESULT=FAIL"; }; check "sentinel: a FIXED gap is reported and fails the run" $?
mk_fixture shrink 'public struct Unrelated3 {}' '        .init(modelType: "foo", registries: [.llm], archs: [], surveyStatus: "x"),'
out=$(PATH="$tmp/fbin:$PATH" RT_LOG="$tmp/shrink.rtlog" SENTINEL_SCRATCH_ROOTS="$tmpr" SENTINEL_UPSTREAM_URL="$tmp/shrink/up" "$here/upstream/sentinel.sh" "$tmp/shrink/repo" 2>&1)
echo "$out" | grep -q "^  REGRESSION baz was .llm"; check "sentinel: a registry that vanished is a REGRESSION line" $?

# sentinel refusals end with SENTINEL_RESULT=REFUSED / EXIT=2 (run-periodic keys off it).
out=$(SENTINEL_SCRATCH_ROOTS=/nonexistent-root "$here/upstream/sentinel.sh" "$tmp/s/repo" 2>&1)
{ echo "$out" | grep -q "SENTINEL_RESULT=REFUSED" && [ "$(echo "$out" | tail -1)" = "EXIT=2" ]; }; check "sentinel: a refusal prints SENTINEL_RESULT=REFUSED and EXIT=2" $?

# the other two upstream self-tests belong to this suite (each prints its own PASS/FAIL).
for t in test-mlx-swift-shadow.sh test-run-periodic.sh test-shadow-check.sh; do
    [ -x "$here/upstream/$t" ] || { check "$t exists and is executable" 1; continue; }
    sub=$("$here/upstream/$t" 2>&1); s=$?
    echo "$sub" | sed 's/^/    | /'
    check "$t: all checks pass (exit $s)" "$s"
done

# 8. run-blocked.sh SCRATCH validation (it rm -rf's SCRATCH): no personal path baked in, root derived from RUNNER_TEMP /
#    TMPDIR, and the rm -rf safety rules still hold. `--validate-scratch` runs only the validation (no sandbox).
rb="$here/offline-build/run-blocked.sh"
vs() { env -u SCRATCH_ROOT "$@" bash "$rb" --validate-scratch >/dev/null 2>&1; }
mkdir -p "$tmp/rt" "$tmp/tmpdir"; rt=$(cd "$tmp/rt" && pwd -P); td=$(cd "$tmp/tmpdir" && pwd -P)
! grep -q "claude-501\|/scratchpad" "$rb"; check "run-blocked: no personal scratchpad path in the script" $?
vs RUNNER_TEMP="$rt" SCRATCH="$rt/swiftlm-blocked-build"; check "run-blocked: SCRATCH under RUNNER_TEMP/swiftlm-blocked* is accepted" $?
vs -u RUNNER_TEMP TMPDIR="$td" SCRATCH="$td/swiftlm-blocked-build"; check "run-blocked: SCRATCH under TMPDIR/swiftlm-blocked* is accepted" $?
vs RUNNER_TEMP="$rt" SCRATCH="$rt/other-dir"; [ $? -ne 0 ]; check "run-blocked: an arbitrary dir under RUNNER_TEMP is refused" $?
vs RUNNER_TEMP="$rt" SCRATCH="$rt"; [ $? -ne 0 ]; check "run-blocked: RUNNER_TEMP itself is refused" $?
vs RUNNER_TEMP="$rt" SCRATCH="/"; [ $? -ne 0 ]; check "run-blocked: / is refused" $?
vs RUNNER_TEMP="$rt" SCRATCH="$HOME"; [ $? -ne 0 ]; check "run-blocked: HOME is refused" $?
vs RUNNER_TEMP="$rt" SCRATCH="$here/../../swiftlm-blocked-x"; [ $? -ne 0 ]; check "run-blocked: a path with .. is refused" $?
vs RUNNER_TEMP="$rt" SCRATCH="$rt/anything" SCRATCH_ROOT="$rt"; check "run-blocked: an explicit SCRATCH_ROOT still allows a dir strictly inside it" $?
env RUNNER_TEMP="$rt" SCRATCH="$rt/anything" SCRATCH_ROOT="" bash "$rb" --validate-scratch >/dev/null 2>&1; [ $? -ne 0 ]; check "run-blocked: an empty SCRATCH_ROOT is not treated as the cwd" $?

# 9. run-blocked.sh stage-1 "fetched from the network" detector (--check-resolve-log <file>: exit 71 = network fetch).
#    Cache hits must pass under any SwiftPM wording that mentions the cache; a bare Fetching/Updating/Cloning https
#    line without it is flagged; unrelated output is ignored.
rb="$here/offline-build/run-blocked.sh"
cl() { printf '%s\n' "$1" >"$tmp/resolve.log"; bash "$rb" --check-resolve-log "$tmp/resolve.log" >/dev/null 2>&1; }
cl 'Fetching https://github.com/apple/swift-numerics from cache'; check "resolve-log: 'Fetching ... from cache' passes" $?
cl 'Fetched https://github.com/apple/swift-numerics from cache (0.41s)'; check "resolve-log: 'Fetched ... from cache (0.41s)' passes" $?
cl 'Fetching https://github.com/apple/swift-numerics (cached)'; check "resolve-log: a differently worded cache hit passes" $?
cl 'Computing version for https://github.com/apple/swift-numerics'; check "resolve-log: 'Computing version' is not a fetch" $?
cl 'Fetching https://github.com/apple/swift-numerics'; [ $? -eq 71 ]; check "resolve-log: a bare 'Fetching https' is flagged (71)" $?
cl 'Updating https://github.com/swiftlang/swift-syntax.git'; [ $? -eq 71 ]; check "resolve-log: a bare 'Updating https' is flagged (71)" $?
cl 'Cloning https://github.com/swiftlang/swift-syntax.git'; [ $? -eq 71 ]; check "resolve-log: a bare 'Cloning https' is flagged (71)" $?
cl '[1/1] Fetching https://github.com/apple/swift-numerics'; [ $? -eq 71 ]; check "resolve-log: a progress-prefixed fetch is flagged (71)" $?

# 10. ci.yml sealed-offline job (SL-28): pinned MLX wheel, no silent metallib copy, explicit PASS/FAIL line, still non-blocking.
ci="$here/../../.github/workflows/ci.yml"
sealed=$(awk '/^  sealed-offline:/{p=1} p' "$ci")
[ -n "$sealed" ]; check "ci.yml: sealed-offline job exists" $?
! printf '%s\n' "$sealed" | grep -Eq 'pip install[^\n]* mlx *$'; check "ci.yml sealed job: no unpinned 'pip install mlx'" $?
printf '%s\n' "$sealed" | grep -q 'mlx==\$MLX_VER'; check "ci.yml sealed job: mlx wheel pinned to the version of the mlx-swift checkout" $?
printf '%s\n' "$sealed" | grep -q 'MLX_VERSION_MAJOR'; check "ci.yml sealed job: version derived from mlx-swift's mlx/version.h" $?
! printf '%s\n' "$sealed" | grep -q '2>/dev/null || true'; check "ci.yml sealed job: no silent '2>/dev/null || true'" $?
printf '%s\n' "$sealed" | grep -q 'metallib not installed'; check "ci.yml sealed job: a metallib copy that installs nothing fails loudly" $?
printf '%s\n' "$sealed" | grep -q 'SEALED-OFFLINE: '; check "ci.yml sealed job: summary prints an explicit SEALED-OFFLINE: PASS/FAIL line" $?
printf '%s\n' "$sealed" | grep -q 'continue-on-error: true'; check "ci.yml sealed job: still continue-on-error (observation phase)" $?
printf '%s\n' "$sealed" | grep -q 'rev-parse HEAD'; check "ci.yml sealed job: records the mlx-swift revision it ran against" $?

exit "$fail"
