#!/usr/bin/env bash
# Self-test for v4-watch.sh (SL-42): clean tree, planted file, planted key, planted type, usage error.
# usage: scripts/conformance/upstream/test-v4-watch.sh   prints PASS/FAIL per check, exits non-zero on any FAIL.
set -u
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sc="${V4_WATCH_SCRIPT:-$here/v4-watch.sh}"
fail=0
check() { if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }
tmp=$(mktemp -d "${TMPDIR:-/tmp}/v4-watch-test.XXXXXX"); trap 'rm -rf "$tmp"' EXIT

clean="$tmp/clean/Libraries"; mkdir -p "$clean/MLXLLM/Models"
echo 'public struct Qwen3 {}' >"$clean/MLXLLM/Models/Qwen3.swift"
out=$("$sc" "$clean" 2>&1); rc=$?
[ "$rc" -eq 0 ] && echo "$out" | grep -q '^V4_RESULT=CLEAN$'; check "clean tree passes (exit 0, CLEAN)" $?

planted="$tmp/planted/Libraries"; mkdir -p "$planted/MLXLLM/Models"
echo 'public final class Foo {}' >"$planted/MLXLLM/Models/DeepseekV4.swift"
out=$("$sc" "$planted" 2>&1); rc=$?
[ "$rc" -eq 5 ] && echo "$out" | grep -q 'V4_ALERT file .*DeepseekV4.swift'; check "planted DeepseekV4*.swift alerts (exit 5)" $?

keyed="$tmp/keyed/Libraries"; mkdir -p "$keyed/MLXLLM"
echo 'let r: [String: Any] = ["deepseek_v4": x]' >"$keyed/MLXLLM/LLMModelFactory.swift"
out=$("$sc" "$keyed" 2>&1); rc=$?
[ "$rc" -eq 5 ] && echo "$out" | grep -q 'V4_ALERT key '; check "planted \"deepseek_v4\" key alerts (exit 5)" $?

typed="$tmp/typed/Libraries"; mkdir -p "$typed/MLXLLM/Models"
echo 'public struct DeepseekV4Model {}' >"$typed/MLXLLM/Models/Other.swift"
out=$("$sc" "$typed" 2>&1); rc=$?
[ "$rc" -eq 5 ] && echo "$out" | grep -q 'V4_ALERT type '; check "planted DeepseekV4 type alerts (exit 5)" $?

"$sc" "$tmp/missing" >/dev/null 2>&1; rc=$?
[ "$rc" -eq 2 ]; check "missing directory is a usage error (exit 2)" $?

[ "$fail" -eq 0 ] && echo "ALL PASS" || echo "SOME FAILED"
exit "$fail"
