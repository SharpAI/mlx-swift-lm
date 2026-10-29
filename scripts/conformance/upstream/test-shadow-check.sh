#!/usr/bin/env bash
# Self-test for shadow-check.sh (no swift, no network). Style mirrors ../test-scripts.sh.
# usage: scripts/conformance/upstream/test-shadow-check.sh   prints PASS/FAIL per check, exits non-zero on any FAIL.
set -u
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sc="$here/shadow-check.sh"
fail=0
check() { if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }
tmp=$(mktemp -d "${TMPDIR:-/tmp}/shadow-check-test.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

# Synthetic fork repo.
repo="$tmp/repo"; up="$tmp/upstream"
mkdir -p "$repo/Libraries/MLXLMCommon/SharpAIOverlay" "$repo/Libraries/MLXLLM" "$up/Libraries/MLXLMCommon"
cat >"$repo/Libraries/MLXLMCommon/SharpAIOverlay/Helper.swift" <<'S'
public enum SharpAIOverlay {
    public static func registerIfAbsent() {}
}
extension Foo {
    func sharpAIThing() {}
}
S
cat >"$repo/Libraries/MLXLLM/Drafter.swift" <<'S'
func go() async { await registry.registerModelType("x") { _ in 1 } }
S
cat >"$repo/Libraries/MLXLLM/Plain.swift" <<'S'
// await registry.registerModelType("commented") { _ in 1 }
public struct Plain {}
S
printf 'Libraries/MLXLLM/Drafter.swift\n' >"$tmp/allow.txt"
cat >"$up/Libraries/MLXLMCommon/Other.swift" <<'S'
public struct Unrelated {}
public func registerModelType(_ a: Int) {}
private enum SharpAIOverlay {}
S

run() { "$sc" --repo "$repo" --upstream "$up" --allowlist "$tmp/allow.txt" "$@" 2>&1; }

# 1. usage errors.
out=$("$sc" --repo "$repo" 2>&1); s=$?
{ [ "$s" -eq 2 ] && echo "$out" | grep -q "\-\-upstream"; }; check "shadow-check: missing --upstream exits 2" $?
out=$("$sc" --repo "$repo" --upstream "$tmp/nope" 2>&1); s=$?
[ "$s" -eq 2 ]; check "shadow-check: nonexistent upstream dir exits 2" $?

# 2. clean (private upstream decl of same name, comment-only and allow-listed calls are ignored).
out=$(run); s=$?
{ [ "$s" -eq 0 ] && echo "$out" | grep -q "SHADOW_CHECK=PASS"; }; check "shadow-check: clean tree passes" $?

# 3. planted type collision.
cp "$up/Libraries/MLXLMCommon/Other.swift" "$tmp/Other.bak"
echo 'public enum SharpAIOverlay { }' >>"$up/Libraries/MLXLMCommon/Other.swift"
out=$(run); s=$?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "COLLISION SharpAIOverlay" && echo "$out" | grep -q "SHADOW_CHECK=FAIL"; }; check "shadow-check: planted public type collision fails" $?

# 4. planted func collision with a sharpai-prefixed member name (non-public on our side, still a build break).
cp "$tmp/Other.bak" "$up/Libraries/MLXLMCommon/Other.swift"
printf 'extension Bar {\n    public func sharpAIThing() {}\n}\n' >>"$up/Libraries/MLXLMCommon/Other.swift"
out=$(run); s=$?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "COLLISION sharpAIThing"; }; check "shadow-check: planted member func collision fails" $?

# 5. planted collision with a public symbol declared in the overlay directory but not sharpai-prefixed.
cp "$tmp/Other.bak" "$up/Libraries/MLXLMCommon/Other.swift"
printf 'public final class Widget {}\n' >>"$repo/Libraries/MLXLMCommon/SharpAIOverlay/Helper.swift"
printf 'public final class Widget {}\n' >>"$up/Libraries/MLXLMCommon/Other.swift"
out=$(run); s=$?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "COLLISION Widget"; }; check "shadow-check: collision on non-prefixed public symbol in overlay dir fails" $?
cp "$tmp/Other.bak" "$up/Libraries/MLXLMCommon/Other.swift"
sed -i.bak '/Widget/d' "$repo/Libraries/MLXLMCommon/SharpAIOverlay/Helper.swift"; rm -f "$repo/Libraries/MLXLMCommon/SharpAIOverlay/Helper.swift.bak"

# 6. new file outside the allow-list calling registerModelType.
cat >"$repo/Libraries/MLXLLM/NewReg.swift" <<'S'
func reg() async {
    await ModelTypeRegistry.shared.registerModelType("new_model") { _ in 1 }
}
S
out=$(run); s=$?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "UNREGISTERED_CALL Libraries/MLXLLM/NewReg.swift"; }; check "shadow-check: non-allow-listed registerModelType call fails" $?

# 7. allow-listing it makes it pass; using registerIfAbsent instead needs no allow-list entry.
echo 'Libraries/MLXLLM/NewReg.swift' >>"$tmp/allow.txt"
out=$(run); s=$?
[ "$s" -eq 0 ]; check "shadow-check: allow-listed file passes" $?
sed '$d' "$tmp/allow.txt" >"$tmp/allow2.txt" && mv "$tmp/allow2.txt" "$tmp/allow.txt"
printf 'func reg() async { _ = await SharpAIOverlay.registerIfAbsent(registry: r, type: "t") { _ in 1 } }\n' >"$repo/Libraries/MLXLLM/NewReg.swift"
out=$(run); s=$?
[ "$s" -eq 0 ]; check "shadow-check: registerIfAbsent use needs no allow-list entry" $?

# 8. the real repo against an empty upstream tree must be clean (guards the committed allow-list).
mkdir -p "$tmp/empty"
out=$("$sc" --upstream "$tmp/empty" 2>&1); s=$?
{ [ "$s" -eq 0 ] && echo "$out" | grep -q "SHADOW_CHECK=PASS"; }; check "shadow-check: real repo + empty upstream passes with committed allow-list" $?

exit "$fail"
