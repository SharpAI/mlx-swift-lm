#!/usr/bin/env bash
# Self-test for mlx-swift-shadow.sh (no swift, no network): planted collisions, overlaps, allow-list, usage.
# usage: scripts/conformance/upstream/test-mlx-swift-shadow.sh   prints PASS/FAIL per check, exits non-zero on any FAIL.
set -u
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sc="${MLX_SHADOW_SCRIPT:-$here/mlx-swift-shadow.sh}"
fail=0
check() { if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }
tmp=$(mktemp -d "${TMPDIR:-/tmp}/mlx-shadow-test.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
repo="$tmp/repo"; mlx="$tmp/mlx"
mkdir -p "$repo/Libraries/MLXLMCommon" "$mlx/Source/MLX"
: >"$tmp/allow.txt"
run() { "$sc" --repo "$repo" --mlx-swift "$mlx" --allowlist "$tmp/allow.txt" 2>&1; }

cat >"$repo/Libraries/MLXLMCommon/Ours.swift" <<'S'
import MLX
public struct Mine {
    public func compile(a: Int) {}          // method of OUR type: never a shadow
}
func internalOnly() {}
public func helperFree(x: Int, y: Int) {}
public extension MLXArray {
    func zzNormalize(axis: Int = -1) -> MLXArray { self }
}
public func compile(
    shapeless: Bool = false,
    _ body: @escaping @Sendable (MLXArray) -> MLXArray
) -> Int { 0 }
S
cat >"$mlx/Source/MLX/Theirs.swift" <<'S'
public func helperFree(x: Int, y: Int, z: Int) {}   // different labels, not call compatible: no finding
public func unrelated(a: Int) {}
public struct Mine { public func compile(a: Int) {} }
S

out=$("$sc" --repo "$repo" 2>&1); s=$?
{ [ "$s" -eq 2 ] && echo "$out" | grep -q "\-\-mlx-swift"; }; check "mlx-swift-shadow: missing --mlx-swift exits 2" $?
out=$("$sc" --repo "$repo" --mlx-swift "$tmp" 2>&1); s=$?
{ [ "$s" -eq 2 ] && echo "$out" | grep -q "Source/"; }; check "mlx-swift-shadow: non-mlx-swift dir exits 2" $?

out=$(run); s=$?
{ [ "$s" -eq 0 ] && echo "$out" | grep -q "MLX_SWIFT_SHADOW=PASS"; }; check "mlx-swift-shadow: clean trees pass (own-type method, internal and non-matching labels ignored)" $?

# planted exact collision on a free func.
printf 'public func helperFree(x: Int, y: Int) {}\n' >>"$mlx/Source/MLX/Theirs.swift"
out=$(run); s=$?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "COLLISION .helperFree(x:y:)" && echo "$out" | grep -q "MLX_SWIFT_SHADOW=FAIL"; }; check "mlx-swift-shadow: planted free-func collision fails" $?

# allow-listing it passes and still prints it.
echo '.helperFree(x:y:)  # test reason' >"$tmp/allow.txt"
out=$(run); s=$?
{ [ "$s" -eq 0 ] && echo "$out" | grep -q "ALLOWED COLLISION .helperFree(x:y:)"; }; check "mlx-swift-shadow: allow-listed collision passes but stays visible" $?
: >"$tmp/allow.txt"
sed -i.bak '$d' "$mlx/Source/MLX/Theirs.swift"; rm -f "$mlx/Source/MLX/Theirs.swift.bak"

# planted extension-member collision (the l2Normalized class): ours extends MLXArray, theirs too.
printf 'public extension MLXArray {\n    func zzNormalize(axis: Int = -1) -> MLXArray { self }\n}\n' >>"$mlx/Source/MLX/Theirs.swift"
out=$(run); s=$?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "COLLISION MLXArray.zzNormalize(axis:)"; }; check "mlx-swift-shadow: planted extension-member collision fails" $?
sed '$d' "$mlx/Source/MLX/Theirs.swift" | sed '$d' | sed '$d' >"$tmp/t.swift" && mv "$tmp/t.swift" "$mlx/Source/MLX/Theirs.swift"

# the compile(shapeless:) incident: theirs adds defaulted inputs:/outputs: before shapeless: -> OVERLAP.
cat >"$mlx/Source/MLX/Compile.swift" <<'S'
public func compile(
    inputs: [Int] = [], outputs: [Int] = [], shapeless: Bool = false,
    _ f: @escaping (MLXArray) -> MLXArray
) -> Int { 0 }
S
out=$(run); s=$?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "OVERLAP .compile(shapeless:_:)"; }; check "mlx-swift-shadow: compile(shapeless:) vs defaulted inputs:/outputs: overload is an OVERLAP" $?

# the same upstream overload WITHOUT defaults is not call-compatible with ours: no finding.
cat >"$mlx/Source/MLX/Compile.swift" <<'S'
public func compile(
    inputs: [Int], outputs: [Int], shapeless: Bool = false,
    _ f: @escaping (MLXArray) -> MLXArray
) -> Int { 0 }
S
out=$(run); s=$?
{ [ "$s" -eq 0 ] && ! echo "$out" | grep -q "compile"; }; check "mlx-swift-shadow: required extra parameters do not overlap" $?

# multi-line signature with exact labels collides too.
cat >"$mlx/Source/MLX/Compile.swift" <<'S'
public func compile(
    shapeless: Bool = false,
    _ f: @escaping @Sendable (MLXArray) -> MLXArray
) -> Int { 0 }
S
out=$(run); s=$?
{ [ "$s" -eq 1 ] && echo "$out" | grep -q "COLLISION .compile(shapeless:_:)"; }; check "mlx-swift-shadow: multi-line exact-label collision fails" $?

# the committed repo against the committed allow-list is covered by the periodic script test (needs a real mlx-swift tree).
exit "$fail"
