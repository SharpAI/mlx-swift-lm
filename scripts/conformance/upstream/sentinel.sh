#!/usr/bin/env bash
# sentinel.sh -- upstream sentinel for the conformance harness (SharpAI/SwiftLM#254).
#
# Merges ml-explore/mlx-swift-lm main into a THROWAWAY local branch of a SCRATCH clone, regenerates
# the expectations, runs the conformance tests with CONFORMANCE_STRICT=1 and prints a short report:
#   conflicts (files), LANDED, REGRESSION, EXIT.
#
# usage: scripts/conformance/upstream/sentinel.sh [scratch-clone-dir]   (default: the clone holding this script)
#
# Safe by construction:
#   * refuses unless the clone lives under a scratch root (SENTINEL_SCRATCH_ROOTS, colon separated;
#     default /private/tmp:/tmp:/private/var/folders) and, when origin is a local path, that path is
#     scratch too (a clone of your real working tree is refused);
#   * refuses a dirty tree (uncommitted or untracked changes);
#   * never pushes: every remote gets push URL "DISABLED-by-sentinel" before anything else happens;
#   * merge conflicts are reported, never resolved (the merge is aborted).
# Optional env: SENTINEL_LOCK="python3 /path/with-lock.py"  (prefix is used as: $SENTINEL_LOCK build 3 -- ... / gpu 1 -- ...)
#               SENTINEL_UPSTREAM_URL (default https://github.com/ml-explore/mlx-swift-lm.git)
set -u

die() { echo "sentinel: REFUSED: $*" >&2; exit 2; }

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
repo=${1:-$here/../../..}
repo=$(cd "$repo" 2>/dev/null && pwd -P) || die "no such directory: ${1:-}"
git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$repo is not a git work tree"
repo=$(git -C "$repo" rev-parse --show-toplevel) && repo=$(cd "$repo" && pwd -P)

roots=${SENTINEL_SCRATCH_ROOTS:-/private/tmp:/tmp:/private/var/folders}
in_scratch() { # in_scratch <abs path>
    local p=$1 r IFS=:
    for r in $roots; do
        r=${r%/}
        case $p in "$r"/*) return 0 ;; esac
    done
    return 1
}
in_scratch "$repo" || die "$repo is not under a scratch root ($roots)"

origin=$(git -C "$repo" remote get-url origin 2>/dev/null) || die "clone has no origin"
case $origin in
    /* | file://* | ./* | ../*)
        o=${origin#file://}
        o=$(cd "$o" 2>/dev/null && pwd -P) || die "origin $origin is a local path that does not resolve"
        in_scratch "$o" || die "origin $origin is a local, non-scratch path"
        ;;
esac
[ -z "$(git -C "$repo" status --porcelain --untracked-files=all)" ] \
    || die "uncommitted or untracked changes in $repo"

# Never push, from any remote, from this point on.
for r in $(git -C "$repo" remote); do
    git -C "$repo" remote set-url --push "$r" "DISABLED-by-sentinel" || die "cannot disable push for $r"
done

work=$(mktemp -d "${TMPDIR:-/tmp}/sentinel.XXXXXX") || die "mktemp failed"
upstream_url=${SENTINEL_UPSTREAM_URL:-https://github.com/ml-explore/mlx-swift-lm.git}
if git -C "$repo" remote get-url upstream >/dev/null 2>&1; then
    git -C "$repo" remote set-url upstream "$upstream_url"
else
    git -C "$repo" remote add upstream "$upstream_url"
fi
git -C "$repo" remote set-url --push upstream "DISABLED-by-sentinel"

# Merges need an identity, set locally only if missing (never global).
git -C "$repo" config user.name >/dev/null || git -C "$repo" config user.name "Simba Zhang"
git -C "$repo" config user.email >/dev/null || git -C "$repo" config user.email "solderzzc@gmail.com"

echo "== sentinel: fetching upstream main"
git -C "$repo" fetch --quiet upstream main || die "fetch of upstream main failed"
base_branch=$(git -C "$repo" rev-parse --abbrev-ref HEAD)
branch="sentinel/upstream-$(date +%Y%m%d-%H%M%S)"
git -C "$repo" checkout -q -b "$branch" || die "cannot create $branch"
echo "== sentinel: throwaway branch $branch (from $base_branch @ $(git -C "$repo" rev-parse --short HEAD)), upstream @ $(git -C "$repo" rev-parse --short FETCH_HEAD)"

conflicts=""
merge_status=0
git -C "$repo" merge --no-edit --no-ff FETCH_HEAD -m "sentinel: merge upstream main" >"$work/sentinel-merge.log" 2>&1 || merge_status=$?
if [ "$merge_status" -ne 0 ]; then
    conflicts=$(git -C "$repo" diff --name-only --diff-filter=U)
    git -C "$repo" merge --abort >/dev/null 2>&1
    echo "== MERGE CONFLICTS (not resolved, merge aborted):"
    if [ -n "$conflicts" ]; then echo "$conflicts" | sed 's/^/  CONFLICT /'; else echo "  (merge failed without conflict files; see $work/sentinel-merge.log)"; fi
    echo "== LANDED: n/a (merge not applied)"
    echo "== REGRESSION: n/a (merge not applied)"
    echo "SENTINEL_RESULT=CONFLICT"
    echo "EXIT=3"
    exit 3
fi
echo "== merge: clean"

exp=Tests/MLXLMTests/Conformance/ConformanceExpectations.swift
python3 "$repo/scripts/conformance/gen_expectations.py" "$repo" || die "gen_expectations.py failed"

# Static LANDED / REGRESSION: compare each model_type's registries before and after regeneration.
parse() { sed -n 's/.*modelType: "\([^"]*\)", registries: \[\([^]]*\)\].*/\1 \2/p'; }
git -C "$repo" show "HEAD:$exp" | parse | sort >"$work/exp-before.txt"
parse <"$repo/$exp" | sort >"$work/exp-after.txt"
echo "== LANDED (registry grew vs committed expectations):"
landed=0
while IFS= read -r l; do
    echo "  LANDED $l"; landed=$((landed + 1))
done < <(comm -13 "$work/exp-before.txt" "$work/exp-after.txt" | awk '{n=$1; $1=""; if ($0 !~ /^ *$/) print n " ->" $0}')
[ "$landed" -eq 0 ] && echo "  (none)"
echo "== REGRESSION (registry shrank vs committed expectations):"
reg=0
while IFS= read -r l; do
    echo "  REGRESSION $l"; reg=$((reg + 1))
done < <(comm -23 "$work/exp-before.txt" "$work/exp-after.txt" | awk '{n=$1; $1=""; if ($0 !~ /^ *$/) print n " was" $0}')
[ "$reg" -eq 0 ] && echo "  (none)"
echo "   (static diff; entries whose registries changed appear in both lists as old/new)"

lock_build=(); lock_gpu=()
if [ -n "${SENTINEL_LOCK:-}" ]; then
    read -r -a lk <<<"$SENTINEL_LOCK"
    lock_build=("${lk[@]}" build 3 --); lock_gpu=("${lk[@]}" gpu 1 --)
fi

echo "== sentinel: building tests"
${lock_build[@]+"${lock_build[@]}"} bash -c 'cd "$1" && swift build --build-tests' _ "$repo" >"$work/sentinel-build.log" 2>&1
build_status=$?
echo "   build exit=$build_status (log: $work/sentinel-build.log)"
if [ "$build_status" -ne 0 ]; then
    grep -E "error:" "$work/sentinel-build.log" | head -20
    echo "SENTINEL_RESULT=BUILD_FAILED"
    echo "EXIT=$build_status"
    exit "$build_status"
fi

echo "== sentinel: conformance tests (CONFORMANCE_STRICT=1)"
export CONFORMANCE_STRICT=1
test_status=0
for f in ConformanceRegistryTests ConformanceForwardTests; do
    ${lock_gpu[@]+"${lock_gpu[@]}"} "$repo/scripts/conformance/run-test.sh" "$f" >"$work/sentinel-$f.log" 2>&1
    s=$?
    echo "   $f exit=$s"
    grep -E "\[conformance\]|REGRESSION|LANDED|Test run with|error:" "$work/sentinel-$f.log" | head -40 | sed 's/^/     /'
    [ "$s" -ne 0 ] && test_status=$s
done
echo "SENTINEL_RESULT=$([ "$test_status" -eq 0 ] && echo PASS || echo FAIL)"
echo "EXIT=$test_status"
exit "$test_status"
