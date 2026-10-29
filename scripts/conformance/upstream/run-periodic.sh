#!/usr/bin/env bash
# run-periodic.sh -- one command for the 2-day upstream check (SharpAI/SwiftLM#254, SL-30).
# It does NOT schedule itself and creates no scheduled task; whoever owns the check runs it (by hand or from
# their own scheduler) and reads the last lines.
#
# Runs, in order, against the CURRENT ml-explore main of both upstreams, inside scratch clones:
#   1. fetch ml-explore/mlx-swift into --mlx-dir (read-only; shallow) and run mlx-swift-shadow.sh on the fork
#   2. upstream/sentinel.sh on the fork clone: shadow-check vs ml-explore/mlx-swift-lm, merge on a throwaway
#      branch, regenerate expectations, build, conformance tests with CONFORMANCE_STRICT=1
#
# usage: scripts/conformance/upstream/run-periodic.sh [--lm-clone DIR] [--mlx-dir DIR]
#   --lm-clone DIR  scratch clone of the fork on the branch to check (default: the clone holding this script);
#                   sentinel.sh refuses a clone that is not under a scratch root, a dirty tree, or a real working tree
#   --mlx-dir DIR   scratch directory holding the ml-explore/mlx-swift checkout (created on first use;
#                   default: <lm-clone>/../up-mlx)
# Env: SENTINEL_LOCK="python3 /path/with-lock.py"   passed through to sentinel.sh
#      RUN_PERIODIC_MLX_URL (default https://github.com/ml-explore/mlx-swift.git)
#      RUN_PERIODIC_SENTINEL (default: upstream/sentinel.sh next to this script; tests substitute a stub)
#
# Output is plain ASCII. Exit 0 = nothing to act on (LANDED lines are information, not failure).
# Exit 1 = act: a REGRESSION, a COLLISION or OVERLAP, a merge CONFLICT, or a failed build/test.
# Exit 2 = could not run (bad arguments, fetch failed, clone refused). Last line: EXIT=<n>.
set -u
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
lm=$here/../../..; mlxdir=""
fatal() { echo "run-periodic: $*"; echo "PERIODIC_RESULT=ERROR"; echo "EXIT=2"; exit 2; }
while [ $# -gt 0 ]; do
    case $1 in
        --lm-clone) lm=${2:-}; shift 2 || fatal "--lm-clone needs a value" ;;
        --mlx-dir) mlxdir=${2:-}; shift 2 || fatal "--mlx-dir needs a value" ;;
        *) fatal "unknown argument: $1" ;;
    esac
done
lm=$(cd "$lm" 2>/dev/null && pwd -P) || fatal "no such fork clone: $lm"
[ -n "$mlxdir" ] || mlxdir=$lm/../up-mlx
mlx_url=${RUN_PERIODIC_MLX_URL:-https://github.com/ml-explore/mlx-swift.git}
sentinel=${RUN_PERIODIC_SENTINEL:-$here/sentinel.sh}
work=$(mktemp -d "${TMPDIR:-/tmp}/run-periodic.XXXXXX") || fatal "mktemp failed"
trap 'rm -rf "$work"' EXIT
plain() { LC_ALL=C sed $'s/\x1b\\[[0-9;]*[A-Za-z]//g' | LC_ALL=C tr -cd '\11\12\15\40-\176'; }

echo "== run-periodic: $(date -u +%Y-%m-%dT%H:%M:%SZ) fork clone $lm"

# 1. mlx-swift: scratch checkout of current main, never pushed to.
echo "== mlx-swift: fetching ml-explore main"
if [ ! -d "$mlxdir/.git" ]; then
    mkdir -p "$mlxdir" || fatal "cannot create $mlxdir"
    git -C "$mlxdir" init -q || fatal "git init failed in $mlxdir"
    git -C "$mlxdir" remote add origin "$mlx_url" || fatal "cannot add remote"
fi
git -C "$mlxdir" remote set-url origin "$mlx_url"
git -C "$mlxdir" remote set-url --push origin "DISABLED-by-run-periodic"
git -C "$mlxdir" fetch -q --depth 1 origin main >"$work/mlx-fetch.log" 2>&1 || { plain <"$work/mlx-fetch.log"; fatal "fetch of mlx-swift main failed"; }
git -C "$mlxdir" checkout -q --force FETCH_HEAD >>"$work/mlx-fetch.log" 2>&1 || { plain <"$work/mlx-fetch.log"; fatal "checkout of mlx-swift FETCH_HEAD failed"; }
mlx_sha=$(git -C "$mlxdir" rev-parse HEAD)
echo "   mlx-swift main @ $mlx_sha"

shadow_exit=0
"$here/mlx-swift-shadow.sh" --repo "$lm" --mlx-swift "$mlxdir" >"$work/mlx-shadow.log" 2>&1 || shadow_exit=$?
plain <"$work/mlx-shadow.log" | sed 's/^/  /'
[ "$shadow_exit" -le 1 ] || { echo "PERIODIC_RESULT=ERROR"; echo "EXIT=2"; exit 2; }

# 2. sentinel on the fork clone (shadow-check, merge, build, tests).
echo "== sentinel"
sentinel_exit=0
"$sentinel" "$lm" >"$work/sentinel.log" 2>&1 || sentinel_exit=$?
plain <"$work/sentinel.log" | sed 's/^/  /'

landed=$(plain <"$work/sentinel.log" | grep -c -E '^ +LANDED ')
regression=$(plain <"$work/sentinel.log" | grep -c -E '^ +REGRESSION |\[conformance\]\[REGRESSION\]')
collision=$(plain <"$work/sentinel.log" | grep -c -E '^ +(COLLISION|UNREGISTERED_CALL) ')
conflict=$(plain <"$work/sentinel.log" | grep -c -E '^ +CONFLICT ')
lm_sha=$(plain <"$work/sentinel.log" | sed -n 's/.*upstream @ \([0-9a-f]*\).*/\1/p' | head -1)

echo "== summary"
echo "mlx_swift_sha=$mlx_sha"
echo "lm_upstream_sha=${lm_sha:-unknown}"
echo "mlx_swift_shadow_exit=$shadow_exit"
echo "sentinel_exit=$sentinel_exit"
echo "landed=$landed regression=$regression collision=$collision conflict=$conflict"
if [ "$shadow_exit" -ne 0 ] || [ "$sentinel_exit" -ne 0 ] || [ "$regression" -gt 0 ] || [ "$collision" -gt 0 ] || [ "$conflict" -gt 0 ]; then
    echo "PERIODIC_RESULT=FAIL"; echo "EXIT=1"; exit 1
fi
echo "PERIODIC_RESULT=PASS"; echo "EXIT=0"; exit 0
