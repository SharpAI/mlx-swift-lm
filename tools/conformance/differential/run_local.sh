#!/usr/bin/env bash
# LOCAL ONLY (CI is air-gapped and has no torch): gen -> swift test (named suite) -> compare.
#
#   tools/conformance/differential/run_local.sh [--n N] [--seed S] [--dir DIR] [--oracle PATH]
#   tools/conformance/differential/run_local.sh --swift-only          # only the Swift step, uses $KDA_DIFF_DIR (shrink calls this)
#
# Environment (all optional):
#   KDA_DIFF_PY          python with torch+numpy (default: <repo>/.venv-golden/bin/python, see gen_kda_golden.py HOW TO RUN)
#   KDA_DIFF_WRAP        words put in front of `swift test`, e.g. "python3 /path/with-lock.py gpu 1 --" (not quoted: word-split on purpose)
#   KDA_DIFF_SWIFT_ARGS  extra `swift test` arguments, e.g. "--skip-build" once the tests are built
# Exit code: the swift test process's own code if it is non-zero, else compare's (0 = every case within tolerance).
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
py="${KDA_DIFF_PY:-$root/.venv-golden/bin/python}"
n=60; seed=1; dir=""; oracle=""; swift_only=0
while [ $# -gt 0 ]; do
  case "$1" in
    --n) n="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --dir) dir="$2"; shift 2 ;;
    --oracle) oracle="$2"; shift 2 ;;
    --swift-only) swift_only=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

run_swift() {
  # $KDA_DIFF_WRAP and $KDA_DIFF_SWIFT_ARGS are deliberately unquoted (word splitting).
  # shellcheck disable=SC2086
  (cd "$root" && KDA_DIFF_DIR="$1" ${KDA_DIFF_WRAP:-} swift test ${KDA_DIFF_SWIFT_ARGS:-} --filter ConformanceKdaDifferentialTests)
}

# run_swift cd's to the repo root, so a relative directory would mean different places to python (caller's cwd) and swift.
abspath() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
[ -z "$dir" ] || dir="$(abspath "$dir")"

if [ "$swift_only" = 1 ]; then
  [ -n "${KDA_DIFF_DIR:-}" ] || { echo "--swift-only needs KDA_DIFF_DIR" >&2; exit 64; }
  run_swift "$(abspath "$KDA_DIFF_DIR")"
  exit $?
fi

[ -n "$dir" ] || dir="$(mktemp -d "${TMPDIR:-/tmp}/kda-diff.XXXXXX")"
echo "== gen: n=$n seed=$seed dir=$dir"
gen_args=(gen --out "$dir" --n "$n" --seed "$seed")
[ -z "$oracle" ] || gen_args+=(--oracle "$oracle")
"$py" -I "$here/kda_diff.py" "${gen_args[@]}" || { echo "gen failed" >&2; exit 2; }

echo "== swift test (ConformanceKdaDifferentialTests)"
run_swift "$dir"
swift_rc=$?
echo "SWIFT_EXIT=$swift_rc"

echo "== compare"
"$py" -I "$here/kda_diff.py" compare --dir "$dir"
cmp_rc=$?
echo "COMPARE_EXIT=$cmp_rc"

if [ "$swift_rc" -ne 0 ]; then exit "$swift_rc"; fi
exit "$cmp_rc"
