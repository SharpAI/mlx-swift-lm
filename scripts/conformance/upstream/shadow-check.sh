#!/usr/bin/env bash
# shadow-check.sh -- overlay-vs-upstream shadowing check (SharpAI/SwiftLM#254, SL-30).
#
# usage: scripts/conformance/upstream/shadow-check.sh --upstream DIR [--repo DIR] [--allowlist FILE]
#   --upstream DIR   a checkout of ml-explore/mlx-swift-lm (read only; never modified)
#   --repo DIR       the fork checkout (default: the clone holding this script)
#   --allowlist FILE files allowed to call registerModelType directly (default: shadow-allowlist.txt next to this script)
#
# Fails (exit 1) when
#   1. COLLISION: a declaration of ours (a sharpai-prefixed name anywhere in Libraries/, case-insensitive, any
#      access level except private/fileprivate; or any public declaration under Libraries/**/SharpAIOverlay/)
#      is also declared (non-private) anywhere in the upstream tree. Upstream adding such a name would break
#      our build (same-named type/func/member), exactly the mlx-swift#28 failure mode.
#   2. UNREGISTERED_CALL: a .swift file under Libraries/ outside the allow-list and outside SharpAIOverlay/
#      calls registerModelType( (comments and `func` declarations ignored) instead of SharpAIOverlay.registerIfAbsent.
# Exit 0 = PASS, 1 = FAIL, 2 = usage / environment error. Last line is SHADOW_CHECK=PASS|FAIL.
# No network, no swift, no writes outside a private mktemp dir.
set -u
die() { echo "shadow-check: $*" >&2; exit 2; }

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
repo=$here/../../..; upstream=""; allow=$here/shadow-allowlist.txt
while [ $# -gt 0 ]; do
    case $1 in
        --upstream) upstream=${2:-}; shift 2 || die "--upstream needs a value" ;;
        --repo) repo=${2:-}; shift 2 || die "--repo needs a value" ;;
        --allowlist) allow=${2:-}; shift 2 || die "--allowlist needs a value" ;;
        *) die "unknown argument: $1" ;;
    esac
done
[ -n "$upstream" ] || die "--upstream DIR is required"
[ -d "$upstream" ] || die "upstream directory not found: $upstream"
repo=$(cd "$repo" 2>/dev/null && pwd -P) || die "repo directory not found"
upstream=$(cd "$upstream" && pwd -P)
[ -d "$repo/Libraries" ] || die "$repo has no Libraries/"
[ -f "$allow" ] || die "allow-list not found: $allow"

work=$(mktemp -d "${TMPDIR:-/tmp}/shadow-check.XXXXXX") || die "mktemp failed"
trap 'rm -rf "$work"' EXIT

# scan <root>: one awk pass per file so file names are known.
scan() { # scan <root> -> lines: name \t relfile \t ispublic
    local root=$1 f rel
    find "$root" -name '*.swift' -not -path '*/.build/*' -not -path '*/Tests/*' | sort | while IFS= read -r f; do
        rel=${f#"$root"/}
        awk -v rel="$rel" '
            /^[[:space:]]*\/\// { next }
            {
                line = $0
                if (line ~ /(^|[[:space:]])(private|fileprivate)([[:space:]]|\()/) next
                if (match(line, /(^|[^A-Za-z0-9_])(enum|struct|class|actor|protocol|typealias|func|var|let)[[:space:]]+`?[A-Za-z_][A-Za-z0-9_]*/)) {
                    s = substr(line, RSTART, RLENGTH)
                    sub(/^[^a-z]*/, "", s)                      # strip leading delimiter
                    n = split(s, w, /[[:space:]]+/)
                    name = w[2]; gsub(/`/, "", name)
                    kw = w[1]
                    # reject keyword used inside an expression, e.g. "let" after "if"/"guard", or a bare "class" modifier
                    if (line ~ /(^|[[:space:]])(if|guard|while|for|case|where)[[:space:]]+(let|var)/) next
                    pub = (line ~ /(^|[[:space:]])(public|open)([[:space:]]|$)/) ? "public" : "other"
                    printf "%s\t%s\t%s\n", name, rel, pub
                }
            }' "$f"
    done
}

scan "$repo/Libraries" >"$work/ours-all.tsv"
if [ -d "$upstream/Libraries" ]; then scan "$upstream/Libraries" >"$work/up-all.tsv"; else scan "$upstream" >"$work/up-all.tsv"; fi

# Ours: sharpai-prefixed (any level), plus public declarations under a SharpAIOverlay/ directory.
awk -F'\t' 'tolower($1) ~ /^sharpai/ || ($3 == "public" && $2 ~ /(^|\/)SharpAIOverlay\//) { print $1 "\t" $2 }' \
    "$work/ours-all.tsv" | sort -u >"$work/ours.tsv"
cut -f1 "$work/up-all.tsv" | sort -u >"$work/up-names.txt"

status=0
echo "== shadow-check: ours=$(cut -f1 "$work/ours.tsv" | sort -u | wc -l | tr -d ' ') overlay/sharpai names, upstream=$(wc -l <"$work/up-names.txt" | tr -d ' ') declared names"
while IFS="$(printf '\t')" read -r name file; do
    if grep -qxF -- "$name" "$work/up-names.txt"; then
        up_file=$(awk -F'\t' -v n="$name" '$1 == n { print $2; exit }' "$work/up-all.tsv")
        echo "COLLISION $name  (ours: $file; upstream: $up_file)"
        status=1
    fi
done <"$work/ours.tsv"

# Direct registerModelType calls outside the allow-list.
grep -v -E '^[[:space:]]*(#|$)' "$allow" | sed 's/[[:space:]]*$//' >"$work/allow.txt"
while IFS= read -r f; do
    rel=${f#"$repo"/}
    case $rel in Libraries/*/SharpAIOverlay/*) continue ;; esac
    grep -qxF -- "$rel" "$work/allow.txt" && continue
    n=$(grep -v -E '^[[:space:]]*//' "$f" | grep -v 'func registerModelType' | grep -c 'registerModelType(')
    if [ "$n" -gt 0 ]; then
        echo "UNREGISTERED_CALL $rel  ($n direct registerModelType call(s); use SharpAIOverlay.registerIfAbsent or add to the allow-list with a reason)"
        status=1
    fi
done < <(find "$repo/Libraries" -name '*.swift' | sort)

if [ "$status" -eq 0 ]; then echo "SHADOW_CHECK=PASS"; else echo "SHADOW_CHECK=FAIL"; fi
exit "$status"
