#!/usr/bin/env bash
# mlx-swift-shadow.sh -- does a public func of ours shadow / overlap one in ml-explore/mlx-swift?
# (SharpAI/SwiftLM#254, SL-30; the mlx-swift#28 incident class: our `compile(shapeless:)` shims vs mlx-swift
# adding its own `compile` overloads.)  upstream-watch: new file, no upstream file is touched.
#
# usage: scripts/conformance/upstream/mlx-swift-shadow.sh --mlx-swift DIR [--repo DIR] [--allowlist FILE]
#   --mlx-swift DIR  a checkout of ml-explore/mlx-swift (read only; Source/ is scanned, minus Source/Cmlx)
#   --repo DIR       the fork checkout (default: the clone holding this script); its Libraries/ is scanned
#   --allowlist FILE default: mlx-swift-shadow-allowlist.txt next to this script. One entry per line:
#                    "<KIND> <Context>.<name>(<labels>)  # reason"  e.g.  "OVERLAP .compile(shapeless:_:)  # reason"
#                    KIND is COLLISION or OVERLAP and must match the finding: an allowed OVERLAP never silences a
#                    COLLISION of the same key (list both lines to allow both). A line without a KIND is an error.
#
# A declaration is keyed as Context.name(label:label:...). Context = enclosing extension/type name, empty for free funcs.
#   COLLISION  same key (name + argument labels) declared public by both sides.
#   OVERLAP    same context and name, and every call written against ours also type-checks against theirs
#              (our labels are an in-order subsequence of theirs and the extra parameters of theirs all have defaults).
#              Such a call can become ambiguous when both modules are imported, which is how mlx-swift#28 broke.
# Both fail the check (exit 1) unless allow-listed. Exit 0 = PASS, 2 = usage/environment error.
# Last line: MLX_SWIFT_SHADOW=PASS|FAIL.  No network, no swift, no writes outside a private mktemp dir.
# Limits (text scan, not the compiler): attributes/where-clauses are ignored, nested contexts use the innermost
# type name, and a same-key pair in different modules that the compiler can tell apart by closure type is still
# reported (allow-list it with the reason). It over-reports rather than misses.
set -u
die() { echo "mlx-swift-shadow: $*" >&2; exit 2; }
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
repo=$here/../../..; mlx=""; allow=$here/mlx-swift-shadow-allowlist.txt
while [ $# -gt 0 ]; do
    case $1 in
        --mlx-swift) mlx=${2:-}; shift 2 || die "--mlx-swift needs a value" ;;
        --repo) repo=${2:-}; shift 2 || die "--repo needs a value" ;;
        --allowlist) allow=${2:-}; shift 2 || die "--allowlist needs a value" ;;
        *) die "unknown argument: $1" ;;
    esac
done
[ -n "$mlx" ] || die "--mlx-swift DIR is required"
[ -d "$mlx" ] || die "mlx-swift directory not found: $mlx"
repo=$(cd "$repo" 2>/dev/null && pwd -P) || die "repo directory not found"
mlx=$(cd "$mlx" && pwd -P)
[ -d "$repo/Libraries" ] || die "$repo has no Libraries/"
[ -d "$mlx/Source" ] || die "$mlx has no Source/ (not an mlx-swift checkout)"
[ -f "$allow" ] || die "allow-list not found: $allow"
work=$(mktemp -d "${TMPDIR:-/tmp}/mlx-swift-shadow.XXXXXX") || die "mktemp failed"
trap 'rm -rf "$work"' EXIT

# Emits TSV: context \t name \t labels(comma sep) \t defaults(comma sep 0/1) \t relfile
cat >"$work/scan.awk" <<'AWK'
function flush_sig(   s, i, c, d, p, parts, n, lab, def, a, k, names, depth2, cur, started, out_l, out_d, w, nw) {
    s = sig; sig = ""
    sub(/^.*func[ \t]+`?/, "", s)
    nm = s; sub(/[^A-Za-z0-9_`].*$/, "", nm); gsub(/`/, "", nm)
    sub(/^[^(]*/, "", s)                      # drop name and generic clause up to the first "("
    gsub(/->/, "→", s)
    d = 0; cur = ""; n = 0
    for (i = 1; i <= length(s); i++) {
        c = substr(s, i, 1)
        if (c == "(" || c == "[" || c == "{" || c == "<") { d++; if (d == 1 && c == "(") continue }
        else if (c == ")" || c == "]" || c == "}" || c == ">") { d--; if (d == 0) break }
        if (d == 1 && c == ",") { parts[++n] = cur; cur = ""; continue }
        cur = cur c
    }
    if (cur ~ /[^ \t]/) parts[++n] = cur
    out_l = ""; out_d = ""
    for (k = 1; k <= n; k++) {
        p = parts[k]; colon = index(p, ":")
        head = (colon > 0) ? substr(p, 1, colon - 1) : p
        gsub(/^[ \t]+|[ \t]+$/, "", head); nw = split(head, w, /[ \t]+/)
        lab = w[1]
        rest = (colon > 0) ? substr(p, colon + 1) : ""
        def = (rest ~ /=/) ? 1 : 0
        out_l = out_l (k > 1 ? "," : "") lab; out_d = out_d (k > 1 ? "," : "") def
    }
    # ours: a method of our OWN type cannot shadow anything; only free funcs and extension members can
    if (mode == "ours" && top > 0 && ctxkind[top] != "extension") return
    printf "%s\t%s\t%s\t%s\t%s\n", ctx_name(), nm, out_l, out_d, rel
}
function ctx_name() { return (top > 0) ? ctxs[top] : "" }
function braces(line,   t, o, c) { t = line; gsub(/"([^"\\]|\\.)*"/, "", t); sub(/\/\/.*$/, "", t)
    o = gsub(/\{/, "{", t); c = gsub(/\}/, "}", t); return o - c }
BEGIN { depth = 0; top = 0; insig = 0; sig = ""; pend = "" }
{
    line = $0
    if (line ~ /^[ \t]*\/\//) next
    if (insig) {
        sig = sig " " line
        if (line ~ /\{|→|->|^[ \t]*$/ || balanced(sig)) { flush_sig(); insig = 0 }
    } else if (line ~ /(^|[ \t])(public|open)[ \t]/ && line ~ /(^|[ \t])func[ \t]+`?[A-Za-z_]/ && line !~ /(^|[ \t])(private|fileprivate|internal|package)[ \t]/) {
        # a protocol requirement or witness: still a public name, keep it
        sig = line
        if (balanced(sig) && (line ~ /\{|->/ || line ~ /\)[ \t]*$/)) { flush_sig() } else { insig = 1 }
    } else if (line ~ /(^|[ \t])func[ \t]+`?[A-Za-z_]/ && inpublicctx() && line !~ /(^|[ \t])(private|fileprivate|internal|package)[ \t]/) {
        sig = line
        if (balanced(sig) && (line ~ /\{|->/ || line ~ /\)[ \t]*$/)) { flush_sig() } else { insig = 1 }
    }
    # context stack (extension / type declarations); the "{" may be on a later line
    if (match(line, /(^|[ \t])(extension|struct|class|enum|actor|protocol)[ \t]+[A-Za-z_][A-Za-z0-9_.]*/) && line !~ /(^|[ \t])func[ \t]/ && line !~ /^[ \t]*(case|let|var)[ \t]/) {
        s2 = substr(line, RSTART, RLENGTH); kind = s2; sub(/^[ \t]+/, "", kind); sub(/[ \t].*$/, "", kind)
        sub(/^[ \t]*(extension|struct|class|enum|actor|protocol)[ \t]+/, "", s2)
        pend = s2; pendkind = kind; pendpub = (line ~ /(^|[ \t])(public|open)[ \t]/ && kind == "extension") ? 1 : 0
    }
    if (pend != "" && line ~ /\{/) {
        top++; ctxs[top] = pend; ctxkind[top] = pendkind; ctxpub[top] = pendpub
        ctxdepth[top] = depth; pend = ""
    }
    depth += braces(line)
    while (top > 0 && depth <= ctxdepth[top]) top--
}
function balanced(s,   t, o, c) { t = s; gsub(/->/, "", t); o = gsub(/\(/, "(", t); c = gsub(/\)/, ")", t); return (o > 0 && o == c) }
function inpublicctx(   i) { for (i = top; i >= 1; i--) if (ctxpub[i]) return 1; return 0 }
AWK
scan() { # scan <root> <ours|theirs>
    local root=$1 f
    find "$root" -name '*.swift' -not -path '*/.build/*' -not -path '*/Tests/*' -not -path '*/Cmlx/*' -not -path '*/Examples/*' | sort | while IFS= read -r f; do
        awk -v mode="$2" -v rel="${f#"$root"/}" -f "$work/scan.awk" "$f"
    done
}
scan "$repo/Libraries" ours >"$work/ours.tsv"
scan "$mlx/Source" theirs >"$work/theirs.tsv"

grep -v -E '^[[:space:]]*(#|$)' "$allow" | sed 's/[[:space:]]*#.*$//; s/[[:space:]]*$//; s/[[:space:]][[:space:]]*/ /g' >"$work/allow.txt"
if bad=$(grep -v -E '^(COLLISION|OVERLAP) [^ ]+$' "$work/allow.txt" | head -1) && [ -n "$bad" ]; then
    die "allow-list entry needs a kind, COLLISION or OVERLAP, then the key (got: $bad)"
fi

echo "== mlx-swift-shadow: ours=$(wc -l <"$work/ours.tsv" | tr -d ' ') public funcs, mlx-swift=$(wc -l <"$work/theirs.tsv" | tr -d ' ') public funcs"
python3 -I - "$work" <<'PY' >"$work/result.txt"
import sys, collections
w = sys.argv[1]
def load(p):
    rows = []
    for l in open(p, encoding="utf-8", errors="replace"):
        c, n, lab, dfl, rel = (l.rstrip("\n").split("\t") + [""] * 5)[:5]
        rows.append((c, n, lab.split(",") if lab else [], dfl.split(",") if dfl else [], rel))
    return rows
ours, theirs = load(w + "/ours.tsv"), load(w + "/theirs.tsv")
allow = set(x.strip() for x in open(w + "/allow.txt") if x.strip())
def key(c, n, labs): return "%s.%s(%s)" % (c, n, "".join(l + ":" for l in labs))
tby = collections.defaultdict(list)
for t in theirs: tby[(t[0], t[1])].append(t)
seen = set()
def subseq(a, b):
    it = iter(b)
    return all(any(x == y for y in it) for x in a)
def extras_default(a, b):
    # b's parameters not matched by a must all have defaults: greedy in-order match
    j = 0; used = set()
    for x in a:
        while j < len(b[0]) and b[0][j] != x: j += 1
        if j >= len(b[0]): return False
        used.add(j); j += 1
    return all(b[1][i] == "1" for i in range(len(b[0])) if i not in used and i < len(b[1]))
for c, n, labs, dfl, rel in ours:
    for t in tby.get((c, n), []):
        k = key(c, n, labs)
        if labs == t[2]:
            kind = "COLLISION"
        elif subseq(labs, t[2]) and extras_default(labs, (t[2], t[3])):
            kind = "OVERLAP"
        else:
            continue
        if (kind, k) in seen: continue
        seen.add((kind, k))
        print("%s\t%s\t%s\t%s\t%s" % (kind, k, rel, t[4], key(t[0], t[1], t[2])))
PY
status=0
while IFS="$(printf '\t')" read -r kind k ours_f theirs_f theirs_k; do
    if grep -qxF -- "$kind $k" "$work/allow.txt"; then
        echo "ALLOWED $kind $k  (ours: $ours_f; mlx-swift: $theirs_f $theirs_k)"
    else
        echo "$kind $k  (ours: $ours_f; mlx-swift: $theirs_f declares $theirs_k)"
        status=1
    fi
done <"$work/result.txt"
if [ "$status" -eq 0 ]; then echo "MLX_SWIFT_SHADOW=PASS"; else echo "MLX_SWIFT_SHADOW=FAIL"; fi
exit "$status"
