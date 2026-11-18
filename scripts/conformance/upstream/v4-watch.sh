#!/usr/bin/env bash
# v4-watch.sh -- SL-42: alert when an upstream tree gains DeepseekV4 code (SL-41 policy: KEEP OURS).
# Our V4 lives under Libraries/MLXLLM/SharpAI/ with a SharpAI prefix; upstream has none of it. Any of these in the
# UPSTREAM tree means upstream started its own V4, which needs a human decision (see UPSTREAM_WATCH.md):
#   * a file named DeepseekV4*.swift
#   * a "deepseek_v4" key (quoted) in Swift source
#   * a declaration of a DeepseekV4-named type
# usage: scripts/conformance/upstream/v4-watch.sh <upstream-Libraries-dir>
# output: one "V4_ALERT <kind> <path>:<line>" line per hit, then "V4_RESULT=ALERT|CLEAN". Never fails the sentinel:
#         exit 0 when clean, 5 when an alert was raised, 2 on usage error. Alerts are reported, not auto-resolved.
set -u
dir=${1:-}
[ -n "$dir" ] && [ -d "$dir" ] || { echo "v4-watch: usage: v4-watch.sh <upstream-Libraries-dir>" >&2; echo "V4_RESULT=ERROR"; exit 2; }
hits=0
while IFS= read -r f; do
    echo "V4_ALERT file $f"; hits=$((hits + 1))
done < <(find "$dir" -type f -name 'DeepseekV4*.swift' 2>/dev/null | sed "s|^$dir/||" | sort)
while IFS= read -r line; do
    echo "V4_ALERT key $line"; hits=$((hits + 1))
done < <(grep -rn --include='*.swift' -F '"deepseek_v4"' "$dir" 2>/dev/null | sed "s|$dir/||")
while IFS= read -r line; do
    echo "V4_ALERT type $line"; hits=$((hits + 1))
done < <(grep -rn --include='*.swift' -E '(struct|class|enum|typealias|protocol|actor) DeepseekV4[A-Za-z0-9_]*' "$dir" 2>/dev/null | sed "s|$dir/||")
if [ "$hits" -gt 0 ]; then echo "V4_RESULT=ALERT"; exit 5; fi
echo "V4_RESULT=CLEAN"; exit 0
