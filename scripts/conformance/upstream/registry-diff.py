#!/usr/bin/env python3
"""registry-diff.py BEFORE.swift AFTER.swift -- classify registry changes between two ConformanceExpectations files.

For every model_type the expected registry SET is compared, not the text line:
  REGRESSION  a registry that BEFORE listed is missing in AFTER (before is not a subset of after),
              including a model_type that vanished or one whose registry was swapped (llm -> vlm).
  LANDED      AFTER lists a registry BEFORE did not, and nothing was lost (new model_type, or llm -> llm, vlm).
Unchanged entries print nothing. Output, one finding per line (sentinel.sh indents and counts them):
  LANDED <model_type> -> <registries>  (was <registries>)
  REGRESSION <model_type> was <registries>  (now <registries>)
Exit 0 always when both files parse; 2 on usage / unreadable input.
"""
import re
import sys

PAT = re.compile(r'modelType: "([^"]*)", registries: \[([^\]]*)\]')


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = PAT.search(line)
            if m:
                regs = [r.strip() for r in m.group(2).split(",") if r.strip()]
                out[m.group(1)] = regs
    return out


def show(regs):
    return ", ".join(sorted(regs)) if regs else "(none)"


def main(argv):
    if len(argv) != 3:
        print("usage: registry-diff.py BEFORE.swift AFTER.swift", file=sys.stderr)
        return 2
    try:
        before, after = load(argv[1]), load(argv[2])
    except OSError as e:
        print("registry-diff: %s" % e, file=sys.stderr)
        return 2
    for name in sorted(set(before) | set(after)):
        b, a = set(before.get(name, [])), set(after.get(name, []))
        if b == a:
            continue
        if not b <= a:
            print("REGRESSION %s was %s  (now %s)" % (name, show(b), show(a)))
        else:
            print("LANDED %s -> %s  (was %s)" % (name, show(a), show(b)))
    return 0


sys.exit(main(sys.argv))
