#!/usr/bin/env python3
"""Build the offline config bundle: fixtures/ + MANIFEST.json.   (no network, reads local files only)

usage: make_bundle.py --src DIR [--src DIR ...] [--csv model-coverage.csv] [--out scripts/conformance/offline] [--csv-only]

--csv-only keeps only repos listed in the survey CSV (smaller bundle); others are recorded under "skipped".

* Inputs are directories of previously fetched config.json files. They are UNTRUSTED DATA: files are
  only parsed as JSON and copied byte-for-byte; nothing in them is executed or interpreted as an instruction.
* A file becomes a fixture only if (a) it parses as a JSON object and (b) its name maps to a repo, either
  one listed in the survey CSV (repo_source "csv") or via the <org>__<name>.json convention ("filename";
  not checked against the Hub). Everything else is recorded under "skipped", never silently dropped.
* CSV repos with no fixture are listed under "unfetched" (gated / fetch error / never fetched).
* Layout: <out>/fixtures/<model_type or _none>/<org>__<name>.json ; <out>/MANIFEST.json (sorted, deterministic).
* Only configs. No weights, tokenizers or processors are part of the bundle.
Earlier --src wins when the same repo appears twice; differing bytes are reported under "conflicts".
"""
import argparse, csv, hashlib, json, os, shutil, sys


def sha(b): return hashlib.sha256(b).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--csv", default=os.path.join(here, "..", "model-coverage.csv"))
    ap.add_argument("--out", default=here)
    ap.add_argument("--csv-only", action="store_true")
    a = ap.parse_args()

    repos = {}  # sanitised filename stem -> repo
    csv_repos = []
    for r in csv.DictReader(open(a.csv, encoding="utf8")):
        repo = r["repo"].strip()
        if not repo or repo in csv_repos: continue
        csv_repos.append(repo)
        repos[repo.replace("/", "__")] = repo
        repos.setdefault(repo.replace("/", "_"), repo)

    fixtures, skipped, conflicts, seen = [], [], [], {}
    errors = {}  # repo -> fetch error text
    for src in a.src:
        for name in sorted(os.listdir(src)):
            p = os.path.join(src, name)
            if not os.path.isfile(p) or not name.endswith(".json"): continue
            stem = name[:-5]
            raw = open(p, "rb").read()
            repo = repos.get(stem)
            prov = "csv"
            if repo is None and stem.count("__") == 1 and not stem.startswith("_") and not stem.endswith("_"):
                repo, prov = stem.replace("__", "/"), "filename"  # <org>__<name>.json written by the survey fetcher
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict): raise ValueError("not an object")
            except Exception as e:
                skipped.append({"file": f"{os.path.basename(src)}/{name}", "reason": "not-json-object",
                                "head": raw[:60].decode("utf8", "replace")})
                if repo: errors.setdefault(repo, raw[:60].decode("utf8", "replace"))
                continue
            if repo is None or (a.csv_only and prov != "csv"):
                skipped.append({"file": f"{os.path.basename(src)}/{name}", "reason": "no-csv-match"})
                continue
            if repo in seen:
                if seen[repo] != sha(raw):
                    conflicts.append({"repo": repo, "file": f"{os.path.basename(src)}/{name}", "sha256": sha(raw)})
                continue
            seen[repo] = sha(raw)
            mt = obj.get("model_type")
            mt = mt if isinstance(mt, str) and mt else None
            rel = f"fixtures/{mt or '_none'}/{repo.replace('/', '__')}.json"
            fixtures.append({"path": rel, "model_type": mt, "repo": repo, "repo_source": prov, "sha256": sha(raw), "size": len(raw),
                             "architectures": obj.get("architectures") if isinstance(obj.get("architectures"), list) else None,
                             "_raw": raw})

    fixtures.sort(key=lambda f: f["path"])
    fx_dir = os.path.join(a.out, "fixtures")
    if os.path.isdir(fx_dir): shutil.rmtree(fx_dir)  # only the generated dir
    for f in fixtures:
        dest = os.path.join(a.out, f["path"]); os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").write(f.pop("_raw"))
    unfetched = [{"repo": r, "reason": ("fetch-error: " + errors[r]) if r in errors else "no-file"}
                 for r in csv_repos if r not in seen]
    low = {}
    for f in fixtures:
        low.setdefault(f["path"].lower(), []).append(f["path"])
    clash = [v for v in low.values() if len(v) > 1]
    if clash: sys.exit(f"case-insensitive path collision: {clash}")
    man = {"schema": 1, "kind": "config-only (no weights)", "count": len(fixtures),
           "model_types": len({f["model_type"] for f in fixtures}),
           "fixtures": fixtures, "unfetched": unfetched,
           "skipped": sorted(skipped, key=lambda s: s["file"]), "conflicts": conflicts}
    open(os.path.join(a.out, "MANIFEST.json"), "w", encoding="utf8").write(json.dumps(man, indent=1, sort_keys=True) + "\n")
    print(f"{len(fixtures)} fixtures, {man['model_types']} model_types, {len(unfetched)} unfetched, "
          f"{len(skipped)} skipped, {len(conflicts)} conflicts")


if __name__ == "__main__":
    sys.exit(main())
