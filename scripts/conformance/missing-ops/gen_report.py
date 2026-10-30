#!/usr/bin/env python3
"""Missing-ops report: which config keys does a registered model's Configuration silently ignore?

For every config-only fixture in scripts/conformance/offline/ whose `model_type` is registered in the
LLM / VLM / embedder type registry, this script
  1. resolves the registry entry to its Configuration type (static read of the *TypeRegistry sources),
  2. extracts the top-level JSON keys that Configuration type decodes (its `CodingKeys` and any other `*CodingKeys: CodingKey` enums, or the
     stored property names when it has none),
  3. diffs them against the fixture's top-level keys.
A key in the fixture that the Configuration does not decode is "silently ignored": the model builds, but
whatever the key controls (an op, a layer, a scale, a quantization format) is not honoured. Keys that
are plain metadata (architectures, dtype, dropout probabilities, token ids ...) are split out so the
interesting list ("candidate ops") is not buried.

This is a STATIC estimate: it reads Swift sources, it does not build a model. The Swift test
ConformanceMissingOpsTests cross-checks it against a real Codable round trip for several families.
Nested containers (`text_config`, `vision_config`, ...) count as consumed when the Configuration
decodes the container key itself; their inner keys are not diffed.

usage: gen_report.py [--root REPO] [--out-dir DIR] [--stdout]
  writes <out-dir>/REPORT.md and <out-dir>/REPORT.csv (default out-dir: this script's directory).
The report is deterministic (sorted, no timestamps) so a diff of it is a diff of the facts.
"""
import argparse, csv, glob, io, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))

REGISTRIES = [
    ("llm", "Libraries/MLXLLM/LLMModelFactory.swift", "LLMTypeRegistry"),
    ("vlm", "Libraries/MLXVLM/VLMModelFactory.swift", "VLMTypeRegistry"),
    ("embedder", "Libraries/MLXEmbedders/ModelFactory.swift", "EmbedderTypeRegistry"),
]

# Top-level keys that are descriptive or training-time only: ignoring them loses no inference op.
METADATA_EXACT = {
    "architectures", "auto_map", "model_type", "torch_dtype", "dtype", "transformers_version", "use_cache",
    "initializer_range", "bos_token_id", "eos_token_id", "pad_token_id", "sep_token_id", "decoder_start_token_id",
    "_name_or_path", "tokenizer_class", "use_return_dict", "output_attentions", "output_hidden_states",
    "return_dict", "problem_type", "label2id", "id2label", "is_decoder", "tie_encoder_decoder",
    "router_aux_loss_coef", "output_router_logits", "load_balance_coeff", "gradient_checkpointing",
    "_attn_implementation", "attn_implementation", "pretraining_tp", "use_mamba_kernels", "num_logits_to_keep",
    "rescale_prenorm_residual", "residual_in_fp32", "mamba_ssm_cache_dtype", "mamba_ssm_dtype", "use_grouped_mm",
    "use_bfloat16", "moe_aux_loss_coeff", "z_loss_coeff", "aux_loss_alpha", "seq_aux",
}
METADATA_RE = re.compile(r"(_dropout|_dropout_prob|_drop_prob|dropout)$|^(image|video|audio|vision)_.*token_id$")


def is_metadata(key):
    return key in METADATA_EXACT or bool(METADATA_RE.search(key))


def swift_files(root):
    out = []
    for d in ("Libraries/MLXLLM", "Libraries/MLXVLM", "Libraries/MLXEmbedders", "Libraries/MLXLMCommon"):
        out += glob.glob(os.path.join(root, d, "**", "*.swift"), recursive=True)
    return sorted(out)


def match_brace(src, open_idx):
    depth, i = 0, open_idx
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        elif c == '"':  # skip string literals
            i += 1
            while i < len(src) and src[i] != '"':
                i += 2 if src[i] == "\\" else 1
        elif c == "/" and src[i:i + 2] == "//":
            while i < len(src) and src[i] != "\n":
                i += 1
        i += 1
    return len(src) - 1


def first_config(text):
    """First `XConfiguration.self` in `text`, skipping the helper that only reads `architectures`."""
    for n in re.findall(r"\b([A-Z][A-Za-z0-9_]*Configuration)\.self", text):
        if n != "ArchitectureConfiguration":
            return n
    return None


def registry_entries(root, path, enum_name):
    """-> {model_type: configuration-type-name or None}"""
    src = open(os.path.join(root, path), encoding="utf8").read()
    start = src.index(f"enum {enum_name}")
    m = re.search(r"\.init\(creators:\s*\[", src[start:])
    body_start = start + m.end()
    depth, i = 1, body_start
    while depth:
        c = src[i]
        depth += (c == "[") - (c == "]")
        i += 1
    body = src[body_start:i - 1]
    heads = list(re.finditer(r'^\s{8}"([A-Za-z0-9_.\-]+)"\s*:', body, re.M))
    out = {}
    for n, h in enumerate(heads):
        end = heads[n + 1].start() if n + 1 < len(heads) else len(body)
        entry = body[h.end():end]
        name = first_config(entry)
        if name is None:
            # `"bert": createBertCompatibleModel,` -- a helper function; read its decode call.
            fm = re.match(r"\s*([A-Za-z_]\w*)\s*,?\s*$", entry)
            if fm:
                fn = re.search(r"func\s+%s\b[^{]*\{" % re.escape(fm.group(1)), src)
                if fn:
                    name = first_config(src[fn.end():match_brace(src, fn.end() - 1)])
        out[h.group(1)] = name
    return out


class TypeIndex:
    """Finds `struct|class X ... {` bodies across the library sources."""

    def __init__(self, root):
        self.sources = {p: open(p, encoding="utf8").read() for p in swift_files(root)}
        self.cache = {}

    def body(self, name, prefer=""):
        """Body of `struct|class name`, searching the registry's own module (`prefer`) before the others,
        because MLXLMCommon carries same-named shared types for a few families."""
        if (name, prefer) in self.cache:
            return self.cache[(name, prefer)]
        pat = re.compile(r"\b(?:struct|class)\s+%s\b[^{]*\{" % re.escape(name))
        found = None
        paths = sorted(self.sources, key=lambda p: (prefer not in p, p))
        for p in paths:
            src = self.sources[p]
            m = pat.search(src)
            if m:
                found = src[m.end():match_brace(src, m.end() - 1)]
                break
        self.cache[(name, prefer)] = found
        return found


def coding_keys(body):
    """-> (keys, how)  how in {'CodingKeys', 'properties', 'unresolved'}

    Reads EVERY enum conforming to CodingKey declared in the Configuration body (`CodingKeys`, but also
    e.g. BertConfiguration's `BertCodingKeys` / `DistilBertCodingKeys`, `VLMCodingKeys` ...): a key
    decoded through any of them is consumed. This can only over-count consumed keys for enums that
    describe nested containers, never report a decoded top-level key as ignored.
    """
    keys, found = set(), False
    for m in re.finditer(r"\benum\s+\w+\s*:[^{]*\bCodingKey\b[^{]*\{|\benum\s+CodingKeys\b[^{]*\{", body):
        found = True
        end = match_brace(body, m.end() - 1)
        enum_body = body[m.end():end]
        for line in enum_body.splitlines():
            line = re.sub(r"//.*", "", line).strip()
            if not line.startswith("case "):
                continue
            for part in line[5:].split(","):
                part = part.strip()
                if not part:
                    continue
                am = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*\"([^\"]*)\")?", part)
                if am:
                    keys.add(am.group(2) if am.group(2) is not None else am.group(1))
    if found:
        return keys, "CodingKeys"
    props = set(re.findall(r"^\s{4}(?:public\s+|private\s+|fileprivate\s+)*(?:let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:=]", body, re.M))
    if props:
        return props, "properties"
    return set(), "unresolved"


def load_fixtures(root):
    man = json.load(open(os.path.join(root, "scripts/conformance/offline/MANIFEST.json"), encoding="utf8"))
    out = []
    for f in man["fixtures"]:
        mt = f.get("model_type")
        if not mt:
            continue
        data = json.load(open(os.path.join(root, "scripts/conformance/offline", f["path"]), encoding="utf8"))
        out.append({"model_type": mt, "repo": f["repo"], "path": f["path"], "keys": sorted(data.keys())})
    return sorted(out, key=lambda r: (r["model_type"], r["repo"]))


def build(root):
    """One row per (fixture, registry): a model_type can be registered in several registries
    (e.g. gemma4 in LLM and VLM) and each registry may use a different Configuration type."""
    regs = {kind: registry_entries(root, path, enum) for kind, path, enum in REGISTRIES}
    index = TypeIndex(root)
    prefer = {"llm": "/MLXLLM/", "vlm": "/MLXVLM/", "embedder": "/MLXEmbedders/"}
    rows = []
    for fx in load_fixtures(root):
        for kind, _, _ in REGISTRIES:
            if fx["model_type"] not in regs[kind]:
                continue
            cfg_name = regs[kind][fx["model_type"]]
            body = index.body(cfg_name, prefer[kind]) if cfg_name else None
            if body is None:
                keys, how = set(), "unresolved"
            else:
                keys, how = coding_keys(body)
            ignored = [] if how == "unresolved" else [k for k in fx["keys"] if k not in keys]
            meta = [k for k in ignored if is_metadata(k)]
            cand = [k for k in ignored if not is_metadata(k)]
            rows.append({
                "model_type": fx["model_type"], "repo": fx["repo"], "registry": kind,
                "configuration": cfg_name or "", "keys_source": how, "fixture_keys": len(fx["keys"]),
                "ignored_total": len(ignored), "ignored_metadata": len(meta), "ignored_candidates": len(cand),
                "candidates": cand, "metadata": meta,
            })
    return rows


def summary(rows):
    return {
        "rows": len(rows),
        "fixtures": len({(r["model_type"], r["repo"]) for r in rows}),
        "model_types": len({r["model_type"] for r in rows}),
        "unresolved": sum(1 for r in rows if r["keys_source"] == "unresolved"),
        "ignored_total": sum(r["ignored_total"] for r in rows),
        "ignored_metadata": sum(r["ignored_metadata"] for r in rows),
        "ignored_candidates": sum(r["ignored_candidates"] for r in rows),
        "fixtures_with_candidates": sum(1 for r in rows if r["ignored_candidates"]),
    }


CSV_FIELDS = ["model_type", "repo", "registry", "configuration", "keys_source", "fixture_keys",
              "ignored_total", "ignored_metadata", "ignored_candidates", "candidates", "metadata"]


def to_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_FIELDS)
    for r in rows:
        w.writerow([";".join(r[f]) if isinstance(r[f], list) else r[f] for f in CSV_FIELDS])
    return buf.getvalue()


def to_md(rows):
    s = summary(rows)
    L = []
    L.append("# Missing-ops report (config keys a registered Configuration silently ignores)")
    L.append("")
    L.append("Generated by `scripts/conformance/missing-ops/gen_report.py` -- do not edit by hand. "
             "Static estimate from the Swift sources (no model is built); see the script header for the method and its limits.")
    L.append("")
    L.append("## Totals")
    L.append("")
    L.append(f"* fixtures whose `model_type` is registered: **{s['fixtures']}** ({s['model_types']} distinct model types); "
             f"one row per (fixture, registry) = **{s['rows']}** rows")
    L.append(f"* rows whose Configuration could not be resolved (counted as 0 ignored): {s['unresolved']}")
    L.append(f"* ignored keys in total: **{s['ignored_total']}** = {s['ignored_metadata']} metadata + "
             f"**{s['ignored_candidates']}** candidate ops")
    L.append(f"* rows with at least one candidate op: {s['fixtures_with_candidates']}")
    L.append("")
    L.append("## Candidate ops per fixture (keys the Configuration does not decode, metadata excluded)")
    L.append("")
    L.append("| model_type | fixture (repo) | registry | Configuration | keys read from | candidates |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        if r["ignored_candidates"]:
            L.append(f"| `{r['model_type']}` | {r['repo']} | {r['registry']} | `{r['configuration']}` | {r['keys_source']} | "
                     + ", ".join(f"`{k}`" for k in r["candidates"]) + " |")
    L.append("")
    L.append("## Rows with no candidate op (Configuration resolved)")
    L.append("")
    clean = [r for r in rows if not r["ignored_candidates"] and r["keys_source"] != "unresolved"]
    L.append(", ".join(f"`{r['model_type']}` ({r['repo']}, {r['registry']})" for r in clean) if clean else "(none)")
    L.append("")
    L.append("## Most common ignored candidate keys")
    L.append("")
    cnt = {}
    for r in rows:
        for k in r["candidates"]:
            cnt[k] = cnt.get(k, 0) + 1
    L.append("| key | fixtures |")
    L.append("|---|---|")
    for k, n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:25]:
        L.append(f"| `{k}` | {n} |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--stdout", action="store_true", help="print a one-line JSON summary instead of writing files")
    a = ap.parse_args()
    rows = build(os.path.abspath(a.root))
    if a.stdout:
        print(json.dumps(summary(rows), sort_keys=True))
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    open(os.path.join(a.out_dir, "REPORT.md"), "w", encoding="utf8").write(to_md(rows))
    open(os.path.join(a.out_dir, "REPORT.csv"), "w", encoding="utf8").write(to_csv(rows))
    print(json.dumps(summary(rows), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
