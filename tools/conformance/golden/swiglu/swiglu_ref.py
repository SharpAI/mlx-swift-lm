"""float64 torch-CPU oracle for the SwiGLU family. Every line cites the vLLM source it is transcribed from
(vLLM commit 739e5945d; crosscheck() reads the checkout named by the VLLM_ROOT environment variable). Lines marked INFERRED have no vLLM source.

All functions take x of shape (..., 2*d) in float64 and return (..., d) in float64. Inputs are pre-rounded
to the test dtype by generate.py, so the oracle is the EXACT real-number result for those inputs; the Swift
side's deviation is therefore pure dtype rounding (see MANIFEST tolerances).
"""
import torch
import torch.nn.functional as F


def silu_and_mul(x):
    # vllm/model_executor/layers/activation.py:137-141 SiluAndMul.forward_native
    d = x.shape[-1] // 2                       # activation.py:140
    return F.silu(x[..., :d]) * x[..., d:]     # activation.py:141  (chunk layout: gate first, up second)


def swigluoai_and_mul(x, alpha=1.702, limit=7.0):
    # vllm/model_executor/layers/activation.py:309-327 SwigluOAIAndMul (defaults alpha=1.702, limit=7.0: :314)
    gate, up = x[..., ::2], x[..., 1::2]       # activation.py:322 (INTERLEAVED layout)
    gate = gate.clamp(min=None, max=limit)     # activation.py:323 (gate: upper clamp only)
    up = up.clamp(min=-limit, max=limit)       # activation.py:324 (up: two-sided)
    glu = gate * torch.sigmoid(gate * alpha)   # activation.py:325
    return (up + 1) * glu                      # activation.py:326 (gated_output), returned :327


def swiglustep_and_mul(x, limit=7.0):
    # vllm/model_executor/layers/activation.py:341-365 SwigluStepAndMul.forward_native (default limit 7.0: :353)
    gate, up = x.chunk(2, dim=-1)              # activation.py:361 (chunk layout)
    gate = F.silu(gate)                        # activation.py:362  silu FIRST
    gate = gate.clamp(max=limit)               # activation.py:363  upper clamp on the silu OUTPUT
    up = up.clamp(min=-limit, max=limit)       # activation.py:364
    return gate * up                           # activation.py:365


def swiglustep_kernel_emulation(x, limit=7.0):
    # Transcription of the Triton kernel the CUDA path really runs: activation.py:27-52 (body :34-52)
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]          # activation.py:43-44 (offsets, offsets + d)
    gate_silu = torch.sigmoid(gate) * gate     # activation.py:46
    gate_clamped = torch.minimum(gate_silu, torch.tensor(limit, dtype=x.dtype))   # :47
    up_clamped = torch.minimum(torch.maximum(up, torch.tensor(-limit, dtype=x.dtype)),
                               torch.tensor(limit, dtype=x.dtype))                # :48
    return gate_clamped * up_clamped           # :50


def v4_clamped(x, limit):
    # DeepSeek-V4 style  silu(clamp(gate)) * clamp(up).  INFERRED: vLLM at 739e5945d has no DeepSeek-V4 model
    # (ls vllm/model_executor/models | grep deepseek -> v2 only) and no V4 swiglu_limit code; the formula is the
    # lane brief + docs/support-kernels.json "swiglu clamp, V4 style". Sidedness of the gate clamp is NOT
    # established by any source available here, so both readings are produced:
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]          # INFERRED chunk layout (HF gate_proj/up_proj are separate; no interleave)
    up_c = up.clamp(min=-limit, max=limit)     # INFERRED two-sided (docs/support-kernels.json evidence line)
    sym = F.silu(gate.clamp(min=-limit, max=limit)) * up_c    # reading A: two-sided gate clamp
    maxonly = F.silu(gate.clamp(max=limit)) * up_c            # reading B: gate clamp(max=limit) only
    return sym, maxonly


def crosscheck(n_cases=400, seed=1234, vllm_root=None):
    """Compare the oracle above with vLLM's own forward_native bodies, extracted from the vLLM source with ast
    (importing vllm needs CUDA-era deps). Returns max relative disagreement per op."""
    import ast, os, types
    vllm_root = vllm_root or os.environ.get("VLLM_ROOT")
    if not vllm_root:
        raise SystemExit("set VLLM_ROOT to a vLLM checkout at commit 739e5945d (the directory that contains vllm/)")
    src_path = os.path.join(vllm_root, "vllm/model_executor/layers/activation.py")
    tree = ast.parse(open(src_path).read())
    fns = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ("SiluAndMul", "SwigluOAIAndMul", "SwigluStepAndMul"):
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == "forward_native":
                    f.decorator_list = []
                    m = ast.Module(body=[f], type_ignores=[])
                    ns = {"torch": torch, "F": F}
                    exec(compile(ast.fix_missing_locations(m), src_path, "exec"), ns)
                    fns[node.name] = ns["forward_native"]
    assert set(fns) == {"SiluAndMul", "SwigluOAIAndMul", "SwigluStepAndMul"}, fns.keys()
    g = torch.Generator().manual_seed(seed)
    worst = {"silu_and_mul": 0.0, "swigluoai_and_mul": 0.0, "swiglustep_and_mul": 0.0, "step_kernel_vs_native": 0.0}
    rel = lambda a, b: float(((a - b).abs() / (b.abs() + 1e-300)).max()) if a.numel() else 0.0
    for i in range(n_cases):
        d = int(torch.randint(1, 40, (1,), generator=g))
        rows = int(torch.randint(1, 6, (1,), generator=g))
        scale = float(torch.rand(1, generator=g) * 20 + 0.1)
        x = (torch.randn(rows, 2 * d, generator=g, dtype=torch.float64) * scale)
        limit = float(torch.rand(1, generator=g) * 12 + 0.5)
        worst["silu_and_mul"] = max(worst["silu_and_mul"], rel(silu_and_mul(x), fns["SiluAndMul"](x)))
        ns = types.SimpleNamespace(alpha=1.702, limit=limit)
        worst["swigluoai_and_mul"] = max(worst["swigluoai_and_mul"], rel(swigluoai_and_mul(x, 1.702, limit), fns["SwigluOAIAndMul"](ns, x)))
        ns2 = types.SimpleNamespace(limit=limit)
        ref = fns["SwigluStepAndMul"](ns2, x)
        worst["swiglustep_and_mul"] = max(worst["swiglustep_and_mul"], rel(swiglustep_and_mul(x, limit), ref))
        worst["step_kernel_vs_native"] = max(worst["step_kernel_vs_native"], rel(swiglustep_kernel_emulation(x, limit), ref))
    return worst
