"""Per-stage error attribution for a specific failing index.

Strategy: build the fp32 reference pipeline ('ground truth'). Then ALSO run
sub-variants where ONE bf16 round trip from ours is injected at each stage.
Watch how max_err on element (1,100,100,113) and channel-113 grows.

Stages from `_ref_kernel`:
  S1.  x.float() and LayerNorm in fp32
  S2.  5 fp32 GEMMs (left/right_proj, left/right_gate, out_gate)
  S3.  sigmoid * mask -> left, right (fp32)
  S4.  einsum (bf16 cast on left/right; ours uses bf16 cast at proj output)
  S5.  bmm output: ours=fp32 (after .float()); ref also fp32
  S6.  to_out_norm LN in fp32
  S7.  out_n * out_gate -> gated cast to bf16 (ours), fp32 (ref)
  S8.  final F.linear(gated, w_out) -> bf16 (ours), fp32 (ref)

We progressively inject ours' bf16 quantizations into the reference pipeline,
matching exactly how `custom_kernel` handles dtypes:
   v1 = ref baseline (all fp32 except einsum bf16 input)
   v2 = +bf16-quantize x_n (ours: x is bf16 entering proj GEMM)
   v3 = +bf16-quantize the 5 PROJECTIONS' weights & their outputs
        (`fused_gate_ln_bmm_layout` consumes proj as bf16, but inputs and weights
         to the projection GEMM are also bf16 — so this models the bf16 GEMM)
   v4 = +bf16-quantize lf, rf (left/right after sigmoid*mask)
   v5 = +bf16 for OG (already bf16 in ours through fused_gate_ln, og is fp32 now in ours)
   v6 = +fp32 for out_bmm (ours has this fp32 already)
   v7 = +bf16-quantize gated (ours casts to bf16 before final linear)
   v8 = +bf16 final linear weight (ours casts to bf16)
   v9 = +bf16 output of final linear (ours stores bf16)

Print, per variant, error at the failing index AND max_err on channel 113.
"""

import math
import os
import sys
import sysconfig
import types

sys.argv = [sys.argv[0], "--no-install"]
dist_packages = sysconfig.get_paths()["purelib"]
os.environ["TRITON_PLUGIN_PATHS"] = os.path.join(
    dist_packages, "utlx_plugin", "libutlx.so"
)

import triton  # noqa
import utlx_plugin  # noqa

sys.path.insert(0, "/home/wychi/oss/wheels/runner")
import tlx_patches

KERNEL_FILE = "/home/wychi/oss/wheels/gpumode/bioml/trimul/work/hopper_gemm_ws.py"
tlx_patches.apply(tlx_patches.resolve_for_kernel(KERNEL_FILE))
src = (
    open(KERNEL_FILE)
    .read()
    .replace("_setup_utlx()\n", "pass  # _setup_utlx() stubbed by wrapper\n")
)
mod = types.ModuleType("hopper_gemm_ws")
mod.__file__ = KERNEL_FILE
exec(compile(src, KERNEL_FILE, "exec"), mod.__dict__)

import torch  # noqa
import torch.nn.functional as F


def _alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(_alloc_fn)
torch.backends.cuda.matmul.allow_tf32 = False


def leaderboard_input(bs, sl, dim, hd, dist, seed, nomask=True):
    cfg = {"hidden_dim": hd, "dim": dim}
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if dist == "cauchy":
        x = (
            torch.distributions.Cauchy(0, 2)
            .sample((bs, sl, sl, dim))
            .to(device="cuda", dtype=torch.float32)
        )
    else:
        x = torch.randn(
            (bs, sl, sl, dim), device="cuda", dtype=torch.float32, generator=gen
        ).contiguous()
    if nomask:
        m = torch.ones(bs, sl, sl, device="cuda")
    else:
        m = torch.randint(0, 2, (bs, sl, sl), device="cuda", generator=gen)
    W = {}
    W["norm.weight"] = torch.randn(dim, device="cuda", dtype=torch.float32)
    W["norm.bias"] = torch.randn(dim, device="cuda", dtype=torch.float32)
    W["left_proj.weight"] = torch.randn(
        hd, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hd)
    W["right_proj.weight"] = torch.randn(
        hd, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hd)
    W["left_gate.weight"] = torch.randn(
        hd, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hd)
    W["right_gate.weight"] = torch.randn(
        hd, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hd)
    W["out_gate.weight"] = torch.randn(
        hd, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hd)
    W["to_out_norm.weight"] = torch.randn(hd, device="cuda", dtype=torch.float32)
    W["to_out.weight"] = torch.randn(
        dim, hd, device="cuda", dtype=torch.float32
    ) / math.sqrt(dim)
    W["to_out_norm.bias"] = torch.randn(hd, device="cuda", dtype=torch.float32)
    return (x, m, W, cfg)


def pipeline(data, q):
    """Reference pipeline with switchable bf16 quantization at named breakpoints.

    q is a dict with keys: 'x_n', 'proj_in', 'lf_rf', 'og', 'bmm', 'gated',
    'wout', 'final'. Each value is True (apply bf16 round-trip at that point).
    """
    x, mask, W, cfg = data
    dim, hd = cfg["dim"], cfg["hidden_dim"]
    x = x.float()
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    x_n = (x - mu) * torch.rsqrt(var + 1e-5) * W["norm.weight"] + W["norm.bias"]

    # x_n is the input to the 5 projection GEMMs. Ours: x_n is bf16.
    if q.get("x_n"):
        x_n = x_n.to(torch.bfloat16).to(torch.float32)

    # 5 projection weights bf16-quantized (the cached B_g is bf16)
    def Wbf(name):
        if q.get("proj_in"):
            return W[name].to(torch.bfloat16).to(torch.float32)
        return W[name]

    m = mask.float().unsqueeze(-1)
    proj_left = F.linear(x_n, Wbf("left_proj.weight"))
    proj_right = F.linear(x_n, Wbf("right_proj.weight"))
    proj_lg = F.linear(x_n, Wbf("left_gate.weight"))
    proj_rg = F.linear(x_n, Wbf("right_gate.weight"))
    proj_og = F.linear(x_n, Wbf("out_gate.weight"))

    # Ours: tlx_ws_matmul_fixed produces bf16 -> proj is bf16
    if q.get("proj_in"):
        proj_left = proj_left.to(torch.bfloat16).to(torch.float32)
        proj_right = proj_right.to(torch.bfloat16).to(torch.float32)
        proj_lg = proj_lg.to(torch.bfloat16).to(torch.float32)
        proj_rg = proj_rg.to(torch.bfloat16).to(torch.float32)
        proj_og = proj_og.to(torch.bfloat16).to(torch.float32)

    left = proj_left * m * torch.sigmoid(proj_lg)
    right = proj_right * m * torch.sigmoid(proj_rg)
    out_gate = torch.sigmoid(proj_og)

    # Ours: L, R buffers are bf16
    if q.get("lf_rf"):
        left = left.to(torch.bfloat16).to(torch.float32)
        right = right.to(torch.bfloat16).to(torch.float32)

    if q.get("og"):
        out_gate = out_gate.to(torch.bfloat16).to(torch.float32)

    # Ref forces bf16 here; mirror that
    out = torch.einsum(
        "...ikd,...jkd->...ijd",
        left.to(torch.bfloat16),
        right.to(torch.bfloat16),
    ).to(torch.float32)

    if q.get("bmm"):
        # Ours has out_bmm = bmm.float() — fp32. The ref is also fp32. Only
        # diff would be if we kept it bf16; toggle here for that hypothetical.
        out = out.to(torch.bfloat16).to(torch.float32)

    mu2 = out.mean(-1, keepdim=True)
    var2 = out.var(-1, keepdim=True, unbiased=False)
    out_n = (out - mu2) * torch.rsqrt(var2 + 1e-5) * W["to_out_norm.weight"] + W[
        "to_out_norm.bias"
    ]

    gated = out_n * out_gate
    if q.get("gated"):
        gated = gated.to(torch.bfloat16).to(torch.float32)

    wout = W["to_out.weight"]
    if q.get("wout"):
        wout = wout.to(torch.bfloat16).to(torch.float32)

    y = F.linear(gated, wout)
    if q.get("final"):
        y = y.to(torch.bfloat16).to(torch.float32)
    return y


def run_for_seed(input_seed, target_idx=(1, 100, 100, 113), target_ch=113):
    bs, sl, dim, hd = 2, 256, 128, 128
    data = leaderboard_input(bs, sl, dim, hd, "normal", input_seed, nomask=True)
    ref = pipeline(data, {})

    # Compare against ours
    ours = mod.custom_kernel(data).float()
    diff_ours = (ours - ref).abs()
    n_bad = (diff_ours > (2e-2 + 2e-2 * ref.abs())).sum().item()
    err_at_idx = diff_ours[target_idx].item()
    print(
        f"# input_seed={input_seed}: ours vs ref "
        f"max_err={diff_ours.max().item():.5f} "
        f"err@{target_idx}={err_at_idx:.5f} n_bad={n_bad}"
    )
    print(
        f"#   ours[{target_idx}]={ours[target_idx].item():.6f} "
        f"ref[{target_idx}]={ref[target_idx].item():.6f}"
    )

    # Cumulative-quantization variants
    seq = [
        ("baseline", {}),
        ("+x_n bf16", {"x_n": True}),
        ("+proj bf16 (incl wts)", {"x_n": True, "proj_in": True}),
        ("+lf/rf bf16", {"x_n": True, "proj_in": True, "lf_rf": True}),
        ("+og bf16", {"x_n": True, "proj_in": True, "lf_rf": True, "og": True}),
        # bmm intentionally OFF — ours already promotes to fp32
        (
            "+gated bf16",
            {
                "x_n": True,
                "proj_in": True,
                "lf_rf": True,
                "og": True,
                "gated": True,
            },
        ),
        (
            "+wout bf16",
            {
                "x_n": True,
                "proj_in": True,
                "lf_rf": True,
                "og": True,
                "gated": True,
                "wout": True,
            },
        ),
        (
            "+final-out bf16 (full ours model)",
            {
                "x_n": True,
                "proj_in": True,
                "lf_rf": True,
                "og": True,
                "gated": True,
                "wout": True,
                "final": True,
            },
        ),
    ]
    print("# Cumulative quantization vs fp32 ref:")
    print(
        f"#   {'variant':40s}  {'max_err':>9}  {'err@idx':>9}  "
        f"{'ch113_max':>9}  {'n_over_tol':>10}"
    )
    prev = None
    for name, q in seq:
        v = pipeline(data, q)
        d = (v - ref).abs()
        tol = 2e-2 + 2e-2 * ref.abs()
        n_over = (d > tol).sum().item()
        ch_max = d[..., target_ch].max().item()
        delta_str = ""
        if prev is not None:
            delta = d.max().item() - prev
            delta_str = f"  Δ={delta:+.5f}"
        prev = d.max().item()
        print(
            f"#   {name:40s}  {d.max().item():9.5f}  {d[target_idx].item():9.5f}  "
            f"{ch_max:9.5f}  {n_over:10d}{delta_str}"
        )

    # Now compare ours vs the v8 variant (full bf16 model)
    v8 = pipeline(data, seq[-1][1])
    d_ours_vs_v8 = (ours - v8).abs()
    print(
        f"# ours vs full-bf16-model: max_err={d_ours_vs_v8.max().item():.5f} "
        f"(should be small if model captures all bf16 round trips)"
    )
    return diff_ours, err_at_idx, n_bad


# Test seed 9371 (the actual server failing seed) and a few neighbors
for s in [9371, 9370, 9372, 731]:
    print(f"\n========== input seed {s} ==========")
    try:
        run_for_seed(s)
    except Exception as e:
        print(f"  error: {e}")
