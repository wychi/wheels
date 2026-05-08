"""For each stage in our pipeline, replace it with the fp32 reference computation
and rerun. This tells us which bf16 stage contributes most to cumulative error.
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


def make_input(seed=731):
    bs, sl, dim, hd = 2, 768, 128, 128
    cfg = {"hidden_dim": hd, "dim": dim}
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        (bs, sl, sl, dim), device="cuda", dtype=torch.float32, generator=gen
    ).contiguous()
    m = torch.ones(bs, sl, sl, device="cuda")
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


def fp32_pipeline(data, stage_overrides=()):
    """Reference pipeline. stage_overrides is a list of names of stages to
    use the OUR-kernel intermediate value for, rest are fp32 reference."""
    x, mask, W, cfg = data
    dim, hd = cfg["dim"], cfg["hidden_dim"]
    x = x.float()
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    x_n = (x - mu) * torch.rsqrt(var + 1e-5) * W["norm.weight"] + W["norm.bias"]
    m = mask.float().unsqueeze(-1)
    left = (
        F.linear(x_n, W["left_proj.weight"])
        * m
        * torch.sigmoid(F.linear(x_n, W["left_gate.weight"]))
    )
    right = (
        F.linear(x_n, W["right_proj.weight"])
        * m
        * torch.sigmoid(F.linear(x_n, W["right_gate.weight"]))
    )
    out_gate = torch.sigmoid(F.linear(x_n, W["out_gate.weight"]))
    # Reference forces bf16 here
    out = torch.einsum(
        "...ikd,...jkd->...ijd", left.to(torch.bfloat16), right.to(torch.bfloat16)
    )
    out = out.to(torch.float32)
    mu2 = out.mean(-1, keepdim=True)
    var2 = out.var(-1, keepdim=True, unbiased=False)
    out_n = (out - mu2) * torch.rsqrt(var2 + 1e-5) * W["to_out_norm.weight"] + W[
        "to_out_norm.bias"
    ]
    return F.linear(out_n * out_gate, W["to_out.weight"])


def main():
    data = make_input(seed=731)
    ref = fp32_pipeline(data)
    ours = mod.custom_kernel(data).float()
    diff = (ours - ref).abs()
    tol = 2e-2 + 2e-2 * ref.abs()
    n_bad = (diff > tol).sum().item()
    print(f"baseline: n_bad={n_bad}, max_err={diff.max().item():.5f}")

    # Now mutate the input weights to bf16 and back to fp32 for each weight
    # — see how reference precision changes when each weight is bf16-quantized.
    for k in [
        "norm.weight",
        "norm.bias",
        "left_proj.weight",
        "right_proj.weight",
        "left_gate.weight",
        "right_gate.weight",
        "out_gate.weight",
        "to_out_norm.weight",
        "to_out_norm.bias",
        "to_out.weight",
    ]:
        d2 = (data[0], data[1], dict(data[2]), data[3])
        d2[2][k] = data[2][k].to(torch.bfloat16).to(torch.float32)
        ref2 = fp32_pipeline(d2)
        diff2 = (ref - ref2).abs()
        tol2 = 2e-2 + 2e-2 * ref.abs()
        n_bad2 = (diff2 > tol2).sum().item()
        print(
            f"  bf16-quantize {k:25s}: max_err={diff2.max().item():.5f}, "
            f"would_fail_n={n_bad2}"
        )

    # Bf16 the input
    d2 = list(data)
    d2[0] = data[0].to(torch.bfloat16).to(torch.float32)
    ref3 = fp32_pipeline(tuple(d2))
    diff3 = (ref - ref3).abs()
    n_bad3 = (diff3 > tol).sum().item()
    print(
        f"  bf16-quantize input x: max_err={diff3.max().item():.5f}, would_fail_n={n_bad3}"
    )


if __name__ == "__main__":
    main()
