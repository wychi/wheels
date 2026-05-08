"""Standalone benchmark: tlx_ws_final_linear vs cuBLAS F.linear.

Measures only the GEMM in isolation on shape-6 dimensions to decide whether
the custom kernel is competitive (PLAN_v4 iter23 abort gate: within 5% means
launch noise will eat any e2e win).
"""

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

KERNEL_FILE = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "hopper_gemm_ws.py")
)
tlx_patches.apply(tlx_patches.resolve_for_kernel(KERNEL_FILE))
src = (
    open(KERNEL_FILE)
    .read()
    .replace("_setup_utlx()\n", "pass  # _setup_utlx() stubbed by wrapper\n")
)
mod = types.ModuleType("hopper_gemm_ws")
mod.__file__ = KERNEL_FILE
exec(compile(src, KERNEL_FILE, "exec"), mod.__dict__)

import torch
import torch.nn.functional as F


def _alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(_alloc_fn)


def bench(label, fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) / iters * 1000
    print(f"  {label:30s} {us:8.2f} µs")
    return us


# Shape 6 final-linear dims:
#   gated  bf16 [T=1*1024^2, hd=128]
#   w_out  bf16 [D=384, hd=128]
#   out    bf16 [T, D=384]
for label, T, K, N in [
    ("shape 6 (T=1048576, K=128, N=384)", 1024 * 1024, 128, 384),
    ("shape 5 (T=589824,  K=128, N=384)", 768 * 768, 128, 384),
    ("shape 2 (T=131072,  K=128, N=384)", 2 * 256 * 256, 128, 384),
]:
    print(f"\n{label}")
    g = torch.randn(T, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # warm tlx kernel via single call to compile
    _ = mod.tlx_ws_final_linear(g, w)

    cublas_us = bench("F.linear (cuBLAS)", lambda: F.linear(g, w))
    tlx_us = bench("tlx_ws_final_linear", lambda: mod.tlx_ws_final_linear(g, w))
    print(f"  ratio tlx/cublas = {tlx_us / cublas_us:.3f}")

    # Correctness check
    a = F.linear(g, w)
    b = mod.tlx_ws_final_linear(g, w)
    err = (a.float() - b.float()).abs().max().item()
    print(f"  max abs diff = {err:.5g}")
