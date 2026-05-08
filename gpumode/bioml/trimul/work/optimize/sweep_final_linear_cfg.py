"""Sweep BM/BN/NUM_STAGES/GROUP_SIZE_M for tlx_ws_final_linear on shape 6.

This bypasses the Python-level wrapper and calls the kernel directly with
overridden configs to find the best tile shape.
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


def run_one(M, N, K, cfg):
    g = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    num_tiles = triton.cdiv(M, cfg["BM"]) * triton.cdiv(N, cfg["BN"])
    grid = (min(mod.NUM_SMS, num_tiles),)

    def call():
        mod.final_linear_kernel_tlx_ws[grid](
            g,
            w,
            out,
            M,
            N,
            K,
            NUM_SMS=mod.NUM_SMS,
            num_stages=1,
            num_warps=4,
            **cfg,
        )

    # warmup compile
    try:
        call()
    except Exception as e:
        return None, str(e)[:80]
    # Correctness vs cuBLAS
    a = F.linear(g, w)
    err = (a.float() - out.float()).abs().max().item()
    if err > 1.0:
        return None, f"WRONG err={err}"

    # bench
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        call()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 50 * 1000, None


def cublas_bench(M, N, K):
    g = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    for _ in range(10):
        F.linear(g, w)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        F.linear(g, w)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 50 * 1000


# shape 6 dims
M, K, N = 1024 * 1024, 128, 384
print(f"Shape 6: M={M}, N={N}, K={K}")

cublas_us = cublas_bench(M, N, K)
print(f"cuBLAS F.linear baseline: {cublas_us:.2f} µs\n")

# (BM, BN, NS, MG, GROUP_SIZE_M)
configs = [
    (128, 128, 3, 2, 8),  # original
    (128, 128, 4, 2, 8),
    (128, 128, 3, 2, 1),
    (128, 192, 3, 2, 1),  # 2 N-tiles
    (128, 192, 3, 2, 8),
    (128, 384, 2, 2, 1),  # 1 N-tile per row
    (128, 384, 3, 2, 1),
    (256, 128, 3, 2, 1),
    (256, 128, 3, 2, 8),
    (256, 192, 3, 2, 1),
    (64, 128, 3, 2, 8),
    (64, 192, 3, 2, 1),
    (64, 384, 3, 2, 1),
]
print(
    f"{'BM':>4} {'BN':>4} {'NS':>3} {'MG':>3} {'GS':>3}  {'µs':>8}  vs cuBLAS  {'note':<40}"
)
for bm, bn, ns, mg, gs in configs:
    cfg = dict(BM=bm, BN=bn, BK=64, GROUP_SIZE_M=gs, NUM_STAGES=ns, NUM_MMA_GROUPS=mg)
    us, err = run_one(M, N, K, cfg)
    if err is not None:
        print(f"{bm:4d} {bn:4d} {ns:3d} {mg:3d} {gs:3d}  ----      ----      {err}")
    else:
        ratio = us / cublas_us
        marker = "**" if ratio < 1.0 else "  "
        print(
            f"{bm:4d} {bn:4d} {ns:3d} {mg:3d} {gs:3d}  {us:8.2f}  {ratio:6.3f}x  {marker}"
        )
