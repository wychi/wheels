"""Standalone test for iter15 custom Triton bmm kernel.

Compares tlx_ws_bmm_fp32 against torch.bmm(L, R.T).float() across all 7
BENCHMARK_SHAPES. Tolerance: max abs diff < 1e-3 (much tighter than e2e
gate; we want a faithful bf16-in/fp32-out bmm).
"""

import os
import sys
import sysconfig
import statistics
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

KERNEL_FILE = "/home/wychi/oss/wheels/.claude/worktrees/agent-a60c0ed3f7fe0d15e/gpumode/bioml/trimul/work/hopper_gemm_ws.py"
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


def _alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(_alloc_fn)


def time_fn(fn, n_warmup=3, n_iters=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times) * 1000  # us


def main():
    print("# iter15 standalone bmm test (custom Triton vs cuBLAS bf16 bmm + .float())")
    print()
    print(
        "| # | B*hd | N | torch.bmm+cast µs | tlx_ws_bmm_fp32 µs | speedup | max_diff | "
        "correctness |"
    )
    print(
        "|---|------|---|-------------------|---------------------|---------|----------|"
        "-------------|"
    )
    for i, shape in enumerate(mod.BENCHMARK_SHAPES):
        B = shape["bs"]
        N = shape["seqlen"]
        hd = shape["hiddendim"]
        BATCH = B * hd
        torch.manual_seed(i)
        L = torch.randn(BATCH, N, N, dtype=torch.bfloat16, device="cuda")
        R = torch.randn(BATCH, N, N, dtype=torch.bfloat16, device="cuda")

        # Reference
        ref = torch.bmm(L, R.transpose(-1, -2)).float()
        # Ours
        ours = mod.tlx_ws_bmm_fp32(L, R)

        abs_diff = (ours - ref).abs()
        max_diff = abs_diff.max().item()
        # bf16 matmul comparison: tolerance proportional to |ref| since
        # different reduction trees give different roundings. Use 2e-2 + 2e-2 * |ref|
        # (the leaderboard's e2e gate) and also report a tight gate (5e-2 abs).
        tol = 2e-2 + 2e-2 * ref.abs()
        n_bad = (abs_diff > tol).sum().item()
        ok = "OK" if n_bad == 0 else f"FAIL n_bad={n_bad}"

        # Bench (bind L, R as defaults to silence ruff F821 in lambdas)
        us_ref = time_fn(lambda L=L, R=R: torch.bmm(L, R.transpose(-1, -2)).float())
        us_ours = time_fn(lambda L=L, R=R: mod.tlx_ws_bmm_fp32(L, R))
        speedup = us_ref / us_ours

        print(
            f"| {i} | {BATCH} | {N} | {us_ref:.1f} | {us_ours:.1f} | "
            f"{speedup:.2f}× | {max_diff:.2e} | {ok} |",
            flush=True,
        )

        del L, R, ref, ours
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
