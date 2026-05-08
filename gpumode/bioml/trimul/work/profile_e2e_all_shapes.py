"""End-to-end profiling of custom_kernel across all 7 benchmark shapes.

Runs hopper_gemm_ws.py's full pipeline (LN-stats + 5-proj GEMM + gate-LN +
2x tr_fwd + bmm + invtr-LN-gate + F.linear) for each shape and reports
median wall time, throughput, and per-kernel breakdown via torch.profiler.
"""

import os
import sys
import sysconfig
import statistics

# Tell hopper_gemm_ws.py's _install_custom_deps() to skip pip install
sys.argv = [sys.argv[0], "--no-install"]

# uTLX setup BEFORE importing triton (kernel module's _setup_utlx() also does this)
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

# Load kernel module via exec, stubbing out the buggy _setup_utlx() (we already set things up)
import types

src = open(KERNEL_FILE).read()
src = src.replace("_setup_utlx()\n", "pass  # _setup_utlx() stubbed by wrapper\n")
mod = types.ModuleType("hopper_gemm_ws")
mod.__file__ = KERNEL_FILE
exec(compile(src, KERNEL_FILE, "exec"), mod.__dict__)
sys.modules["hopper_gemm_ws"] = mod

import torch


# Triton allocator for runtime-allocated scratch
def _alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(_alloc_fn)

# ----------------------------------------------------------------------------
# Per-shape timing
# ----------------------------------------------------------------------------


def time_shape(shape_idx: int, shape: dict, n_warmup: int = 3, n_iters: int = 30):
    """Time custom_kernel on one shape; return median ms and throughput."""
    inp = mod._make_input_from_shape(shape)
    fn = mod.custom_kernel

    # Warmup
    for _ in range(n_warmup):
        out = fn(inp)
    torch.cuda.synchronize()

    # Per-iter timing using cuda events
    times = []
    for _ in range(n_iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn(inp)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    ms = statistics.median(times)
    p10 = statistics.quantiles(times, n=10)[0] if len(times) >= 10 else min(times)
    p90 = statistics.quantiles(times, n=10)[-1] if len(times) >= 10 else max(times)

    # Compute roofline numbers for total pipeline
    B, S, D, H = shape["bs"], shape["seqlen"], shape["dim"], shape["hiddendim"]
    # Total FLOPs = S1 (5-proj) + einsum + S3 (to_out)
    flops_s1 = 5 * 2 * B * S * S * D * H
    flops_einsum = 2 * B * S * S * S * H
    flops_s3 = 2 * B * S * S * H * D
    flops_total = flops_s1 + flops_einsum + flops_s3
    tflops = flops_total / (ms / 1e3) / 1e12

    # Accuracy check vs reference
    out_actual = fn(inp)
    out_ref = mod._ref_kernel(inp)
    abs_err = (out_actual.float() - out_ref.float()).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()

    return {
        "idx": shape_idx,
        "shape": shape,
        "ms_med": ms,
        "ms_p10": p10,
        "ms_p90": p90,
        "tflops_e2e": tflops,
        "flops_total_g": flops_total / 1e9,
        "max_err": max_err,
        "mean_err": mean_err,
    }


# ----------------------------------------------------------------------------
# Per-kernel breakdown for one representative shape (largest, D=384)
# ----------------------------------------------------------------------------


def kernel_breakdown(shape: dict, n_iters: int = 20):
    inp = mod._make_input_from_shape(shape)
    fn = mod.custom_kernel
    # Warmup
    for _ in range(3):
        fn(inp)
    torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False) as p:
        for _ in range(n_iters):
            fn(inp)
        torch.cuda.synchronize()

    # Per-kernel CUDA time
    table = p.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=20,
    )
    return table


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    print(f"# TriMul end-to-end profile — {KERNEL_FILE}")
    print(
        f"# Triton {triton.__version__}, torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}"
    )
    print()

    print("## Per-shape end-to-end")
    print(
        "| # | bs | seqlen | dim | hd | mask | dist | ms (med) | ms (p10–p90) | TF/s e2e | max abs err | mean abs err |"
    )
    print(
        "|---|----|--------|-----|----|------|------|----------|--------------|----------|-------------|--------------|"
    )

    results = []
    for i, shape in enumerate(mod.BENCHMARK_SHAPES):
        try:
            r = time_shape(i, shape)
            results.append(r)
            mask_str = "no" if shape["nomask"] else "yes"
            print(
                f"| {i} | {shape['bs']} | {shape['seqlen']} | {shape['dim']} | "
                f"{shape['hiddendim']} | {mask_str} | {shape['distribution']} | "
                f"**{r['ms_med']:.3f}** | {r['ms_p10']:.3f}–{r['ms_p90']:.3f} | "
                f"{r['tflops_e2e']:.1f} | {r['max_err']:.4g} | {r['mean_err']:.4g} |",
                flush=True,
            )
        except Exception as e:
            print(f"| {i} | -- shape={shape} FAILED: {e} -- |", flush=True)

    print()

    # Per-kernel breakdown for largest shape
    largest = mod.BENCHMARK_SHAPES[-1]  # B=1, S=1024, D=384
    print(f"## Per-kernel breakdown — largest shape: {largest}")
    print("```")
    try:
        table = kernel_breakdown(largest)
        print(table)
    except Exception as e:
        print(f"FAILED: {e}")
    print("```")


if __name__ == "__main__":
    main()
