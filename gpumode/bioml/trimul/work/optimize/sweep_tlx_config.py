"""Sweep TLX matmul configs and report bench + adversarial pass rate.

For iter11 — measures bench across all 7 BENCHMARK_SHAPES per config and
runs a small adversarial sweep to confirm precision didn't regress.
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


def _alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(_alloc_fn)
torch.backends.cuda.matmul.allow_tf32 = False

sys.path.insert(0, "/home/wychi/oss/wheels/gpumode/bioml/trimul/work/optimize")
from check_leaderboard_seeds import leaderboard_input  # noqa


def smem_bytes(BM, BN, BK, NUM_STAGES, NUM_MMA_GROUPS, dtype_bytes=2):
    """Approximate SMEM footprint for the WS GEMM (matches NCU 213 KB on baseline).

    A is shared across consumer warpgroups via cluster-broadcast (no MG factor).
    B per-stage. Barriers + scratch ~64 KB on this kernel.
    Hopper cap is 232,448 bytes (228 KB).
    """
    a = BM * BK * dtype_bytes * NUM_STAGES
    b = BK * BN * dtype_bytes * NUM_STAGES
    scratch = 64 * 1024  # observed: scratch + barriers ~64 KB on this WS kernel
    return a + b + scratch


def divisibility_ok(BM, BN, BK):
    """N=5*hd is always 5*128=640. K=D in {128,384}. M=bs*sl^2 (always >= 65536)."""
    if 640 % BN:
        return False, f"N=640 % BN={BN} != 0"
    if 128 % BK or 384 % BK:
        return False, f"K in {{128,384}} not divisible by BK={BK}"
    if 65536 % BM:
        return False, f"smallest M=65536 % BM={BM} != 0"
    return True, "ok"


def bench_config(cfg_dict, n_iters=10, n_warmup=3):
    """Override TLX_CONFIG, run all 7 shapes, return (geo_mean_speedup, per_shape)."""
    # Restore between configs.
    orig = dict(mod.TLX_CONFIG)
    mod.TLX_CONFIG.update(cfg_dict)
    # Reset weight cache (cached B_g shape may not match new BK divisibility)
    mod._W_CACHE.clear()

    per_shape = []
    try:
        for sh in mod.BENCHMARK_SHAPES:
            data = mod._make_input_from_shape(sh)
            for _ in range(n_warmup):
                mod.custom_kernel(data)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n_iters):
                mod.custom_kernel(data)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / n_iters
            per_shape.append((sh, ms))
            del data
            mod._W_CACHE.clear()
            torch.cuda.empty_cache()
    finally:
        mod.TLX_CONFIG.update(orig)
        mod._W_CACHE.clear()
    return per_shape


def adversarial_check(cfg_dict, n_trials=10, seed=731):
    """Quick sanity: 10 trials × 4 D=128 shapes (the precision-sensitive ones)."""
    orig = dict(mod.TLX_CONFIG)
    mod.TLX_CONFIG.update(cfg_dict)
    mod._W_CACHE.clear()

    shapes = [
        (2, 256, 128, 128, "normal"),
        (1, 768, 128, 128, "cauchy"),
        (1, 1024, 128, 128, "cauchy"),
        (2, 768, 128, 128, "normal"),
    ]
    fails_total = 0
    runs_total = 0
    try:
        for sh in shapes:
            for t in range(n_trials):
                d = leaderboard_input(*sh, seed=seed)
                ours = mod.custom_kernel(d).float()
                ref = mod._ref_kernel(d)
                ae = (ours - ref).abs()
                tol = 2e-2 + 2e-2 * ref.abs()
                if (ae > tol).any():
                    fails_total += 1
                runs_total += 1
                del ours, ref, d
                mod._W_CACHE.clear()
                torch.cuda.empty_cache()
    finally:
        mod.TLX_CONFIG.update(orig)
        mod._W_CACHE.clear()
    return fails_total, runs_total


CONFIGS = [
    # baseline (iter10b current)
    dict(
        name="baseline_iter10b", BM=256, BN=128, BK=64, NUM_STAGES=3, NUM_MMA_GROUPS=2
    ),
    # NS=4 with same M tile (more pipeline depth; A: 256*64*2*4=128KB + B:48KB +scratch =~240KB,
    # may not fit, but let's try)
    dict(name="BM256_NS4_MG2", BM=256, BN=128, BK=64, NUM_STAGES=4, NUM_MMA_GROUPS=2),
    # NS=4 with MG=1 (loses pingpong, may give back NS budget)
    dict(name="NS4_MG1", BM=256, BN=128, BK=64, NUM_STAGES=4, NUM_MMA_GROUPS=1),
    # smaller M, deeper pipe, keep pingpong
    dict(name="BM128_NS4_MG2", BM=128, BN=128, BK=64, NUM_STAGES=4, NUM_MMA_GROUPS=2),
    # NUM_STAGES=2 with bigger M tiles
    dict(name="BM256_NS2_MG2", BM=256, BN=128, BK=64, NUM_STAGES=2, NUM_MMA_GROUPS=2),
    # larger M tile (works for shapes 1+: M = 1*sl^2 >= 65536, 65536/512=128 — won't divide)
    # so 65536 % 512 = 0 actually (65536 = 512*128). Smallest sl=256 -> M=65536 for bs=1
    # but bs=2 sl=256 -> M=131072. So BM=512 only works for bs>=2 sl>=256 OR bs=1 sl>=362.
    # Skip — doesn't divide all shapes evenly.
    # smaller M with NS=5
    dict(name="BM128_NS5_MG2", BM=128, BN=128, BK=64, NUM_STAGES=5, NUM_MMA_GROUPS=2),
]


def main():
    print(f"{'config':30s}  | shape6 ms | sum 7 shapes ms | adv fails | SMEM KB")
    print("-" * 90)
    base_results = None
    for cfg in CONFIGS:
        name = cfg.pop("name")
        ok, reason = divisibility_ok(cfg["BM"], cfg["BN"], cfg["BK"])
        if not ok:
            print(f"{name:30s}  | SKIP — {reason}")
            cfg["name"] = name
            continue
        smem_kb = (
            smem_bytes(
                cfg["BM"],
                cfg["BN"],
                cfg["BK"],
                cfg["NUM_STAGES"],
                cfg["NUM_MMA_GROUPS"],
            )
            // 1024
        )
        if smem_kb > 228:
            print(f"{name:30s}  | SKIP — SMEM {smem_kb} KB > 228 KB")
            cfg["name"] = name
            continue
        try:
            per_shape = bench_config(cfg)
        except Exception as e:
            print(f"{name:30s}  | EXEC FAIL — {type(e).__name__}: {str(e)[:80]}")
            cfg["name"] = name
            continue
        shape6_ms = per_shape[6][1]
        total_ms = sum(p[1] for p in per_shape)
        # Adversarial check (quick)
        fails, runs = adversarial_check(cfg)
        if base_results is None:
            base_results = (shape6_ms, total_ms)
            relstr = ""
        else:
            rel6 = (shape6_ms / base_results[0] - 1) * 100
            reltotal = (total_ms / base_results[1] - 1) * 100
            relstr = f" ({rel6:+.1f}% / {reltotal:+.1f}%)"
        print(
            f"{name:30s}  | {shape6_ms:7.3f} ms | {total_ms:8.3f} ms{relstr} | "
            f"{fails}/{runs} | {smem_kb}"
        )
        cfg["name"] = name


if __name__ == "__main__":
    main()
