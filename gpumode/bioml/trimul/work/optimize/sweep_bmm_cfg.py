"""Sweep BM/BN/BK/NUM_STAGES/GROUP_SIZE_M for `bmm_kernel_tlx_ws` on shape 6.

PLAN_v5 §4 / iter31 / B3 — bmm tile-config sweep.

Modeled on `sweep_final_linear_cfg.py`. Calls the kernel directly with
overridden configs to find the best tile shape, skipping configs that
exceed the H100 SMEM cap (232,448 B/CTA) or violate WGMMA / TMA
constraints.

Locked axes (structural to the kernel — would require kernel surgery to
sweep, out of scope for a config-only iter):
  - NUM_MMA_GROUPS = 2  (producer issues 2 separate A loads, indexed by
                         `tlx.async_task_replica_id()`; barriers sized as
                         NUM_STAGES * NUM_MMA_GROUPS)
  - replicate = 2       (consumer warpgroups; same indexing as MMA_GROUPS)

Sweep axes:
  - BM ∈ {128, 256}   (BM=64 violates WGMMA min: BLOCK_M_SPLIT=32 < 64)
  - BN ∈ {64, 128, 256}   (must be power of 2 for K=N TMA + local_trans)
  - BK ∈ {32, 64, 128}
  - NUM_STAGES ∈ {2, 3, 4}
  - GROUP_SIZE_M ∈ {1, 4, 8}
"""

import os
import statistics
import sys
import sysconfig
import types

sys.argv = [sys.argv[0], "--no-install"]
dist_packages = sysconfig.get_paths()["purelib"]
os.environ["TRITON_PLUGIN_PATHS"] = os.path.join(
    dist_packages, "utlx_plugin", "libutlx.so"
)

import triton  # noqa: E402
import utlx_plugin  # noqa: E402, F401

sys.path.insert(0, "/home/wychi/oss/wheels/runner")
import tlx_patches  # noqa: E402

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

import torch  # noqa: E402


def _alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(_alloc_fn)


SMEM_CAP = 232_448  # H100 cap per CTA (PLAN_v4 C4)
NUM_MMA_GROUPS = 2  # locked — see module docstring


def smem_bytes(BM, BN, BK, NS):
    """Approximate SMEM footprint matching the kernel's allocations.

    a: (BLOCK_M_SPLIT, BK) bf16, NS * NUM_MMA_GROUPS slots
    b: (BN, BK)            bf16, NS slots
    Barriers and other slop ~1 KiB.
    """
    BLOCK_M_SPLIT = BM // NUM_MMA_GROUPS
    a_bytes = NS * NUM_MMA_GROUPS * BLOCK_M_SPLIT * BK * 2
    b_bytes = NS * BN * BK * 2
    return a_bytes + b_bytes + 1024


def valid_config(BM, BN, BK, NS, GSM, N):
    BLOCK_M_SPLIT = BM // NUM_MMA_GROUPS
    if BLOCK_M_SPLIT < 64:
        return False, "BLOCK_M_SPLIT<64 (WGMMA)"
    if BN & (BN - 1) != 0:
        return False, "BN not power-of-2"
    if N % BM != 0:
        return False, f"N={N} not divisible by BM={BM}"
    if N % BN != 0:
        return False, f"N={N} not divisible by BN={BN}"
    if N % BK != 0:
        return False, f"N={N} not divisible by BK={BK}"
    sb = smem_bytes(BM, BN, BK, NS)
    if sb > SMEM_CAP:
        return False, f"SMEM {sb} > cap {SMEM_CAP}"
    return True, ""


def call_one(L, R, out, cfg, BATCH, N):
    num_tiles = BATCH * triton.cdiv(N, cfg["BM"]) * triton.cdiv(N, cfg["BN"])
    grid = (min(mod.NUM_SMS, num_tiles),)
    mod.bmm_kernel_tlx_ws[grid](
        L,
        R,
        out,
        BATCH,
        N,
        NUM_SMS=mod.NUM_SMS,
        num_stages=1,
        num_warps=4,
        **cfg,
    )


def time_fn(fn, n_warmup=5, n_iters=30):
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


def bench_cfg(L, R, BATCH, N, cfg):
    out = torch.empty((BATCH, N, N), dtype=torch.float32, device="cuda")
    # compile/warmup
    try:
        call_one(L, R, out, cfg, BATCH, N)
        torch.cuda.synchronize()
    except Exception as e:
        return None, f"COMPILE_FAIL {type(e).__name__}: {str(e)[:80]}"
    # correctness vs torch.bmm reference (loose tol — bf16 reductions)
    ref = torch.bmm(L, R.transpose(-1, -2)).float()
    out2 = torch.empty_like(out)
    call_one(L, R, out2, cfg, BATCH, N)
    torch.cuda.synchronize()
    abs_diff = (ref - out2).abs()
    tol = 2e-2 + 2e-2 * ref.abs()
    n_bad = (abs_diff > tol).sum().item()
    if n_bad > 0:
        return None, f"WRONG n_bad={n_bad} max={abs_diff.max().item():.2e}"
    us = time_fn(lambda: call_one(L, R, out, cfg, BATCH, N))
    return us, None


def main():
    # shape 6 dims: bs=1, hd=128, sl=1024, dim=384
    BATCH, N = 128, 1024
    print(f"# Shape 6 bmm sweep: BATCH={BATCH}, N={N}")

    torch.manual_seed(31)
    L = torch.randn(BATCH, N, N, dtype=torch.bfloat16, device="cuda")
    R = torch.randn(BATCH, N, N, dtype=torch.bfloat16, device="cuda")

    # baseline
    baseline_cfg = dict(
        BM=128, BN=128, BK=64, GROUP_SIZE_M=8, NUM_STAGES=3, NUM_MMA_GROUPS=2
    )
    base_us, base_err = bench_cfg(L, R, BATCH, N, baseline_cfg)
    if base_err:
        print(f"BASELINE FAILED: {base_err}")
        return
    print(f"# Baseline (BM128 BN128 BK64 NS3 GSM8): {base_us:.1f} µs\n")

    # ref (torch.bmm)
    ref_us = time_fn(lambda: torch.bmm(L, R.transpose(-1, -2)).float())
    print(f"# torch.bmm+cast reference: {ref_us:.1f} µs\n")

    # First pass: focused sweep. Drop GSM=4 (usually between 1 and 8).
    # 2 BM × 3 BN × 3 BK × 3 NS × 2 GSM = 108 raw → ~86 valid after SMEM/WGMMA filters.
    # Allow `--quick` to cut further (drop BK=32 and NS=4) for fast iteration.
    quick = "--quick" in sys.argv
    if quick:
        BM_OPT = [128, 256]
        BN_OPT = [64, 128, 256]
        BK_OPT = [64, 128]
        NS_OPT = [2, 3]
        GSM_OPT = [1, 8]
    else:
        BM_OPT = [128, 256]
        BN_OPT = [64, 128, 256]
        BK_OPT = [32, 64, 128]
        NS_OPT = [2, 3, 4]
        GSM_OPT = [1, 8]

    print(
        f"{'BM':>4} {'BN':>4} {'BK':>4} {'NS':>3} {'GSM':>4}  "
        f"{'SMEM':>7}  {'µs':>8}  {'vs base':>8}  note"
    )

    results = []
    skipped = []
    for BM in BM_OPT:
        for BN in BN_OPT:
            for BK in BK_OPT:
                for NS in NS_OPT:
                    for GSM in GSM_OPT:
                        ok, why = valid_config(BM, BN, BK, NS, GSM, N)
                        sb = smem_bytes(BM, BN, BK, NS)
                        if not ok:
                            skipped.append((BM, BN, BK, NS, GSM, why))
                            continue
                        cfg = dict(
                            BM=BM,
                            BN=BN,
                            BK=BK,
                            GROUP_SIZE_M=GSM,
                            NUM_STAGES=NS,
                            NUM_MMA_GROUPS=2,
                        )
                        us, err = bench_cfg(L, R, BATCH, N, cfg)
                        if err is not None:
                            print(
                                f"{BM:4d} {BN:4d} {BK:4d} {NS:3d} {GSM:4d}  "
                                f"{sb:7d}  ----      ----     {err}",
                                flush=True,
                            )
                            continue
                        ratio = us / base_us
                        marker = "**" if ratio < 0.99 else "  "
                        print(
                            f"{BM:4d} {BN:4d} {BK:4d} {NS:3d} {GSM:4d}  "
                            f"{sb:7d}  {us:8.1f}  {ratio:7.4f}x  {marker}",
                            flush=True,
                        )
                        results.append((us, ratio, BM, BN, BK, NS, GSM, sb))

    print(f"\n# Skipped {len(skipped)} configs:")
    for BM, BN, BK, NS, GSM, why in skipped[:8]:
        print(f"#   BM{BM} BN{BN} BK{BK} NS{NS} GSM{GSM} → {why}")
    if len(skipped) > 8:
        print(f"#   ... ({len(skipped) - 8} more)")

    print("\n# Top 10 configs:")
    results.sort()
    print(
        f"{'rank':>4}  {'BM':>4} {'BN':>4} {'BK':>4} {'NS':>3} {'GSM':>4}  "
        f"{'SMEM':>7}  {'µs':>8}  {'vs base':>8}"
    )
    for i, (us, ratio, BM, BN, BK, NS, GSM, sb) in enumerate(results[:10]):
        print(
            f"{i + 1:4d}  {BM:4d} {BN:4d} {BK:4d} {NS:3d} {GSM:4d}  "
            f"{sb:7d}  {us:8.1f}  {ratio:7.4f}x"
        )


if __name__ == "__main__":
    main()
