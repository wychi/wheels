"""Match the leaderboard's `generate_input` exactly (weights from global RNG)
and search many seeds to find correctness failures.

Two CLI modes:

  Single-shape (legacy):
    python check_leaderboard_seeds.py [bs] [sl] [dim] [hd] [dist] [n_trials]

  Tiered (PLAN_v4 §3):
    python check_leaderboard_seeds.py --tier T0|T1|T2|T3
       T0 — shape 4 only, seed 731, 1 trial      (~5 s, smoke)
       T1 — shapes {1,4,6}, seeds {731,17}, 5    (~30 s, per-iter sentinel)
       T2 — shapes {0,1,4,6}, 3 seeds, 8 trials  (~2 min, mid-iter precision)
       T3 — all 7 shapes, 6 seeds, 30 trials     (~10 min, ship-grade)
"""

import argparse
import math
import os
import sys
import sysconfig
import types

# Strip our flags out of sys.argv so the kernel module doesn't see them.
_user_args = sys.argv[1:]
sys.argv = [sys.argv[0], "--no-install"]
dist_packages = sysconfig.get_paths()["purelib"]
os.environ["TRITON_PLUGIN_PATHS"] = os.path.join(
    dist_packages, "utlx_plugin", "libutlx.so"
)

import triton  # noqa
import utlx_plugin  # noqa

sys.path.insert(0, "/home/wychi/oss/wheels/runner")
import tlx_patches

KERNEL_FILE = os.environ.get(
    "TRIMUL_KERNEL",
    "/home/wychi/oss/wheels/gpumode/bioml/trimul/work/hopper_gemm_ws.py",
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

import torch  # noqa


def _alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(_alloc_fn)


def leaderboard_input(bs, sl, dim, hd, dist, seed, nomask=True):
    """Mirrors reference-kernels/.../trimul/reference.py:generate_input."""
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


def _shape_label(s):
    return (
        f"bs={s['bs']} sl={s['seqlen']:4d} dim={s['dim']} "
        f"hd={s['hiddendim']} dist={s['distribution']:6s}"
    )


def _shape_to_args(s):
    return dict(
        bs=s["bs"],
        sl=s["seqlen"],
        dim=s["dim"],
        hd=s["hiddendim"],
        dist=s["distribution"],
        nomask=s["nomask"],
    )


def _run_one_trial(shape_args, seed):
    """Returns (n_bad, max_err)."""
    data = leaderboard_input(seed=seed, **shape_args)
    ours = mod.custom_kernel(data).float()
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref = mod._ref_kernel(data)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    abs_err = (ours - ref).abs()
    tol = 2e-2 + 2e-2 * ref.abs()
    n_bad = (abs_err > tol).sum().item()
    max_err = abs_err.max().item()
    del ours, ref, data
    torch.cuda.empty_cache()
    return n_bad, max_err


# Tier matrices per PLAN_v4 §3.
TIER_MATRICES = {
    "T0": {"shape_idxs": [4], "seeds": [731], "trials": 1},
    "T1": {"shape_idxs": [1, 4, 6], "seeds": [731, 17], "trials": 5},
    "T2": {"shape_idxs": [0, 1, 4, 6], "seeds": [731, 17, 99], "trials": 8},
    "T3": {
        "shape_idxs": list(range(7)),
        "seeds": [731, 17, 99, 1, 2026, 9999],
        "trials": 30,
    },
}

# Pass criteria: tier-name → max allowed adversarial fail rate.
TIER_PASS_RATE = {"T0": 0.0, "T1": 0.05, "T2": 0.01, "T3": 0.015}


def run_tier(tier):
    matrix = TIER_MATRICES[tier]
    pass_threshold = TIER_PASS_RATE[tier]
    shape_idxs = matrix["shape_idxs"]
    seeds = matrix["seeds"]
    trials = matrix["trials"]

    print(
        f"# tier={tier} shapes={shape_idxs} seeds={seeds} trials={trials}/seed "
        f"pass_threshold≤{pass_threshold * 100:.2f}%"
    )

    BENCHMARK_SHAPES = mod.BENCHMARK_SHAPES
    total_runs = 0
    total_fails = 0
    worst_max_err = 0.0
    worst_label = None
    per_shape = {}

    for sidx in shape_idxs:
        s = BENCHMARK_SHAPES[sidx]
        shape_args = _shape_to_args(s)
        shape_runs = 0
        shape_fails = 0
        shape_max_err = 0.0
        for seed in seeds:
            for t in range(trials):
                n_bad, max_err = _run_one_trial(shape_args, seed)
                shape_runs += 1
                total_runs += 1
                if n_bad > 0:
                    shape_fails += 1
                    total_fails += 1
                if max_err > shape_max_err:
                    shape_max_err = max_err
                if max_err > worst_max_err:
                    worst_max_err = max_err
                    worst_label = (
                        f"shape{sidx} seed={seed} trial={t} "
                        f"n_bad={n_bad} max_err={max_err:.5f}"
                    )
        per_shape[sidx] = (shape_fails, shape_runs, shape_max_err)
        flag = "OK " if shape_fails / shape_runs <= pass_threshold else "FAIL"
        print(
            f"  {flag} shape{sidx} {_shape_label(s)} → "
            f"{shape_fails}/{shape_runs} fail "
            f"({100 * shape_fails / shape_runs:.2f}%), max_err={shape_max_err:.5f}"
        )

    rate = total_fails / total_runs if total_runs else 0.0
    verdict = "PASS" if rate <= pass_threshold else "FAIL"
    print(
        f"# {verdict} tier={tier} {total_fails}/{total_runs} "
        f"({rate * 100:.2f}%) worst: {worst_label}"
    )
    return 0 if verdict == "PASS" else 1


def main():
    args_only = [a for a in _user_args if a != "--no-install"]
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--tier", choices=list(TIER_MATRICES.keys()))
    parser.add_argument("rest", nargs="*")
    parsed, _unknown = parser.parse_known_args(args_only)

    if parsed.tier:
        rc = run_tier(parsed.tier)
        sys.exit(rc)

    # Legacy single-shape path.
    a = parsed.rest
    bs = int(a[0]) if len(a) > 0 else 2
    sl = int(a[1]) if len(a) > 1 else 768
    dim = int(a[2]) if len(a) > 2 else 128
    hd = int(a[3]) if len(a) > 3 else 128
    dist = a[4] if len(a) > 4 else "normal"
    n_trials = int(a[5]) if len(a) > 5 else 20

    print(f"# bs={bs} sl={sl} dim={dim} hd={hd} dist={dist} n_trials={n_trials}")
    fails = 0
    max_err_global = 0.0
    worst = None
    shape_args = dict(bs=bs, sl=sl, dim=dim, hd=hd, dist=dist, nomask=True)
    for trial in range(n_trials):
        n_bad, max_err = _run_one_trial(shape_args, seed=731)
        if n_bad > 0:
            fails += 1
            print(f"  trial {trial}: n_bad={n_bad}, max_err={max_err:.5f}  FAIL")
        else:
            print(f"  trial {trial}: n_bad={n_bad}, max_err={max_err:.5f}  ok")
        if max_err > max_err_global:
            max_err_global = max_err
            worst = (trial, n_bad, max_err)
    print(
        f"# Summary: {fails}/{n_trials} failures; worst trial {worst} "
        f"max_err={max_err_global:.5f}"
    )


if __name__ == "__main__":
    main()
