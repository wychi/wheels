"""Match the leaderboard's `generate_input` exactly (weights from global RNG)
and search many seeds to find correctness failures.

Usage:
    python check_leaderboard_seeds.py [bs] [sl] [dim] [hd] [dist] [n_trials]
"""

import math
import os
import sys
import sysconfig
import types

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


def main():
    args = [a for a in _user_args if a != "--no-install"]
    bs = int(args[0]) if len(args) > 0 else 2
    sl = int(args[1]) if len(args) > 1 else 768
    dim = int(args[2]) if len(args) > 2 else 128
    hd = int(args[3]) if len(args) > 3 else 128
    dist = args[4] if len(args) > 4 else "normal"
    n_trials = int(args[5]) if len(args) > 5 else 20

    print(f"# bs={bs} sl={sl} dim={dim} hd={hd} dist={dist} n_trials={n_trials}")
    fails = 0
    max_err_global = 0.0
    worst = None
    for trial in range(n_trials):
        # Same seed every trial — only weights vary because they use the global RNG.
        data = leaderboard_input(bs, sl, dim, hd, dist, seed=731)
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
        if n_bad > 0:
            fails += 1
            print(f"  trial {trial}: n_bad={n_bad}, max_err={max_err:.5f}  FAIL")
            bad = abs_err > tol
            idxs = bad.nonzero()
            ch = idxs[:, -1]
            unique, counts = torch.unique(ch, return_counts=True)
            print(f"    channels: {dict(zip(unique.tolist(), counts.tolist()))}")
            for k in range(min(5, idxs.shape[0])):
                i = idxs[k].tolist()
                o = ours[tuple(i)].item()
                r = ref[tuple(i)].item()
                print(f"    {tuple(i)}: ours={o:.6g} ref={r:.6g} diff={abs(o - r):.6g}")
        else:
            print(f"  trial {trial}: n_bad={n_bad}, max_err={max_err:.5f}  ok")
        if max_err > max_err_global:
            max_err_global = max_err
            worst = (trial, n_bad, max_err)
        del ours, ref, data
        torch.cuda.empty_cache()
    print(
        f"# Summary: {fails}/{n_trials} failures; worst trial {worst} max_err={max_err_global:.5f}"
    )


if __name__ == "__main__":
    main()
