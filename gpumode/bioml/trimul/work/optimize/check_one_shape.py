"""Verify one custom-shape against the fp32 reference, printing max_err and bad indices.

Usage:
    python check_one_shape.py [bs] [sl] [dim] [hd] [dist] [seed]

Defaults reproduce the gpumode-server failure: 2 768 128 128 normal 731.
"""

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


def main():
    args = [a for a in _user_args if a != "--no-install"]
    bs = int(args[0]) if len(args) > 0 else 2
    sl = int(args[1]) if len(args) > 1 else 768
    dim = int(args[2]) if len(args) > 2 else 128
    hd = int(args[3]) if len(args) > 3 else 128
    dist = args[4] if len(args) > 4 else "normal"
    seed = int(args[5]) if len(args) > 5 else 731

    shape = {
        "bs": bs,
        "seqlen": sl,
        "dim": dim,
        "hiddendim": hd,
        "distribution": dist,
        "nomask": True,
    }
    print(f"# Shape: {shape}, seed={seed}")

    n_bad, max_err = mod._check_vs_ref(shape, seed=seed, atol=2e-2, rtol=2e-2)
    print(f"n_bad={n_bad}, max_err={max_err:.6f}")

    # Detailed worst offenders
    data = mod._make_input_from_shape(shape, seed=seed)
    ours = mod.custom_kernel(data).float()
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref = mod._ref_kernel(data)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    abs_err = (ours - ref).abs()
    tol = 2e-2 + 2e-2 * ref.abs()
    bad = abs_err > tol
    if bad.any():
        idxs = bad.nonzero()
        print(f"# {idxs.shape[0]} bad elements; channel histogram (last dim):")
        ch = idxs[:, -1]
        unique, counts = torch.unique(ch, return_counts=True)
        for c, n in zip(unique.tolist(), counts.tolist()):
            print(f"  channel {c}: {n}")
        print("# top 10 worst offenders:")
        worst = abs_err.flatten().topk(10)
        for k in range(min(10, idxs.shape[0])):
            i = idxs[k].tolist()
            o = ours[tuple(i)].item()
            r = ref[tuple(i)].item()
            print(
                f"  {tuple(i)}: ours={o:.6g} ref={r:.6g} diff={abs(o - r):.6g} "
                f"tol={2e-2 + 2e-2 * abs(r):.6g}"
            )


if __name__ == "__main__":
    main()
