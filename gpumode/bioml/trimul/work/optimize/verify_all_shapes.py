"""Run _check_vs_ref on all 7 BENCHMARK_SHAPES to confirm correctness."""

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


for i, shape in enumerate(mod.BENCHMARK_SHAPES):
    n_bad, max_err = mod._check_vs_ref(shape, seed=0, atol=2e-2, rtol=2e-2)
    flag = "OK" if n_bad == 0 else "FAIL"
    print(
        f"shape{i}: bs={shape['bs']} sl={shape['seqlen']} dim={shape['dim']} "
        f"hd={shape['hiddendim']} dist={shape['distribution']:6s} "
        f"n_bad={n_bad}, max_err={max_err:.5f}  {flag}"
    )
