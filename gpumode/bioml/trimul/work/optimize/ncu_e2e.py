"""NCU harness — runs custom_kernel on a single shape with cudaProfilerStart/Stop
gating so NCU only samples the steady-state iters (skipping JIT compile warmup).

Usage:
    ncu --profile-from-start off --target-processes all \
        --set basic --csv \
        python work/optimize/ncu_e2e.py [shape_idx] [num_profile_iters]
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
    idx_args = [a for a in _user_args if a != "--no-install"]
    idx = int(idx_args[0]) if idx_args else 6
    n_iters = int(idx_args[1]) if len(idx_args) > 1 else 1

    shape = mod.BENCHMARK_SHAPES[idx]
    print(
        f"# NCU e2e on shape {idx}: {shape}, n_profile_iters={n_iters}",
        file=sys.stderr,
    )
    inp = mod._make_input_from_shape(shape)
    fn = mod.custom_kernel

    # Warmup JIT and weight cache outside the profiled region.
    for _ in range(5):
        fn(inp)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(n_iters):
        fn(inp)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
