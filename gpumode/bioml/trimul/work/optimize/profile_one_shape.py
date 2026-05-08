"""Per-kernel breakdown for a single shape (default shape index 4: D=128, S=1024)."""

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
    idx = int(idx_args[0]) if idx_args else 4
    shape = mod.BENCHMARK_SHAPES[idx]
    print(f"# Per-kernel breakdown — shape {idx}: {shape}")
    inp = mod._make_input_from_shape(shape)
    fn = mod.custom_kernel
    for _ in range(3):
        fn(inp)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=False
    ) as p:
        for _ in range(20):
            fn(inp)
        torch.cuda.synchronize()
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
