"""NCU wrapper: setup uTLX, build args once, then launch ONE kernel call (NCU profiles it)."""

import os
import sys
import sysconfig

dist_packages = sysconfig.get_paths()["purelib"]
os.environ["TRITON_PLUGIN_PATHS"] = os.path.join(
    dist_packages, "utlx_plugin", "libutlx.so"
)
import triton

sys.path.insert(0, "/home/wychi/oss/wheels/runner")
import tlx_patches

tlx_patches.apply(
    tlx_patches.resolve_for_kernel(
        "/home/wychi/oss/wheels/gpumode/bioml/trimul/work/hopper_gemm_ws.py"
    )
)

repro_path = "/home/wychi/.mpp_captures/matmul_kernel_tlx_ws_e3d8dab24f76/repro_line1_mpp_20260507224206.py"
src = open(repro_path).read().replace('if __name__ == "__main__":\n    main()\n', "")
ns = {"__name__": "__repro_mod__", "__file__": repro_path}
exec(compile(src, repro_path, "exec"), ns)

import torch
from pathlib import Path


def _alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


triton.set_allocator(_alloc_fn)

script_dir = Path(repro_path).resolve().parent
json_file = script_dir / "repro_line1_context_20260507224206.json"
grid, args_dict = ns["create_args_from_json_file"](str(json_file))
kernel = ns["imported_kernel_function"]


def kfn():
    kernel[tuple(grid)](
        a_ptr=args_dict["a_ptr"],
        b_ptr=args_dict["b_ptr"],
        c_ptr=args_dict["c_ptr"],
        M=args_dict["M"],
        N=args_dict["N"],
        K=args_dict["K"],
    )


# Warmup (JIT compile + autotune)
for _ in range(3):
    kfn()
torch.cuda.synchronize()

# NCU range to profile
torch.cuda.cudart().cudaProfilerStart()
for _ in range(3):
    kfn()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
