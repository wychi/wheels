"""Wrapper: setup uTLX, exec repro, then build args ONCE and call kernel in tight loop."""
import os, sys, sysconfig
dist_packages = sysconfig.get_paths()["purelib"]
os.environ["TRITON_PLUGIN_PATHS"] = os.path.join(dist_packages, "utlx_plugin", "libutlx.so")
import triton, utlx_plugin
sys.path.insert(0, "/home/wychi/oss/wheels/runner")
import tlx_patches
tlx_patches.apply(tlx_patches.resolve_for_kernel("/home/wychi/oss/wheels/gpumode/bioml/trimul/work/hopper_gemm_ws.py"))

repro_path = "/home/wychi/.mpp_captures/matmul_kernel_tlx_ws_e3d8dab24f76/repro_line1_mpp_20260507224206.py"
src = open(repro_path).read()
src = src.replace('if __name__ == "__main__":\n    main()\n', '')
ns = {"__name__": "__repro_mod__", "__file__": repro_path}
exec(compile(src, repro_path, "exec"), ns)

import torch
from pathlib import Path

# Build args ONCE
def _alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device='cuda')
triton.set_allocator(_alloc_fn)

script_dir = Path(repro_path).resolve().parent
json_file = script_dir / "repro_line1_context_20260507224206.json"
grid, args_dict = ns["create_args_from_json_file"](str(json_file))
kernel = ns["imported_kernel_function"]
ir_override_file = ns["_IR_OVERRIDE_FILE"]
print(f"[wrap] grid={grid}, ir_override={ir_override_file}", file=sys.stderr)

def kfn():
    kernel[tuple(grid)](
        a_ptr=args_dict["a_ptr"], b_ptr=args_dict["b_ptr"], c_ptr=args_dict["c_ptr"],
        M=args_dict["M"], N=args_dict["N"], K=args_dict["K"],
    )

# Warmup
for _ in range(5):
    kfn()
torch.cuda.synchronize()

# Per-iter timing using cuda events
import time
print("=== Per-call cuda-event times ===", file=sys.stderr)
times = []
for i in range(15):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    kfn()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e)
    times.append(ms)
    print(f"  iter {i}: {ms:.4f} ms", file=sys.stderr)

import statistics as st
ms = st.median(times)
M, N, K = 131072, 640, 128
flops = 2 * M * N * K
bytes_total = (M*K + K*N + M*N) * 2
sec = ms / 1e3
tflops = flops / sec / 1e12
gbs = bytes_total / sec / 1e9
print(f"[wrap] median: {ms:.4f} ms   {tflops:.2f} TF/s   {gbs:.1f} GB/s", file=sys.stderr)
print(f"[wrap] roofline: peak bf16 989 TF/s, peak HBM 3350 GB/s")
print(f"[wrap]           tflops/peak = {tflops/989*100:.1f}%   bw/peak = {gbs/3350*100:.1f}%")
print(f"[wrap] arithmetic intensity = {flops/bytes_total:.2f} FLOP/byte (H100 ridge ≈ 295 FLOP/byte for bf16)")
