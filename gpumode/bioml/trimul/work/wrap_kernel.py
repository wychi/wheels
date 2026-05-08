"""Run uTLX setup first, then init tritonparse, then run smoke test."""
import os, sys, importlib.util

# Load the user's kernel script as a module (this triggers _setup_utlx() before
# any triton import)
spec = importlib.util.spec_from_file_location(
    "hopper_gemm_ws", "/home/wychi/oss/wheels/gpumode/bioml/trimul/work/hopper_gemm_ws.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["hopper_gemm_ws"] = mod
spec.loader.exec_module(mod)
print(f"[wrap] kernel module loaded; triton is set up", file=sys.stderr)

# NOW init tritonparse (after triton is imported)
import tritonparse.structured_logging as tsl
trace_dir = os.environ.get("TRITON_TRACE")
print(f"[wrap] tritonparse.init({trace_dir!r})", file=sys.stderr)
tsl.init(trace_dir)

# Run the smoke test (or BENCHMARK_SHAPES[-1] for a heavier load)
print("[wrap] running smoke test...", file=sys.stderr)
mod._smoke_test()
