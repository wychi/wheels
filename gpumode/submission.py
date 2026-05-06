#!/usr/bin/env python3
"""
Submission to verify patched Triton wheel with uTLX extension support.
"""

import subprocess
import sys

TRITON_WHEEL_URL = "https://github.com/wychi/wheels/releases/download/v0.2.0/triton-3.7.0+gitb7fa781f-cp313-cp313-linux_x86_64.whl"
UTLX_WHEEL_URL = "https://github.com/plotfi/plotfi-wheels/raw/main/utlx-0.1.0-py3-none-any.whl"

print(f"[DEBUG] Python: {sys.version}", file=sys.stderr)
print(f"[DEBUG] Installing triton from: {TRITON_WHEEL_URL}", file=sys.stderr)
result = subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall",
                         f"triton @ {TRITON_WHEEL_URL}",
                         f"utlx @ {UTLX_WHEEL_URL}"],
                        capture_output=True, text=True)
print(f"[DEBUG] pip exit code: {result.returncode}", file=sys.stderr)
print(f"[DEBUG] pip stdout: {result.stdout[-500:]}", file=sys.stderr)
if result.returncode != 0:
    print(f"[DEBUG] pip stderr: {result.stderr[-500:]}", file=sys.stderr)
    sys.exit(1)

import os
import sysconfig

dist_packages = sysconfig.get_paths()["purelib"]
libutlx_path = os.path.join(dist_packages, "utlx_plugin", "libutlx.so")
assert os.path.isfile(libutlx_path), f"libutlx.so not found at {libutlx_path}"
os.environ["TRITON_PLUGIN_PATHS"] = libutlx_path

for binary in ("ptxas", "ptxas-blackwell"):
    p = os.path.join(dist_packages, "triton", "backends", "nvidia", "bin", binary)
    if os.path.isfile(p) and not os.access(p, os.X_OK):
        os.chmod(p, 0o755)

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

print(f"[DEBUG] Triton: {triton.__version__}", file=sys.stderr)
print(f"[DEBUG] uTLX loaded: {libutlx_path}", file=sys.stderr)


@triton.jit
def copy_kernel(src_ptr, dst_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    buf = tlx.local_alloc((N,), tlx.dtype_of(src_ptr), 1)
    view = tlx.local_view(buf, 0)
    token = tlx.async_load(src_ptr + offs, view, mask=offs < N)
    tlx.async_load_commit_group([token])
    tlx.async_load_wait_group(0)
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs), mask=offs < N)


def custom_kernel(data):
    N = 128
    src = torch.randn(N, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(src)
    copy_kernel[(1,)](src, dst, N=N)
    assert torch.allclose(src, dst), "copy kernel failed"
    print("--- uTLX copy kernel PASS ---", file=sys.stderr)
    return data
