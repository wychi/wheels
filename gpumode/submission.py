#!/usr/bin/env python3
"""
Test script to verify the patched Triton wheel works with uTLX.
Runs a small pipelined Hopper GEMM via uTLX async ops.

Usage:
    python submission.py
"""

import subprocess
import sys

# TODO: update URL after uploading wheel to GitHub Releases
TRITON_WHEEL_URL = "https://github.com/wychi/wheels/releases/download/v0.1.0/triton-3.7.0+gitb7fa781f-cp313-cp313-linux_x86_64.whl"
UTLX_WHEEL_URL = "https://github.com/plotfi/plotfi-wheels/raw/main/utlx-0.1.0-py3-none-any.whl"

subprocess.check_call([sys.executable, "-m", "pip", "install",
                       f"triton @ {TRITON_WHEEL_URL}",
                       f"utlx @ {UTLX_WHEEL_URL}"])

import os
import sysconfig
from typing import Optional, Tuple, Any
import builtins

dist_packages = sysconfig.get_paths()["purelib"]
libutlx_path = os.path.join(dist_packages, "utlx_plugin", "libutlx.so")
assert os.path.isfile(libutlx_path), f"libutlx.so not found at {libutlx_path}"
os.environ["TRITON_PLUGIN_PATHS"] = libutlx_path
os.environ["MLIR_ENABLE_DUMP"] = "1"
os.environ["TRITON_ALWAYS_COMPILE"] = "1"

for binary in ("ptxas", "ptxas-blackwell"):
    p = os.path.join(dist_packages, "triton", "backends", "nvidia", "bin", binary)
    if os.path.isfile(p) and not os.access(p, os.X_OK):
        os.chmod(p, 0o755)

import torch
import triton
import triton.language as tl
import triton.language.semantic as triton_semantic
from triton import knobs
import utlx_plugin as tlx

# ---------------------------------------------------------------------------
# Monkey-patches required by the custom TLX plugin
# ---------------------------------------------------------------------------

def dot_precheck(
    self, lhs: tl.tensor, rhs: tl.tensor, acc: tl.tensor,
    input_precision: Optional[str], allow_tf32, max_num_imprecise_acc: int,
    out_dtype: tl.dtype, tlx_paired_ctas: bool = False,
) -> Tuple[Any]:
    input_precision = tl._unwrap_if_constexpr(input_precision)
    allow_tf32 = tl._unwrap_if_constexpr(allow_tf32)
    assert input_precision is None or tl._unwrap_if_constexpr(allow_tf32) is None
    if input_precision is None:
        supports_tf32 = "tf32" in self.builder.options.allowed_dot_input_precisions
        input_precision = knobs.language.fp32_default or ("tf32" if
            (supports_tf32 and (allow_tf32 or allow_tf32 is None)) else "ieee")
    input_precision = tl._unwrap_if_constexpr(input_precision)
    out_dtype = tl._unwrap_if_constexpr(out_dtype)
    max_num_imprecise_acc = tl._unwrap_if_constexpr(max_num_imprecise_acc)
    acc = tl._unwrap_if_constexpr(acc)
    assert lhs.type.is_block() and rhs.type.is_block()
    if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
        pass
    else:
        assert lhs.dtype == rhs.dtype, f"Both operands must be same dtype. Got {lhs.dtype} and {rhs.dtype}"
    if input_precision is None:
        input_precision = self.builder.options.default_dot_input_precision
    input_precision = self._str_to_dot_input_precision(input_precision)
    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)
    assert lhs_rank == rhs_rank == 2 or lhs_rank == rhs_rank == 3
    assert tl._unwrap_if_constexpr(lhs.shape[-1]) == tl._unwrap_if_constexpr(rhs.shape[-2])
    min_dot_size = self.builder.codegen_fns["min_dot_size"](lhs.type, rhs.type)
    assert (tl._unwrap_if_constexpr(lhs.shape[-2]) >= min_dot_size[0]
            and tl._unwrap_if_constexpr(lhs.shape[-1]) >= min_dot_size[2]
            and tl._unwrap_if_constexpr(rhs.shape[-1]) >= min_dot_size[1])
    if lhs.type.scalar.is_int():
        _0 = self.builder.get_int32(0)
        ret_scalar_ty = tl.int32
    elif out_dtype.is_bf16():
        raise ValueError("out_dtype=bfloat16 is unsupported")
    elif lhs.type.scalar.is_fp32() or lhs.type.scalar.is_bf16():
        _0 = self.builder.get_fp32(0)
        ret_scalar_ty = tl.float32
    elif lhs.type.scalar.is_fp64():
        _0 = self.builder.get_fp64(0)
        ret_scalar_ty = tl.float64
    else:
        _0 = self.builder.get_fp16(0) if out_dtype.is_fp16() else self.builder.get_fp32(0)
        ret_scalar_ty = out_dtype
    M = lhs.type.shape[-2]
    if tlx_paired_ctas:
        N = 2 * rhs.type.shape[-1]
    else:
        N = rhs.type.shape[-1]
    K = lhs.type.shape[-1]
    B = lhs.type.shape[0] if lhs_rank == 3 else None
    ret_ty = tl.block_type(ret_scalar_ty, [B, M, N] if B else [M, N])
    if acc is None:
        acc_handle = self.builder.create_splat(ret_ty.to_ir(self.builder), _0)
    else:
        acc_handle = acc.handle
        assert acc.type.shape == ret_ty.shape and acc.type.element_ty == out_dtype
    if max_num_imprecise_acc is None:
        if lhs.dtype.is_fp8() and rhs.dtype.is_fp8():
            max_num_imprecise_acc = self.builder.options.max_num_imprecise_acc_default
        else:
            max_num_imprecise_acc = 0
    else:
        if lhs.dtype.is_fp8() and rhs.dtype.is_fp8() and max_num_imprecise_acc > K:
            raise ValueError(f"max_num_imprecise_acc ({max_num_imprecise_acc}) must be <= K ({K})")
    return (lhs, rhs, acc_handle, input_precision, max_num_imprecise_acc, ret_ty)


def _prepare_legacy_load(self, ptr, mask, other, boundary_check, padding):
    if not ptr.type.scalar.is_ptr():
        raise ValueError(f"Unsupported ptr type {ptr.type.__repr__()} in `tl.load`")
    if mask is None and other is not None:
        raise ValueError("`other` cannot be provided without `mask`")
    if padding or boundary_check:
        raise ValueError("`padding_option` or `boundary_check` argument is not supported for legacy loads")
    if not ptr.type.is_block():
        if mask and mask.type.is_block():
            raise ValueError("Mask argument cannot be block type if pointer argument is not a block")
        if other and other.type.is_block():
            raise ValueError("Other argument cannot be block type if pointer argument is not a block")
    if ptr.type.is_block():
        if mask is not None:
            ptr, mask = self.broadcast_impl_value(ptr, mask)
        if other is not None:
            ptr, other = self.broadcast_impl_value(ptr, other)
    ptr_ty = ptr.type.scalar
    elt_ty = ptr_ty.element_ty
    is_bool = elt_ty == tl.int1
    if is_bool:
        elt_ty = tl.int8
        ptr_ty = tl.pointer_type(elt_ty, ptr_ty.address_space)
        ptr = self.cast(ptr, ptr_ty)
    if other is not None:
        other = self.cast(other, elt_ty)
    if ptr.type.is_block():
        shape = ptr.type.get_block_shapes()
        dst_ty = tl.block_type(elt_ty, shape)
    else:
        dst_ty = elt_ty
    return dst_ty, ptr, mask, other, is_bool


def _unwrap_if_constexpr(o):
    if isinstance(o, list):
        return [_unwrap_if_constexpr(x) for x in o]
    if isinstance(o, builtins.tuple):
        return builtins.tuple(_unwrap_if_constexpr(x) for x in o)
    if isinstance(o, tuple):
        return tuple(_unwrap_if_constexpr(x) for x in o)
    return o.value if isinstance(o, tl.constexpr) else o


setattr(triton.language, "_unwrap_if_constexpr", _unwrap_if_constexpr)
setattr(triton_semantic.TritonSemantic, "_prepare_legacy_load", _prepare_legacy_load)
setattr(triton_semantic.TritonSemantic, "dot_precheck", dot_precheck)

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_STAGES)
    buffers_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_STAGES)

    for i in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        a = tlx.local_view(buffers_A, i)
        b = tlx.local_view(buffers_B, i)
        ta = tlx.async_load(a_ptrs, a, mask=offs_k[None, :] < K - i * BLOCK_K)
        tb = tlx.async_load(b_ptrs, b, mask=offs_k[:, None] < K - i * BLOCK_K)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        tlx.async_load_commit_group([ta, tb])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=0):
        buf = k % NUM_STAGES
        a_k = tlx.local_view(buffers_A, buf)
        b_k = tlx.local_view(buffers_B, buf)
        tlx.async_load_wait_group(NUM_STAGES - 2)
        acc = tlx.async_dot(a_k, b_k, acc)

        i = k + NUM_STAGES - 1
        a_next = tlx.local_view(buffers_A, i % NUM_STAGES)
        b_next = tlx.local_view(buffers_B, i % NUM_STAGES)
        acc = tlx.async_dot_wait(1, acc)
        ta = tlx.async_load(a_ptrs, a_next, mask=offs_k[None, :] < K - i * BLOCK_K)
        tb = tlx.async_load(b_ptrs, b_next, mask=offs_k[:, None] < K - i * BLOCK_K)
        tlx.async_load_commit_group([ta, tb])
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = tlx.async_dot_wait(0, acc)
    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def main():
    M, N, K = 512, 512, 512
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    NUM_STAGES = 3

    print(f"Python:  {sys.version}")
    print(f"Triton:  {triton.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.version.cuda}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print(f"GEMM:    M={M}, N={N}, K={K}")

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_STAGES=NUM_STAGES, num_warps=8,
    )

    ref = torch.matmul(a.float(), b.float()).half()
    max_diff = (c - ref).abs().max().item()
    rel_err = max_diff / ref.abs().max().item()
    print(f"Max diff: {max_diff:.6f}, Relative error: {rel_err:.6f}")
    assert rel_err < 0.01, f"GEMM correctness check failed: rel_err={rel_err}"
    print("--- PASS ---")


if __name__ == "__main__":
    main()
