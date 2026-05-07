#!/usr/bin/env python3
"""
uTLX submission — verifies patched Triton wheel with uTLX extension.

Usage:
    python submission.py                # eval mode: pip install + run
    python submission.py --no-install   # local test: skip pip install
"""

import builtins
import os
import subprocess
import sys
import sysconfig
from typing import Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _install_custom_deps():
    if "--no-install" in sys.argv:
        return

    TRITON_WHEEL_URL = "https://github.com/wychi/wheels/raw/refs/heads/main/gpumode/triton-3.7.0+gitb7fa781f-cp313-cp313-linux_x86_64.whl"
    UTLX_WHEEL_URL = "https://github.com/plotfi/plotfi-wheels/raw/main/utlx-0.1.0-py3-none-any.whl"

    print(f"[DEBUG] Python: {sys.version}", file=sys.stderr)
    print(f"[DEBUG] Installing triton from: {TRITON_WHEEL_URL}", file=sys.stderr)
    result = subprocess.run(["uv", "pip", "install", "--force-reinstall",
                             f"triton @ {TRITON_WHEEL_URL}",
                             f"utlx @ {UTLX_WHEEL_URL}"],
                            capture_output=True, text=True)
    print(f"[DEBUG] pip exit code: {result.returncode}", file=sys.stderr)
    print(f"[DEBUG] pip stdout: {result.stdout[-500:]}", file=sys.stderr)
    if result.returncode != 0:
        print(f"[DEBUG] pip stderr: {result.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)


def _setup_utlx():
    dist_packages = sysconfig.get_paths()["purelib"]
    libutlx_path = os.path.join(dist_packages, "utlx_plugin", "libutlx.so")
    assert os.path.isfile(libutlx_path), f"libutlx.so not found at {libutlx_path}"
    os.environ["TRITON_PLUGIN_PATHS"] = libutlx_path

    import triton
    print(f"[DEBUG] Triton: {triton.__version__}", file=sys.stderr)
    print(f"[DEBUG] uTLX loaded: {libutlx_path}", file=sys.stderr)


_install_custom_deps()
_setup_utlx()

import torch
import triton
import triton.language as tl
import triton.language.semantic as triton_semantic
from triton import knobs
import utlx_plugin as tlx


# ---------------------------------------------------------------------------
# Monkey-patches — remove when triton natively supports uTLX
# ---------------------------------------------------------------------------

def apply_tlx_patches():

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
        if not (lhs.dtype.is_fp8() and rhs.dtype.is_fp8()):
            assert lhs.dtype == rhs.dtype
        if input_precision is None:
            input_precision = self.builder.options.default_dot_input_precision
        input_precision = self._str_to_dot_input_precision(input_precision)
        lhs_rank = len(lhs.shape)
        assert lhs_rank == len(rhs.shape)
        assert tl._unwrap_if_constexpr(lhs.shape[-1]) == tl._unwrap_if_constexpr(rhs.shape[-2])
        min_dot_size = self.builder.codegen_fns["min_dot_size"](lhs.type, rhs.type)
        assert (tl._unwrap_if_constexpr(lhs.shape[-2]) >= min_dot_size[0]
                and tl._unwrap_if_constexpr(lhs.shape[-1]) >= min_dot_size[2]
                and tl._unwrap_if_constexpr(rhs.shape[-1]) >= min_dot_size[1])
        if lhs.type.scalar.is_int():
            _0 = self.builder.get_int32(0)
            ret_scalar_ty = tl.int32
        elif lhs.type.scalar.is_fp32() or lhs.type.scalar.is_bf16():
            _0 = self.builder.get_fp32(0)
            ret_scalar_ty = tl.float32
        else:
            _0 = self.builder.get_fp16(0) if out_dtype.is_fp16() else self.builder.get_fp32(0)
            ret_scalar_ty = out_dtype
        M = lhs.type.shape[-2]
        N = rhs.type.shape[-1]
        K = lhs.type.shape[-1]
        B = lhs.type.shape[0] if lhs_rank == 3 else None
        ret_ty = tl.block_type(ret_scalar_ty, [B, M, N] if B else [M, N])
        if acc is None:
            acc_handle = self.builder.create_splat(ret_ty.to_ir(self.builder), _0)
        else:
            acc_handle = acc.handle
        if max_num_imprecise_acc is None:
            max_num_imprecise_acc = self.builder.options.max_num_imprecise_acc_default if (lhs.dtype.is_fp8() and rhs.dtype.is_fp8()) else 0
        return (lhs, rhs, acc_handle, input_precision, max_num_imprecise_acc, ret_ty)

    setattr(triton.language, "_unwrap_if_constexpr", _unwrap_if_constexpr)
    setattr(triton_semantic.TritonSemantic, "_prepare_legacy_load", _prepare_legacy_load)
    setattr(triton_semantic.TritonSemantic, "dot_precheck", dot_precheck)


apply_tlx_patches()

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@triton.jit
def tiny_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    buf_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 1)
    buf_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), 1)
    a = tlx.local_view(buf_a, 0)
    b = tlx.local_view(buf_b, 0)
    ta = tlx.async_load(a_ptrs, a, mask=offs_k[None, :] < BLOCK_K)
    tb = tlx.async_load(b_ptrs, b, mask=offs_k[:, None] < BLOCK_K)
    tlx.async_load_commit_group([ta, tb])
    tlx.async_load_wait_group(0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tlx.async_dot(a, b, acc)
    acc = tlx.async_dot_wait(0, acc)

    c = acc.to(tlx.dtype_of(c_ptr))
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, c)

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def custom_kernel(data):
    M, N, K = 128, 256, 64
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    tiny_gemm_kernel[(1,)](
        a, b, c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=M, BLOCK_N=N, BLOCK_K=K, num_warps=8,
    )

    ref = torch.matmul(a.float(), b.float()).half()
    rel_err = (c - ref).abs().max().item() / ref.abs().max().item()
    print(f"[DEBUG] uTLX tiny GEMM rel_err={rel_err:.6f}", file=sys.stderr)
    assert rel_err < 0.01, f"uTLX GEMM failed: rel_err={rel_err}"
    print("--- uTLX PASS ---", file=sys.stderr)
    return data


if __name__ == "__main__":
    custom_kernel({})
