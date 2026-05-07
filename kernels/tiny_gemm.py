#!/usr/bin/env python3
"""uTLX tiny GEMM using local_alloc + async_load + async_dot."""

import sys
import torch
import triton
import triton.language as tl
import utlx_plugin as tlx


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


def test_tiny_gemm():
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
    print(f"rel_err={rel_err:.6f}")
    assert rel_err < 0.01, f"FAILED: rel_err={rel_err}"
    print("PASS")


if __name__ == "__main__":
    test_tiny_gemm()
