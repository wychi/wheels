#!/usr/bin/env python3
"""Hopper warp-specialized GEMM with async descriptor loads and double buffering."""

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx


@triton.jit
def hopper_ws(a_ptr, b_ptr, c_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    NS: tl.constexpr, NMG: tl.constexpr, NSMS: tl.constexpr):
    BMS: tl.constexpr = BM // NMG

    # Create TMA descriptors (compiler auto-allocates global scratch via the
    # host-side triton.set_allocator callback). The desc_ptr-based path in
    # tlx.make_tensor_descriptor needs a 7-arg create_make_tensor_descriptor
    # binding that the upstream wheel doesn't expose, so we use the auto-alloc
    # form which maps to the single binding in triton's ir.cc.
    a_desc = tlx.make_tensor_descriptor(
        desc_ptr=None, base=a_ptr,
        shape=[M, K], strides=[K, 1], block_shape=[BMS, BK],
    )
    b_desc = tlx.make_tensor_descriptor(
        desc_ptr=None, base=b_ptr,
        shape=[K, N], strides=[N, 1], block_shape=[BK, BN],
    )
    c_desc = tlx.make_tensor_descriptor(
        desc_ptr=None, base=c_ptr,
        shape=[M, N], strides=[N, 1], block_shape=[BM // NMG, BN],
    )

    a = tlx.local_alloc((BMS, BK), tlx.dtype_of(a_ptr), NS * NMG)
    b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_ptr), NS)
    bea = tlx.alloc_barriers(num_barriers=NS * NMG, arrive_count=1)
    beb = tlx.alloc_barriers(num_barriers=NS, arrive_count=NMG)
    bfa = tlx.alloc_barriers(num_barriers=NS * NMG, arrive_count=1)
    bfb = tlx.alloc_barriers(num_barriers=NS, arrive_count=1)
    with tlx.async_tasks():
        with tlx.async_task("default"):  # producer
            sm = tl.program_id(0)
            tid = sm
            while tid < tl.cdiv(M, BM) * tl.cdiv(N, BN):
                pid_m = tid // tl.cdiv(N, BN)
                pid_n = tid % tl.cdiv(N, BN)
                for k in range(tl.cdiv(K, BK)):
                    buf = k % NS
                    p = (k // NS) & 1
                    ok = k * BK
                    # Load A half 1
                    tlx.barrier_wait(bar=tlx.local_view(bea, buf), phase=p ^ 1)
                    tlx.barrier_expect_bytes(tlx.local_view(bfa, buf), BMS * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                    tlx.async_descriptor_load(a_desc, tlx.local_view(a, buf), [pid_m * BM, ok], tlx.local_view(bfa, buf))
                    # Load B
                    tlx.barrier_wait(bar=tlx.local_view(beb, buf), phase=p ^ 1)
                    tlx.barrier_expect_bytes(tlx.local_view(bfb, buf), BN * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                    tlx.async_descriptor_load(b_desc, tlx.local_view(b, buf), [ok, pid_n * BN], tlx.local_view(bfb, buf))
                    # Load A half 2
                    tlx.barrier_wait(bar=tlx.local_view(bea, buf + NS), phase=p ^ 1)
                    tlx.barrier_expect_bytes(bar=tlx.local_view(bfa, buf + NS), size=BMS * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                    tlx.async_descriptor_load(a_desc, tlx.local_view(a, buf + NS), [pid_m * BM + BMS, ok], tlx.local_view(bfa, buf + NS))
                tid += NSMS

        with tlx.async_task(num_warps=4, replicate=2):  # consumers
            sm = tl.program_id(0)
            tid = sm
            while tid < tl.cdiv(M, BM) * tl.cdiv(N, BN):
                pid_m = tid // tl.cdiv(N, BN)
                pid_n = tid % tl.cdiv(N, BN)
                acc = tl.zeros([BM // 2, BN], dtype=tl.float32)
                for k in range(tl.cdiv(K, BK)):
                    buf = k % NS
                    p = (k // NS) & 1
                    tlx.barrier_wait(bar=tlx.local_view(bfa, buf + NS * tlx.async_task_replica_id()), phase=p)
                    tlx.barrier_wait(bar=tlx.local_view(bfb, buf), phase=p)
                    acc = tlx.async_dot(tlx.local_view(a, buf + NS * tlx.async_task_replica_id()), tlx.local_view(b, buf), acc)
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)
                    tlx.barrier_arrive(tlx.local_view(bea, buf + NS * tlx.async_task_replica_id()))
                    tlx.barrier_arrive(tlx.local_view(beb, buf))
                c_desc.store([pid_m * BM + BMS * tlx.async_task_replica_id(), pid_n * BN], acc.to(tlx.dtype_of(c_desc)))
                tid += NSMS


def launch_hopper_ws(a, b, *, BM=128, BN=256, BK=64, NS=2, NMG=2, num_sms=None):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(a.device).multi_processor_count

    # Backing pool for the TMA descriptors that the kernel allocates via
    # tlx.make_tensor_descriptor(desc_ptr=None, ...).
    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )

    hopper_ws[(num_sms,)](
        a, b, c, M, N, K,
        BM=BM, BN=BN, BK=BK, NS=NS, NMG=NMG, NSMS=num_sms,
    )
    return c


def test_hopper_ws():
    M, N, K = 512, 512, 256
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    c = launch_hopper_ws(a, b)

    ref = torch.matmul(a.float(), b.float()).half()
    rel_err = (c - ref).abs().max().item() / ref.abs().max().item()
    print(f"rel_err={rel_err:.6f}")
    assert rel_err < 0.01, f"FAILED: rel_err={rel_err}"
    print("PASS")


if __name__ == "__main__":
    test_hopper_ws()
