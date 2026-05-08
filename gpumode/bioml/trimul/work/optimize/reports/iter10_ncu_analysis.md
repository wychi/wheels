# NCU memory-movement analysis — iter10 best kernel (shape 6)

**Setup.** H100 SM90, single iter of `custom_kernel(shape 6: B=1, S=1024, D=384, H=128)`
profiled with `ncu --set full` after 5 warmup iters. Bench wall (10-iter median) = 5.10 ms.
Profile: `work/optimize/profile/iter10_e2e_shape6_full.ncu-rep`.

## Per-kernel summary

| Kernel | Dur (us) | %wall | DRAM% | DRAM (TB/s) | L2-hit% | Compute% | Regs | SMEM (KB) | Occ% | Class |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `matmul_kernel_tlx_ws` (TLX 5-proj GEMM) | 1200 | 23.5 | 72 | 1.77 | 80 | **65** | 154 | 213 | 19 | balanced (compute-leaning) |
| `ln_stats_and_bf16_cast` (Triton) | 1090 | 21.4 | **91** | 2.22 | 36 | 17 | 44 | 0 | 49 | bandwidth-bound (saturated) |
| `fused_gate_ln` (Triton) | 948 | 18.6 | **92** | 2.26 | 38 | 54 | 42 | 0.5 | 60 | bandwidth-bound (saturated) |
| `nvjet_192x192_64x4..._TNN` (cuBLAS bf16 einsum bmm) | 535 | 10.5 | 80 | 1.96 | 81 | 29 | 168 | 213 | 15 | bandwidth-bound |
| `nvjet_256x128_64x4..._TNT` (cuBLAS bf16 H→D linear) | 503 | 9.9 | 64 | 1.58 | 65 | **82** | 168 | 213 | 15 | compute-bound |
| `tr_fwd_pair` (Triton) | 495 | 9.7 | **87** | 2.12 | 51 | 10 | 186 | 33 | 12 | bandwidth-bound (saturated) |
| `fused_invtr_ln_gate` (Triton, D=384 path) | 357 | 7.0 | **91** | 2.21 | 34 | 27 | 103 | 33 | 24 | bandwidth-bound (saturated) |
| `nvjet_64x8_64x16` (cuBLAS splitK reduce) | 5 | 0.1 | 1 | — | 92 | 2 | 168 | 164 | 15 | tiny |

**Sum = 5134 us ≈ 101 % of 5100 us wall** — every microsecond accounted.

## Memory-movement breakdown (shape 6, T = B·S² = 1 048 576)

| Stage | Reads | Writes | Total bytes | Theoretical @ 2.5 TB/s | Measured | BW eff |
|---|---|---|---:|---:|---:|---:|
| `ln_stats_and_bf16_cast` | `x` fp32 [T,D]=**1.61 GB** | mean+rstd 8 MB + `x` bf16 [T,D]=**0.81 GB** | 2.42 GB | 0.97 ms | 1.09 ms | 89 % |
| `matmul_kernel_tlx_ws` (compute-leaning) | `x` bf16 0.81 GB + `B_g` 0.49 MB | `proj` bf16 [T,5H]=**1.34 GB** | 2.15 GB | (~0.86 ms BW; 0.78 ms TC at 65 % peak) | 1.20 ms | TC-limited |
| `fused_gate_ln` | `proj` 1.34 GB + tables + `mask` 4 MB | 3 × bf16 [T,hd] = **0.81 GB** | 2.16 GB | 0.86 ms | 0.95 ms | 91 % |
| `tr_fwd_pair` | `lf+rf` 0.54 GB | `L+R` [B·hd,N²] 0.54 GB | 1.07 GB | 0.43 ms | 0.50 ms | 86 % |
| cuBLAS bf16 einsum bmm | `L+R` 0.54 GB | `out_bmm` 0.27 GB | 0.81 GB | 0.32 ms | 0.54 ms | 60 % (small grid 132 CTAs) |
| `fused_invtr_ln_gate` | `bmm` 0.27 + `og` 0.27 + tables | `gated` 0.27 GB | 0.81 GB | 0.32 ms | 0.36 ms | 90 % |
| cuBLAS bf16 final linear | `gated` 0.27 + `W_out` 0.19 MB | `out` bf16 [T,D]=0.81 GB | 1.07 GB | 0.43 ms | 0.50 ms | compute-limited (82 %) |
| **Pipeline total** | — | — | **~10.5 GB/iter** | **4.2 ms ideal** | **5.10 ms** | **82 %** |

The five bandwidth-bound Triton kernels all sit at **86–92 % of the 2.5 TB/s
achievable HBM3 ceiling** — they are essentially saturated. We are not leaving
bandwidth on the table for these stages individually.

## Where the data is being moved

```
[fp32 input x: 1.6 GB read]
    ↓ ln_stats_and_bf16_cast    (writes bf16 x: 0.8 GB)
[bf16 x: 0.8 GB]
    ↓ matmul_kernel_tlx_ws      (reads bf16 x + B_g, writes proj 1.34 GB)
[bf16 proj [T, 5H]: 1.34 GB]    ← 5-projection output, 5 fans of bf16
    ↓ fused_gate_ln             (writes 3 × bf16 [T, hd] = 0.81 GB total)
[bf16 lf, rf, og: 3 × 0.27 GB]
    ↓ tr_fwd_pair               (transpose lf,rf → bmm-friendly layout)
[bf16 L, R: 2 × 0.27 GB]
    ↓ cuBLAS bf16 bmm           (einsum, fp32 accum, bf16 out: 0.27 GB)
[bf16 out_bmm: 0.27 GB]
    ↓ fused_invtr_ln_gate       (LN+gate, writes gated bf16 0.27 GB)
[bf16 gated: 0.27 GB]
    ↓ cuBLAS bf16 H→D linear    (writes out bf16: 0.81 GB)
[bf16 output: 0.81 GB]
```

**Total intermediate writes ≈ 4.4 GB/iter** plus the 1.6 GB input read and the
0.81 GB output write. Each intermediate is read back once → ~10.5 GB total HBM traffic.

## What the picture tells us

- **The Triton kernels are at the HBM ceiling.** `fused_gate_ln` is the cleanest
  example: 92 % DRAM, 2.26 TB/s. There is no per-kernel tuning left on these
  five Triton stages; further wins must reduce **total HBM traffic** via
  fusion, not improve individual kernel efficiency.
- **The TLX matmul has 35 % TC headroom.** 65 % Compute (≈ 640 TF/s of bf16,
  vs 989 peak) suggests there is real room — typical Hopper bf16 GEMMs hit
  75–85 %. Profiling the WGMMA pipeline (stall-barrier vs stall-long-scoreboard)
  would pinpoint whether to add stages, change tile shape, or fix the producer
  drain pattern. Achievable saving: ~0.2–0.3 ms (4–6 % e2e).
- **The cuBLAS einsum bmm is 80 % DRAM-bound.** Only 132 output CTAs (B·H = 128
  bmm batches × small N=1024 grid). The bmm kernel can only saturate ~80 % of
  HBM here; reducing K-axis traffic (e.g., via tiling that keeps R rows in L2)
  would help, but cuBLAS's selection is opaque.
- **The cuBLAS H→D linear is 82 % compute-bound** — already efficient; no
  obvious win from replacing it.

## Top-3 remaining opportunities ranked by ROI

| # | Lever | Savings (est) | Difficulty | Risk |
|---|---|---|---|---|
| 1 | Fuse `fused_gate_ln` epilogue into TLX `matmul_kernel_tlx_ws` (write bf16 lf/rf/og directly from the matmul, skip the 1.34 GB `proj` intermediate) | **~0.7 ms (14 %)** | High (TLX warp-spec epilogue + per-projection LN math) | Med (numerics OK; complex kernel) |
| 2 | Improve TLX matmul TC throughput 65 → 80 % via tile/stage tuning | 0.2–0.3 ms (4–6 %) | Med (sweep BM/BK/NUM_STAGES with proper SMEM math) | Low |
| 3 | Skip `tr_fwd_pair` by writing `L,R` directly in [B·hd, N²] layout from `fused_gate_ln` (requires 2D-tiled writer with SMEM transpose, the failed iter7a done correctly) | 0.4–0.5 ms (8–10 %) | High (coalescing-sensitive) | Low (purely layout) |

Item #1 is the single largest remaining bandwidth lever — the 1.34 GB `proj`
write+read is the second-biggest tensor in the pipeline after the fp32 input.
Items #2 and #3 are independent and could be combined.
