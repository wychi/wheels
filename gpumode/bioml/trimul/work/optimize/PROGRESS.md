# Optimization Progress

Source kernel: `work/hopper_gemm_ws.py`. Target shape: #6 (B=1, S=1024, D=384). All times = `--bench` median, 10 iters, warmup=3, on H100 (CUDA_VISIBLE_DEVICES=4).

## Benchmark Table (shape 6 unless noted)

| Iter | Variant | shape6 ms | speedup | Notes |
|------|---------|-----------|---------|-------|
| 0 | baseline (`work/hopper_gemm_ws.py`) | 9.66 | 1.00× | profile in `profile/e2e_results.md` |
| 1 | O3: skip bf16 downcast on out_bmm | 9.45 | 1.022× | accuracy passes all 7; fused_invtr_ln_gate now reads fp32 |
| 2 | O2: thin-K matmul cfg for D=128 — **REVERTED** | 9.39 | (no-op) | matmul on shape 4 went 922→928 us (slightly worse); D=128 shapes already reach steady state |
| 3 | O4: fused tr+cast (`tr_cast_fwd`) for bmm operands | 8.78 | **1.10×** | 6-10% across all shapes; replaces 2× tr_fwd + 2× elementwise upcast with one kernel; bf16 [B*hd,N²] intermediate eliminated |

## Per-iteration log

### iter1 — O3: drop `out_bmm.to(torch.bfloat16)` (commit 1)
- One-line removal at custom_kernel; `fused_invtr_ln_gate` reads fp32 directly.
- All 7 shapes pass atol=2e-2 (max 0.0224 on shape 0, well within budget).
- Per-kernel torch.profiler delta (sum ms / 20 iters):
  - `vectorized_elementwise`: 50.07 → 42.90 ms (−7.17, the cast removal)
  - `unrolled_elementwise`: 31.28 → 26.82 ms (−4.46, related downcasts also gone)
  - `fused_invtr_ln_gate`: 7.36 → 9.43 ms (+2.07, now reads fp32)
- **Council (Codex/GPT-5.2):** "next 4 iters → O2 thin-K, O4 drop tr_fwd, O1 bf16 einsum (viable: bf16→fp32 cast pre-bmm doesn't restore mantissa, so a Triton bf16-input + fp32-accum kernel is numerically equivalent), then O5". Pitfall: don't optimize shape 6 in isolation — track geomean across all 7 shapes. (`reports/iter1_consult_codex.md`)

### iter2 — O2: thin-K config (BK=128, NUM_STAGES=1) for D=128 — REVERTED
- Profile-targeted shape 4 (D=128, S=1024) matmul: FAT_K = **922 µs**, THIN_K = **928 µs** (essentially equal).
- D=128 shapes have enough M-tiles per SM (4096 tiles / 132 SMs ≈ 31 per SM) that the WS pipeline reaches steady state even with only 2 K-iters. The doc's "thin-K" diagnosis was for the smallest shape (shape 0, 19 tiles per SM), where the matmul is already <0.1 ms — too small to matter at e2e.
- **Lesson:** the planning doc's tile-config recommendation extrapolated from a single profile point; in practice, no measurable gain at any shape that costs >1 ms. Skip O2 for the rest of this wave.

### iter3 — O4: fused transpose + bf16→fp32 upcast (`tr_cast_fwd`)
- New Triton kernel: reads bf16 lf+rf in their native [T, hd] layout, writes fp32 Lf+Rf in bmm-ready [B*hd, N²]. Single launch handles both operands.
- Eliminates the [B*hd, N²] **bf16 intermediate** (≈0.54 GB write+read for shape 6) and one launch.
- Per-shape speedup: 0=10.0%, 1=8.2%, 2=6.8%, 3=9.5%, 4=9.0%, 5=5.8%, 6=6.5% (geo-mean ≈ 7.9%).
- Profile delta on shape 6: tr_fwd ×2 (0.51 ms) + L,R fp32 cast (~2.14 ms hidden inside vectorized_elementwise) → **tr_cast_fwd 0.79 ms**. Some PyTorch elementwise calls remain (input upcast, mask cast, etc.).
- Accuracy: max err on shape 1 nudged 0.0181→0.0183 (still well under 2e-2). Other shapes unchanged.



