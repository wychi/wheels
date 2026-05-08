# Optimization Progress

Source kernel: `work/hopper_gemm_ws.py`. Target shape: #6 (B=1, S=1024, D=384). All times = `--bench` median, 10 iters, warmup=3, on H100 (CUDA_VISIBLE_DEVICES=4).

## Benchmark Table (shape 6 unless noted)

| Iter | Variant | shape6 ms | speedup | Notes |
|------|---------|-----------|---------|-------|
| 0 | baseline (`work/hopper_gemm_ws.py`) | 9.66 | 1.00× | profile in `profile/e2e_results.md` |
| 1 | O3: skip bf16 downcast on out_bmm | 9.45 | 1.022× | accuracy passes all 7; fused_invtr_ln_gate now reads fp32 |
| 2 | O2: thin-K matmul cfg for D=128 — **REVERTED** | 9.39 | (no-op) | matmul on shape 4 went 922→928 us (slightly worse); D=128 shapes already reach steady state |
| 3 | O4: fused tr+cast (`tr_cast_fwd`) for bmm operands | 8.78 | **1.10×** | 6-10% across all shapes; replaces 2× tr_fwd + 2× elementwise upcast with one kernel; bf16 [B*hd,N²] intermediate eliminated |
| 4 | O6 (partial): bf16 final linear (cuBLAS bf16 GEMM, fp32 accum) | 6.69 | **1.44×** | 14-24% per shape; cumulative 30.7% from baseline. Skip `gated.float()` upcast and `out.to(bf16)` downcast; cuBLAS uses fp32 accum internally |
| 5 | O1: bf16 einsum bmm (cuBLAS bf16 GEMM, fp32 accum) | 5.82 | **1.66×** | 13-20% per shape; cumulative 39.7% from baseline. Skip the fp32 cast on L,R; tr_cast_fwd → tr_fwd_pair (no upcast). bmm output now bf16 (LN inside fused_invtr_ln_gate reloads as fp32). |
| 6 | O10: fold fp32→bf16 cast of x into ln_stats (`ln_stats_and_bf16_cast`) | 5.12 | **1.89×** | 6-13% per shape; cumulative 47.0% from baseline. Saves second read of [T,D] fp32. |
| 7a | strided fused_gate_ln write (skip tr_fwd_pair) — **REVERTED** | 8.08 | (-58%) | Uncoalesced strided write tanked everything. |
| 7b | replace TLX matmul with cuBLAS bf16 GEMM | 5.12 | 1.89× | -3.6% on shape 0, 0% on shape 6; cuBLAS ≈ TLX WS for the 5-proj GEMM. |

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

### iter4 — O6 (partial): bf16 final linear (skip the fp32 ping-pong)
- Replace `F.linear(gated.float(), W["to_out.weight"].float())` + final `.to(bf16)` with `F.linear(gated, W_out.to(bf16))`.
- cuBLAS bf16 GEMM uses fp32 accumulator internally. K=hd=128 is small enough that bf16 input mantissa loss stays inside atol=2e-2.
- Per-shape speedup: 0=13.8%, 1=17.4%, 2=23.3%, 3=17.8%, 4=17.6%, 5=23.6%, 6=23.8% (**huge — 19% geo-mean**).
- Cumulative: shape 6 9.66 → 6.69 ms = **1.44× faster than baseline**.
- Profile delta on shape 6: fp32 cuBLAS H→D (sm90_xmma 64x256) at 0.95 ms — gone. bf16 cuBLAS replacement is in noise. `vectorized_elementwise` halved (was 2.14 → now 1.07 ms — the `gated.float()` and `out_fp32.to(bf16)` casts are gone).
- Accuracy: max err on shape 0 0.0224→0.0254, shape 3 0.0201→0.0226 — still passes (`atol+rtol*|ref|` gate). All other shapes hold.

### iter7 — strided fused_gate_ln write (REVERTED) + cuBLAS 5-proj matmul
- **7a (reverted):** modify fused_gate_ln to write `lf, rf` directly in [B, hd, N²] (bmm-friendly) layout to skip the tr_fwd_pair pass. Within each program (1 thread block per row of length hd), the strided writes (offsets spaced N²=1M apart) destroyed coalescing and made every shape 50-60% slower. A correct fix would require 2D tiling (TI rows × hd cols per program), which is more invasive.
- **7b:** replace `tlx_ws_matmul_fixed` with `torch.matmul(x_flat, B_g)` (cuBLAS bf16 GEMM, fp32 accum). Modest improvement on small shapes (shape 0 0.524→0.505, ≈3.6%), neutral on shape 6. Worth keeping for code simplicity — the 5-proj GEMM no longer relies on the bespoke TLX warp-spec kernel for any shape.

### iter6 — O10 (partial): fold fp32→bf16 cast of x into ln_stats
- New Triton kernel `ln_stats_and_bf16_cast` reads fp32 x once, writes mean+rstd AND bf16-cast x. Replaces `ln_stats_multirow` + `x_flat.to(bf16)` elementwise pair.
- Saves the second pass over [T, D] fp32 (~1.6 GB on shape 6) — partly offset by the new bf16 write (~0.8 GB).
- Per-shape speedup: 0=5.4%, 1=6.3%, 2=12.6%, 3=6.2%, 4=6.1%, 5=12.1%, 6=12.1%.
- Cumulative shape 6: 5.82 → 5.12 ms = **1.89× faster than baseline**.
- Note: shape 0 first-iter shows 2.5 ms (JIT compile of new kernel costs); steady-state matches the warm number above.

### iter5 — O1: bf16 einsum bmm (Codex's "no precision restoration" insight)
- Drop the bf16→fp32 upcast in `tr_cast_fwd` → renamed `tr_fwd_pair` (just transpose). Run `torch.bmm` on bf16 inputs; cuBLAS uses fp32 accumulator internally. Output is bf16 (cuBLAS lacks bf16-in/fp32-out via `torch.bmm`).
- The bf16→fp32 upcast we'd been doing didn't restore mantissa bits the bf16 storage already lost; per Codex's analysis, the multiply input precision is identical to bf16 input + fp32 accum.
- Per-shape speedup: 0=19.9%, 1=17.9%, 2=12.6%, 3=16.1%, 4=17.7%, 5=13.5%, 6=13.0%.
- Cumulative shape 6: 6.69 → 5.82 ms = **1.66× faster than baseline**.
- Profile delta on shape 6: tr_cast_fwd 0.79 ms → tr_fwd_pair 0.51 ms (-0.28); fp32 cuBLAS einsum 1.10 ms → bf16 cuBLAS einsum is no longer in the top 9 (sub-0.5 ms); fused_invtr_ln_gate 0.47 → 0.36 ms (reads bf16 instead of fp32).
- Accuracy: cauchy shape 1 max_err 0.0192→0.0217, shape 4 0.0187→0.0205 — still passes the `atol+rtol*|ref|` gate. Tighter than feared.





