# TriMul `hopper_gemm_ws.py` — Optimization Plan

Target GPU: H100 (SM90). Run command: `python work/hopper_gemm_ws.py --no-install --bench`
(uses pre-installed wheels from `~/oss/wheels/.venv/`).

## Baseline (2026-05-08)

Re-measured this session, single GPU (CUDA_VISIBLE_DEVICES=4), warmup=3, iters=10.

| # | bs | sl | dim | hd | mask | dist | ms (this run) | doc ms |
|---|----|----|-----|----|------|------|---------------|--------|
| 0 | 2 | 256 | 128 | 128 | no | normal | 7.49* | 1.04 |
| 1 | 1 | 768 | 128 | 128 | no | cauchy | 3.89 | 4.09 |
| 2 | 2 | 256 | 384 | 128 | yes | normal | 1.28 | 1.37 |
| 3 | 1 | 512 | 128 | 128 | no | normal | 1.76 | 1.87 |
| 4 | 1 | 1024| 128 | 128 | no | cauchy | 6.94 | 7.33 |
| 5 | 1 | 768 | 384 | 128 | yes | normal | 5.42 | 5.64 |
| 6 | 1 | 1024| 384 | 128 | no | normal | **9.66** | 10.06 |

*shape 0 is noise from JIT amortization tail across 7 shapes — re-measured alone, it lands ≈ 1.0 ms.
Reference baseline for ranking iterations: **shape 6 = 9.66 ms** and **geo-mean across shapes 1-6 = ~3.91 ms**.

## Bottleneck breakdown — largest shape (#6)

From `profile/e2e_results.md`:
| Kernel | ms | % | Note |
|---|---|---|---|
| vectorized_elementwise (bf16→fp32 of L,R) | 2.50 | 25.0% | Pure HBM (1.07 GB write each) |
| unrolled_elementwise (fp32→bf16 of out_bmm) | 1.56 | 15.6% | 1.07 GB read+write |
| matmul_kernel_tlx_ws (5-proj) | 1.19 | 11.9% | bf16 GEMM, prior NCU baseline |
| sm90_xmma_gemm fp32 (einsum) | 1.11 | 11.0% | fp32 cuBLAS bmm |
| sm90_xmma_gemm fp32 (final linear) | 1.01 | 10.1% | fp32 cuBLAS H→D |
| fused_gate_ln | 1.00 | 10.0% | Triton |
| ln_stats_multirow | 0.73 | 7.3% | Triton |
| tr_fwd ×2 | 0.51 | 5.1% | Triton |
| fused_invtr_ln_gate | 0.37 | 3.7% | Triton |

**62 % of wall time is the precision tax** (fp32 cast + fp32 cuBLAS + bf16 cast back).

## Ranked Opportunities

| # | Idea | Hypothesis | Risk | Expected win |
|---|------|------------|------|--------------|
| O1 | **bf16 einsum with fp32 accum** (replace `Lf, Rf, bmm, out_bmm.to(bf16)` with a Triton bf16 GEMM keeping fp32 accum) | Eliminates the 25 % + 16 % cast tax and the fp32 cuBLAS einsum (11 %); accumulator is fp32 so accuracy should hold | Cauchy shapes (#1, #4) may fail atol=2e-2 — bf16 inputs lose precision before MAC | **35–45 %** on largest |
| O2 | **Persistent / tile-tuned matmul_kernel_tlx_ws "thin-K" path** for D=128 | Per h100.md: `BM=128, BN=128, BK=128, NUM_STAGES=1, NUM_MMA_GROUPS=1, EPILOGUE_SUBTILE=True` could close the 24% Stall Barrier on D=128 shapes (1.19 → ~0.5 ms) | Requires dispatcher; D=384 keeps current cfg | 3–7 % on largest, **15-20 % on D=128 shapes** |
| O3 | **Fuse the bf16 cast of `out_bmm` into `fused_invtr_ln_gate`** | `fused_invtr_ln_gate` currently reads `out_bmm.bf16` — if einsum stays fp32 cuBLAS, do the LN+gate consuming fp32 directly, skip the elementwise downcast (1.56 ms) | None if einsum is fp32; cleanly orthogonal to O1 | **15 %** on largest if O1 not adopted |
| O4 | **Remove explicit transposes (`tr_fwd ×2`)** | Today writes `[B*hd, N²]` for the bmm. Replace with strided bmm input or a fused Triton einsum that reads `[B,S,S,H]` directly | If we move to Triton einsum, transposes vanish entirely | 5 % on largest |
| O5 | **Fuse 5-proj matmul + LN epilogue + sigmoid + gate-mul** | The current chain runs ln_stats → matmul → fused_gate_ln. Folding LN-affine + gate sigmoid + gate-mul into the matmul epilogue saves one matmul-output write (1.07 GB) and one read | Moderate complexity; SMEM for 5-fan-out | 8–12 % |
| O6 | **Final linear (H→D) folded into invtr_ln_gate** | The H→D linear is small but is fp32 cuBLAS (10 %); fuse into the post-norm-gate kernel as a Triton matmul epilogue | Per-shape D | 7 % |
| O7 | **TMA multicast cluster_size=2 for einsum** | Einsum reads `L,R` of size [B*H,N,N]; multicasting `R` across CTA pair saves L2 BW | Cluster setup complexity | 3-5 % |
| O8 | **Reduce matmul out-buffer alloc churn** | `tlx_ws_matmul_fixed` reallocates `c` each call; pin to a workspace buffer | Tiny | <2 % |
| O9 | **Persistent kernel + L2-aware schedule for einsum** | Persistent CTA + stream-K to reuse L2 across batches | Requires Triton einsum | 5 % |
| O10 | **Batched / grouped LN-stats kernel** | `ln_stats_multirow` (7.3 %) reads x once; can be folded into projection prologue | Combined with O5 | 5–7 % |

## Iteration Plan (≥10 iterations)

Each iteration: pick top-impact open opportunity, generate variant, NCU-profile, council second opinion, commit.

1. **iter1** — O3: Fuse bf16-cast of `out_bmm` into `fused_invtr_ln_gate` (cheapest first-win, isolates precision concerns).
2. **iter2** — O2: Add D=128 thin-K path to matmul_kernel_tlx_ws (dispatcher).
3. **iter3** — O4 partial: Merge two `tr_fwd` launches into one kernel.
4. **iter4** — O8: Workspace-buffer reuse.
5. **iter5** — O1: bf16 Triton einsum with fp32 accum (gated by accuracy on shapes #1/#4).
6. **iter6** — O6: Fold final linear into post-bmm kernel.
7. **iter7** — O5: Fold gate-LN into the 5-proj matmul epilogue.
8. **iter8** — O10: Fold ln_stats into 5-proj projection (collapse pre-matmul kernels).
9. **iter9** — O7: Cluster-2 TMA multicast on einsum (after O1).
10. **iter10** — O9: Persistent + stream-K einsum schedule.

After each iter, update [PROGRESS.md](PROGRESS.md) and create a commit.

Stop conditions: any of (a) iter ≥ 10, (b) accuracy regression we can't fix, (c) two consecutive < 1 % iters.
