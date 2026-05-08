# Per-GPU Tailoring — Index & Cross-GPU Summary

This directory tailors the AlphaFold3 TriMul fused-subgraph kernel to four GPU targets. Per-GPU detail lives in:

- [`a100.md`](a100.md) — NVIDIA Ampere SM80 (legacy / verification)
- [`h100.md`](h100.md) — NVIDIA Hopper SM90 (**current submission target**, includes profiling validation)
- [`b200.md`](b200.md) — NVIDIA Blackwell SM100 (prototype)
- [`mi300.md`](mi300.md) — AMD CDNA3 gfx942 (high-effort fork)

For the underlying decomposition see [`../fused_subgraphs.md`](../fused_subgraphs.md). For the problem spec / benchmark shapes see [`../trimul_problem.md`](../trimul_problem.md). For profiling artifacts referenced from the H100 page, see [`../../profile/`](../../profile/).

## Shared Background

The benchmark harness fixes `H = 128` (head dim) and sweeps `D ∈ {128, 384}`, `S ∈ {256, 512, 768, 1024}`, `B ∈ {1, 2}`. The largest shape is `B=1, S=1024, D=384, H=128`. Some configurations run with `nomask=true`.

The fused pipeline has three hot regions:
- **Subgraph 1 (S1)** — fused LayerNorm + 5 linear projections (`a, b, gA, gB, gO`) over `[B, S, S, D]`. Memory-heavy.
- **Einsum** — `i k d, j k d -> i j d` (two `[B, S, S, H]` operands, one `[B, S, S, H]` accumulator). Compute-heavy at `H=128`.
- **Subgraph 3 (S3)** — output gate + linear projection. Memory-heavy.

### IO accounting note

Subgraph 1 IO at the largest shape:
- **Conservative unfused** (5 separate projections re-reading `x_norm`) ≈ 2.5 GB
- **Cache-aware fused** (single read of `x_norm`) ≈ 1.2 GB (planning estimate)
- **Profile-validated** (1× input read + 2 bf16 outputs + 1 fp32 out_gate, weights negligible) ≈ **2.68 GB**

The H100 profile (see [`h100.md`](h100.md)) confirms the 2.68 GB number — Subgraph 1 IO was underestimated in earlier roofline tables. The 1.2 GB figure assumed more aggressive intra-kernel cache reuse than the production kernel actually achieves.

### FLOP accounting

- **Subgraph 1**, largest shape: `2 · B · S² · D · 5H = 2 · 1 · 1,048,576 · 384 · 640 ≈ 515 GFLOP`. (Earlier drafts that quoted 5,120 GFLOP or 1.28 GFLOP are incorrect.)
- **Einsum**, largest shape: `2 · B · S³ · H = 2 · 1 · 2³⁰ · 128 ≈ 274 GFLOP`. Comparable to S1 in raw FLOPs but much higher arithmetic intensity per byte moved.

---

## Cross-GPU Summary

| Metric | A100 | H100 | B200 | MI300X |
|---|---|---|---|---|
| Peak bf16 TF/s | 312 | 989 | ~2,250 (sustained ~1,800) | 1,307 |
| HBM BW (TB/s) | 2.0 | 3.35 | 8.0 | 5.3 |
| L2 / cache (MB) | 40 | 50 | 126 | 256 (Infinity) |
| SMEM/LDS per SM/CU (KB) | 164 | 228 | 227 (+256 KB TMEM) | 64 |
| Einsum tile (M×N×K) | 128×128×32 | 128×256×64 | 256×256×64 (2-CTA) | 128×128×32 |
| Einsum num_stages | 3 | 4 | 5–6 | 2 |
| B+C fusion verdict | **No** | **Yes** | **Marginal** | **No** |
| Peak e2e (ms) | ~3.0 | ~0.94 | ~0.41 | ~0.71 |
| Realistic e2e (ms), measured | — | **10.06** ⚠️ measured (largest shape, current impl) | — | — |
| Realistic e2e (ms), planning estimate | 4.0–4.5 | ~~1.5–1.8~~ (was 6× too optimistic) | 0.65–0.95 | 1.0–1.3 |
| All-bf16 doc design (predicted, unbuilt) | n/a | ~3–4 (if accuracy gate passes) | n/a | n/a |
| utlx wheel support | partial (sm80 fallback paths) | **canonical target** | beta (sm100 lowering) | **none** (NVIDIA-only wheel) |
| Viability for submission | Low payoff (legacy) | **Proceed (low effort, high payoff)** | Prototype (medium effort, very high payoff) | High-effort fork (medium payoff) |

> ⚠️ Only the H100 row is profile-validated, and the measurement reveals the **planning estimate was 6× too optimistic** — the actual implementation runs in 10 ms, not 1.5–1.8 ms, because it pays a 60%+ precision tax (bf16↔fp32 casts + fp32 cuBLAS einsum) to handle cauchy-distributed inputs at S ≥ 512. The A100/B200/MI300 numbers assumed similar 60–75% utilization and similarly underestimated precision/cast costs; treat them as **lower-bound** planning estimates that could easily be 3–6× higher in practice. See [`h100.md`](h100.md) → "End-to-End Measurements" for full breakdown.

---

## B+C Fusion Per-GPU Verdict

The earlier "universal 17% win" claim is **overstated**. The fusion's payoff depends on whether the H-dim intermediate buffer fits in SMEM/LDS *with double-buffering*, and whether the saved 0.13–0.5 ms of intermediate I/O is larger than the launch and synchronization overhead of the fused kernel.

- **A100 — NO.** 16×16 fused buffer (~128 KB H-vector) exceeds the workable single-buffer footprint inside 164 KB SMEM. Single-buffer-only kills pipelining; shrinking to 8×8 collapses parallelism. Net loss.
- **H100 — YES.** 228 KB SMEM + WGMMA pipeline absorbs the fused intermediate at 16×16 with double-buffering. Saves ~0.13 ms intermediate I/O — **3–5% e2e win**. This is the case the original analysis was actually based on.
- **B200 — MARGINAL.** TMEM+SMEM headroom is ample, so fitting the buffer is not the issue. But the einsum is so brief (~0.12 ms) that fusion synchronization can break even or lose 1–2%. Worth doing only after baseline tcgen05 is stable. Defer.
- **MI300X — NO.** 64 KB LDS forces an 8×8 fused tile, halving useful parallelism. Sync overhead exceeds the ~190 µs intermediate I/O savings. Net loss.

**B+C fusion is an H100-specific optimization, not a universal win.** Treat it as a Hopper recipe and disable on the other three targets.

---

## Council Disagreements & Open Questions

1. **Subgraph 1 IO size: 2.5 GB vs 1.2 GB vs 2.68 GB.** The H100 profile resolves this for H100 (effective ~2.68 GB), but on GPUs with much larger L2 (B200 126 MB, MI300 256 MB Infinity Cache) cross-projection reuse may genuinely deliver lower effective IO. Until measured, plan with the 2.68 GB number on H100 and flag the others as TBD.
2. **B200 sustained vs peak clock**: cite ~1,800–2,000 TF/s as the realistic peak under sustained TriMul load (1000 W TDP throttles); 2,250 TF/s is datasheet max.
3. **B200 tcgen05 maturity**: uTLX/Triton lowering is beta. Council split on whether to ship tcgen05 or fall back to cuBLASLt + WGMMA. Recommendation: write both, gate by config, ship whichever is faster.
4. **MI300 wave32 vs wave64**: wave32 may give 5–10% on the einsum tile via higher waves/CU occupancy. Untested; flag as a tuning knob, not a recommendation.
5. **CUDA Graphs threshold**: definitely required on B200 (kernels < 0.1 ms each); marginal on H100 (~5% if launch overhead is observed in profiling); not needed on A100/MI300.
6. **utlx wheel scope**: NVIDIA-only currently. Whether to invest in an AMD fork is a separate ecosystem decision, not a per-kernel one.
7. **End-to-end estimates spread**: realistic figures here use 60–75% of peak. The H100 profile shows the current S1 kernel hits ~45% of peak HBM, suggesting our planning numbers were too optimistic. Re-validate other GPUs once a kernel actually runs there.
