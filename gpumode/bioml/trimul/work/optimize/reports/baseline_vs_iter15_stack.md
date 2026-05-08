# Baseline vs iter15+iter13 stack — where the 1.81× came from

Comparison of the original kernel (`profile/e2e_results.md`, pre-iter1) against the current stack (HEAD: iter15-einsum + iter13, commits `e08bce7` + `07f0c5e`). Target: shape 6 (B=1, S=1024, D=384), H100, 20 iters via torch.profiler.

NCU artifacts:
- Baseline: `profile/baseline.ncu-rep` (matmul only, smaller shape — not directly comparable to shape 6 SoL).
- Current: `work/optimize/profile/iter15_stack/{matmul,fused_gate_ln,bmm}/baseline_full.ncu-rep` (shape 6, full sections).

## Per-call kernel cost — shape 6

| Kernel | Baseline (ms) | Current (ms) | Δ | Notes |
|---|---:|---:|---:|---|
| `matmul_kernel_tlx_ws` | 1.192 | 1.213 | +0.02 | Same kernel, ~unchanged |
| `vectorized_elementwise` (bf16↔fp32 casts) | **2.503** | 0 | **−2.50** | folded into iter6 ln_stats / iter15 bmm |
| `unrolled_elementwise` (bf16↔fp32 casts) | **1.564** | 0 | **−1.56** | folded into iter15 bmm fp32 epilogue |
| `ln_stats_multirow` | 0.728 | 0 | −0.73 | merged into `ln_stats_and_bf16_cast` (iter6) |
| `ln_stats_and_bf16_cast` | 0 | 1.103 | +1.10 | new fused kernel |
| `fused_gate_ln` | 0.999 | 0 | −1.00 | merged into iter14 |
| `fused_gate_ln_bmm_layout` | 0 | 1.190 | +1.19 | gate_ln + tr_fwd, writes bmm layout |
| `tr_fwd` (×2) | 0.512 | 0 | −0.51 | folded into iter14 |
| `fused_invtr_ln_gate` (D=384 only) | 0.368 | 0.595 | +0.23 | reads fp32 now (iter10b precision) |
| `fused_invtr_ln_gate_proj` (D=128 only) | 0 | (D=128 path) | n/a | iter13 fused proj |
| cuBLAS einsum bmm (fp32) | 1.012 | 0 | −1.01 | replaced by `bmm_kernel_tlx_ws` |
| `bmm_kernel_tlx_ws` (bf16-in, fp32-out) | 0 | 0.797 | +0.80 | iter15 new TLX bmm |
| cuBLAS final linear (fp32) | 1.105 | 0.550 (bf16) | −0.55 | iter4 dtype swap |
| **Total** | **~10.06** | **~5.55** | **−4.5 ms (1.81×)** | |

## Where the wins came from

1. **Cast elimination — 4.07 ms saved.** Half of the baseline's wall time was bf16↔fp32 casts in `vectorized_elementwise` + `unrolled_elementwise`. iter6, iter14, iter15 progressively folded these into the kernels that produced or consumed the data, leaving zero standalone cast kernels.
2. **Pipeline fusion — fewer launches, less HBM round-trip.** 7 kernels collapsed to 5: `fused_gate_ln` + 2× `tr_fwd` → `fused_gate_ln_bmm_layout` (iter14); `ln_stats_multirow` + cast → `ln_stats_and_bf16_cast` (iter6).
3. **GEMM dtype swaps — 1.6 ms saved.** Final linear (iter4) and einsum bmm (iter5) went fp32→bf16. cuBLAS routes to Hopper Tensor Cores (~1000 TF/s) instead of CUDA cores (~67 TF/s).
4. **Custom TLX bmm with fp32 epilogue (iter15)** — saved the 451 µs post-bmm bf16→fp32 cast pass that cuBLAS couldn't avoid.
5. **D=128 fused proj retried (iter13)** — restored the iter8 fused inv-tr+LN+gate+H→D linear with strict precision. Mooted the standalone wout→fp32 attempt entirely; gave back the iter10b D=128 precision tax (-26% on D=128 shapes) without regressing adversarial fail rate (1.53% → 0.95%).

## What did NOT improve — the matmul ceiling

Per-NCU on shape 6 (full `--set full` capture):

| Metric | Baseline matmul (~iter10 NCU) | Current matmul |
|---|---:|---:|
| Duration | ~1.19 ms | 1.213 ms |
| %TC SoL | ~65% | 65.2% |
| %DRAM SoL | ~70% | 71.4% |
| %L2 hit | ~80% | 80.0% |
| Long-scoreboard stalls | dominant | 48% (still dominant) |

The `matmul_kernel_tlx_ws` 5-proj GEMM is **essentially unchanged** between baseline and current. Four iters tried to push it:
- iter11 (tile/stage sweep): NO WIN — already at local SMEM optimum.
- iter12 (cluster + TMA multicast): BLOCKED — uTLX wheel doesn't register `ttng.map_to_remote_buffer`.
- iter17 (ln_stats prologue fold): bailed at static analysis — naive scheduler gives 5× redundant fp32-x reads (num_pid_n=5 on shape 6); the "real" fix needs a row-persistent scheduler rewrite.
- iter18 (all-in-one S1): uTLX MLIR codegen crash on conditional barriers in multi-warpgroup tasks; phase B blocked by 80 KB register cliff (5 fp32 accs > 64 KB consumer warpgroup register file).

**Interpretation**: the 65% TC SoL ceiling is a fixed architectural constant under the current uTLX wheel. The wins came from **eliminating work around the matmul** (casts, intermediate-buffer round-trips, dtype-induced slow paths), not from making the matmul itself faster.

## NCU headroom estimates on current stack

| Opportunity | Kernel | NCU "Est. local speedup" | Approx e2e impact |
|---|---|---:|---:|
| L2 compression (writes 0% compressed today) | fused_gate_ln_bmm_layout | 33.5% | ~7% |
| L2 compression | matmul | 21.3% | ~5% |
| L2 compression | bmm | 23.8% | ~3% |
| SMEM bank conflicts (`local_trans` on B) | bmm | 28% | ~1% |
| SMEM bank conflicts | matmul | 8.3% | ~2% |
| FP32 fused/unfused FMA mix | fused_gate_ln_bmm_layout | 6.4% | ~1.5% |

L2 compression is the most interesting — across all three TLX kernels the compression unit sees ~3 GB of writes per call but compresses 0%. A config-level investigation (allocation flags, `__align_value`) could unlock 10-15% e2e at near-zero implementation cost. The bmm bank-conflict fix (store B as `[BK, BN]` directly instead of `[BN, BK] + local_trans`) is a surgical ~50 µs e2e win.

## Per-shape end-to-end (baseline → current stack, ms)

| # | bs | seqlen | dim | dist | baseline | current | speedup |
|---|---|---|---|---|---:|---:|---:|
| 0 | 2 | 256 | 128 | normal | 1.039 | 0.603 | 1.72× |
| 1 | 1 | 768 | 128 | cauchy | 4.090 | 2.440 | 1.68× |
| 2 | 2 | 256 | 384 | normal | 1.370 | 0.803 | 1.71× |
| 3 | 1 | 512 | 128 | normal | 1.866 | 1.098 | 1.70× |
| 4 | 1 | 1024 | 128 | cauchy | 7.334 | 4.238 | 1.73× |
| 5 | 1 | 768 | 384 | normal | 5.639 | 3.169 | 1.78× |
| 6 | 1 | 1024 | 384 | normal | 10.059 | 5.550 | **1.81×** |

Adversarial sweep (6 input seeds × 7 shapes × 30 trials = 1260 runs): **0.95% combined fail rate**, vs iter14's 1.03%. Functional verify: max_err 0.012-0.016 on all 7 shapes (atol gate 0.02).

## Lessons (durable)

1. **Cast elimination dominates pipeline-fusion wins on this kernel.** Half the baseline was elementwise casts. Anywhere bf16↔fp32 round-trips appear between kernels, they can be folded into the producer's epilogue or the consumer's prologue with no precision cost.
2. **The TLX matmul has zero SMEM headroom for additional staging at current tile size.** Both iter16 (epilogue fold) and iter17 (prologue fold) hit the 228 KB cap. Going further requires a tile-geometry rewrite or a row-persistent scheduler — both major undertakings.
3. **uTLX wheel is the ceiling for the exotic stuff.** Cluster TMA multicast (iter12) and conditional barriers in multi-task warpgroups (iter18) both hit wheel bugs, not algorithmic walls.
4. **Custom TLX bmm beats cuBLAS not by being faster at the matmul, but by absorbing the surrounding cast.** iter15-einsum's standalone bmm is only 1.26-1.40× over `torch.bmm`; the e2e win came from `+ .float()` post-cast being eliminable in the epilogue.
5. **Precision discipline is non-negotiable.** iter10b documented the bf16-cascade rule (LN math FP32, cast to bf16 only at the dot input); iter13 applied it strictly to recover both the perf and the precision the precision-fix attempts couldn't deliver.
