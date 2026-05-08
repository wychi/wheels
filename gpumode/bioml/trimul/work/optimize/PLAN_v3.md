# TriMul `hopper_gemm_ws.py` — Optimization Plan v3 (iter11 onward, post-precision-incident)

Picks up from iter10b (shape 6 ≈ 6.0 ms, 1.55× over baseline post-precision-fix).
Driven by:
- The post-incident state in [`reports/precision_postmortem.md`](reports/precision_postmortem.md)
- The original bandwidth analysis in [`reports/iter10_ncu_analysis.md`](reports/iter10_ncu_analysis.md) (still valid for the shape distribution)
- TLX skills under `fbsource/fbcode/triton/tools/kperfagent/.../tlx_prompt/skills/`

PLAN_v2 is superseded — this doc folds in (a) a mandatory precision pre-flight, (b) reordered priorities reflecting the precision risk added by deep-fusion changes, and (c) explicit references to TLX skills with their preconditions.

## Where we stand (iter10b)

| Bucket | shape 6 ms | %wall | Class | Notes |
|---|---:|---:|---|---|
| TLX matmul (5-proj GEMM) | 1.20 | 20 | TC at 65 % peak | unchanged from iter10 |
| `ln_stats_and_bf16_cast` | 1.09 | 18 | DRAM 91 % | unchanged |
| `fused_gate_ln` | 0.95 | 16 | DRAM 92 % | now writes fp32 og (was bf16); +~0.27 GB traffic |
| cuBLAS einsum bmm | 0.54 | 9 | DRAM 80 % | now `.float()` cast on output; +~0.5 GB traffic, ~+250 µs |
| cuBLAS final linear | 0.50 | 8 | TC 82 % | unchanged |
| `tr_fwd_pair` | 0.50 | 8 | DRAM 87 % | unchanged |
| `fused_invtr_ln_gate` | 0.36 | 6 | DRAM 91 % | now used for D=128 too (iter8 reverted) |
| **Pipeline total** | ~6.0 | 100 | — | ~12 GB / 6 ms ≈ 2.0 TB/s avg |

The fp32 promotions added ~1 ms of memory-bound work but eliminated the precision cliff. PLAN_v3 must preserve that margin.

## Mandatory pre-flight for every iteration

Before merging any iter:

1. **Functional verify**: all 7 BENCHMARK_SHAPES pass `_check_vs_ref` with seed 0.
2. **Adversarial sweep**: 30 trials × 3 seeds (731, 17, 4242) × all 7 shapes + 3 extra shapes (cauchy×D=128 mid/small). **Required**: < 1 % failure rate on the sweep, with NO single shape > 5 % failure rate. Use `work/optimize/check_leaderboard_seeds.py`.
3. **Bench**: median across 10 iters per shape via `_bench_one`.
4. **NCU spot-check** if behavior is surprising (compile errors, perf cliff).
5. **Commit**: one iter per commit, with bench delta in the message.

Drop any iter that regresses the adversarial sweep beyond budget — even if perf wins.

## Priority groups (re-ranked given precision risk)

**Group A — Low precision risk, mechanical:**
- iter11: TLX matmul tile/stage sweep (config-only, numerics unchanged)
- iter12: Cluster size 2 + TMA multicast for TLX matmul (numerics unchanged)
- iter19: CUDA Graph capture (numerics unchanged)
- iter20: Workspace pre-allocation (numerics unchanged)

**Group B — Medium precision risk, layout-only:**
- iter14: 2D-tiled `fused_gate_ln` writes `L, R` directly in bmm-friendly layout (kills `tr_fwd_pair`); use `tlx.local_trans` per `transpose_in_shared_memory` skill. Numerics identical to current path; verify with sweep.

**Group C — Higher precision risk, compute fusion:**
- iter13: Re-attempt `fused_invtr_ln_gate_proj` with **fp32 output AND fp32 gated cast removed** (the iter8 path that we reverted). Need to confirm precision matches the cuBLAS path under the adversarial sweep before keeping it.
- iter15: Custom Triton einsum bmm with bf16 input + fp32 output (matches our current `.float()` cast cost but saves the cuBLAS bf16-out + cast roundtrip). High implementation cost, modest expected gain.

**Group D — Deep fusion (deferred until precision discipline is proven):**
- iter16: Fold gate-LN epilogue into TLX matmul (eliminates `proj [T, 5H]` 1.34 GB intermediate)
- iter17: Fold ln_stats + bf16 cast into TLX matmul prologue
- iter18: All-in-one S1 (iter16 + iter17)

These remain the biggest perf levers (~25 % combined) but each one introduces multiple new bf16 cast points and SMEM-staged paths. Per the postmortem (L4), each new bf16 cast risks pushing the cumulative cascade back over the precision cliff. Run iter16 in **isolation** (without iter17) and re-run the adversarial sweep before chaining further.

## Iter-by-iter plan

### iter11 — TLX matmul tile/stage sweep

**Hypothesis.** 65 % Compute SoL on shape 6's 5-proj GEMM. Standard Hopper bf16 GEMM hits 75–85 %; ~0.2-0.3 ms (3-5 %) of headroom.

**Method.**
1. Profile current `matmul_kernel_tlx_ws` with NCU `--section WarpStateStats` to identify dominant stall (Long Scoreboard, Barrier, Wait, etc.). Record in `reports/iter11_ncu_stalls.md`.
2. Per `backbone/tiling_strategy.md`: SMEM cap is **232,448 bytes** (228 KB) on H100; NUM_STAGES first, then BLOCK_N, last BLOCK_M when cutting tiles.
3. Sweep `(BM, BN, BK, NUM_STAGES, NUM_MMA_GROUPS)`:
   - Try `BK=64 NUM_STAGES=4`, `BM=128 NUM_MMA_GROUPS=2`, `BM=256 BN=128 NUM_STAGES=4`.
   - Verify SMEM math: `BM × BK × 2 × NUM_STAGES × NUM_MMA_GROUPS + BK × BN × 2 × NUM_STAGES + barriers ≤ 228 KB`.
4. For shape 6 (M=1M, N=640, K=384): 6 K-iters per output tile is steady-state-friendly.

**Expected.** 3-5 % on shape 6, possibly more on smaller shapes. Numerics unchanged.

**Risk.** Low. Pure config sweep. May be a wash if stalls are barrier-bound (not config-fixable per dump).

**Skills referenced.** `backbone/tiling_strategy.md` (SMEM cap, reduction priority), `backbone/autotune_parameters.md` (constexpr/signature rules if introducing autotune).

### iter12 — Cluster size 2 + TMA multicast

**Hypothesis.** Two CTAs in a cluster share `B_g [D, 5H]` weight tile via TMA multicast — halves L2 pressure on B. The TLX kernel already has the `NUM_CTAS == 2` codepath wired up.

**Method.**
1. Set `TLX_CONFIG["NUM_CTAS"] = 2`.
2. Verify cluster grid math (`NUM_SMS / 2` clusters of 2).
3. Confirm output tiling divides cleanly across the cluster.
4. Fall back to plain L2-resident broadcast if multicast misaligns with N-tile sizes.

**Expected.** 3-5 % on shape 6.

**Risk.** Medium. Cluster launch can fail on certain shapes; need divisibility checks. uTLX cluster shims may need patching.

**Skills referenced.** `optimization/tlx_host_tma.md` (TMA pre-hook patterns).

### iter14 — 2D-tiled `fused_gate_ln` (kill `tr_fwd_pair`)

**Hypothesis.** Make `fused_gate_ln` write `L, R` in `[B, hd, N²]` layout directly. iter7a failed via uncoalesced strided writes; the fix is **SMEM transpose** via `tlx.local_trans`. Saves 0.50 ms (`tr_fwd_pair` entire cost) plus 0.27 GB of `lf, rf` intermediate.

**Method.**
1. Restructure grid to `(B, cdiv(N², TI))`, each program processes TI rows of `proj [T, 5H]`.
2. Compute lv, rv into SMEM tiles `[TI, hd]` (along with the LN-affine + sigmoid + mask + gate-mul).
3. Use `tlx.local_trans` per `backbone/transpose_in_shared_memory.md` to get `[hd, TI]` views with the correct `nv_mma_shared_layout`.
4. Cooperative store: warp `w` writes `L[b·hd + (w·32 .. w·32+32), ij]` — 32 d-rows × TI ij-cols, with consecutive threads on consecutive ij offsets.
5. `og` keeps the original `[T, hd]` layout (consumed downstream as such).

**Expected.** 0.4-0.5 ms (7-9 % e2e).

**Risk.** Medium. Coalescing math is fragile but verifiable with NCU `Memory Throughput TByte/s` and `L2 Sector Promotion Misses`. Per `transpose_in_shared_memory.md`, never load X.T directly via swapped pointer indices.

**Skills referenced.** `backbone/transpose_in_shared_memory.md` (mandatory `tlx.local_trans` usage), `backbone/tensor_memory_layout.md` (SMEM/REGISTER transfer patterns), `optimization/software_pipelining_sync.md` (if pipelining the LN→transpose→store).

### iter13 — Retry `fused_invtr_ln_gate_proj` with stricter precision

**Hypothesis.** iter8's fused-into-final-linear path saved ~6 % on D=128 but failed the adversarial sweep ~2-4 % more often than the cuBLAS path. With our new precision rules (fp32 output, no hardcoded bf16 store cast), the fused kernel should be re-evaluable. Per `accuracy/dtype_precision_debug.md` Pattern 4: keep FP32 across LN, then convert to input dtype only at the dot input.

**Method.**
1. Restore `fused_invtr_ln_gate_proj` for D=128, with:
   - bf16 cast on `gated` ONLY at the `tl.dot` input (mandatory for tensor cores)
   - fp32 output store (no `.to(tl.bfloat16)` on the result)
   - bmm input as fp32 (already done)
2. Run adversarial sweep. Compare fail rate vs current cuBLAS path.
3. Keep iff fail rate is **same or better** than cuBLAS path AND perf wins.

**Expected.** Restore the ~6 % D=128 perf win without precision regression.

**Risk.** Medium. Fail rate may be intrinsically worse; need objective A/B.

**Skills referenced.** `accuracy/dtype_precision_debug.md` (matmul-chain rule + LN exception), `backbone/precision_handling.md` (dtype consistency).

### iter11.5 — pingpong consumer (conditional)

**Hypothesis.** After iter11 picks a config with `NUM_MMA_GROUPS >= 2`, applying `pingpong_consumer` could overlap WGMMAs across the two consumer warpgroups.

**Method.**
1. Per `optimization/pingpong_consumer.md`: hard prerequisites are (a) warp-spec already in place ✓, (b) NUM_MMA_GROUPS ≥ 2 (verify post iter11), (c) SMEM fits with **doubled** consumer footprint under H100's 232,448-byte cap.
2. Compute `smem_bytes = stages × (BM × BK + BK × BN) × bytes_per_elem`. If even the smallest config exceeds cap, **skip this iter** and document why.
3. Otherwise apply named-barrier 9/10 dance with `replicate=2`, init arrive on consumer 1, double the consumer stride.

**Expected.** 5-10 % on shape 6 if SMEM allows.

**Risk.** Medium. Skill explicitly warns about silent regressions if NUM_MMA_GROUPS=1 or SMEM is tight. Document skip if either applies.

**Skills referenced.** `optimization/pingpong_consumer.md` (with all six "Common LLM Rationalizations to Reject" — read before attempting).

### iter15 — Custom Triton einsum bmm

**Hypothesis.** cuBLAS bf16 bmm is 80 % DRAM-bound (132 output CTAs). A custom persistent Triton kernel with L2 grouping could keep `L` tiles in L2 across multiple `R` tiles, cutting ~0.1-0.2 ms.

**Method.**
1. Per `optimization/warp_specialization_producer_consumer.md`: GEMM variant with `replicate=1`, persistent grid `(NUM_SMS,)`, monotonic `accum_cnt` outside tile loop.
2. Tile M=128, N=128, K=64. L2 grouping: each SM walks consecutive `(i, j)` tiles within one batch.
3. Output dtype = bf16 (matches current cuBLAS); fp32 accum.

**Expected.** 0.1-0.2 ms (2-3 %).

**Risk.** High. cuBLAS is a strong baseline. Likely to ship something marginally slower.

**Skills referenced.** `optimization/warp_specialization_producer_consumer.md` (mandatory — phase tracking, `_get_bufidx_phase` helper, scoping rules), `optimization/software_pipelining_sync.md` (cp.async per-thread bytes check), `optimization/tlx_host_tma.md` (Pattern B for static shapes).

### iter16 — Fold gate-LN into TLX matmul epilogue (deep fusion)

**Hypothesis.** `proj [T, 5H]` is the largest unwasted intermediate (1.34 GB write+read = 2.68 GB). Fusing the LN-affine + sigmoid + mask + gate-mul as the TLX consumer's epilogue lets us write only `lf, rf, og [T, hd]` (3 × 0.27 GB = 0.81 GB).

**Method.**
1. After WGMMA acc finishes for an output tile, perform 5 per-output-column LN-correction blocks in registers:
   - `(rs * (acc - mu * s1) + s2)` for each of the 5 hd-wide projections
   - sigmoid for projections 2/3/4
   - multiply lv·lg·m, rv·rg·m
2. Write 3 separate TMA stores (lf, rf, og) instead of one 5-wide store.
3. Pass `mean, rstd, mask, s1, s2` as additional buffers (small).
4. Per `optimization/multi_phase_continuous_pipeline.md`: if the matmul has multiple K-loop phases, ensure every phase has trip count ≥ NUM_STAGES.

**Pre-condition checks (mandatory before code change).**
- `optimization/cross_proxy_fence.md`: register → SMEM stores in epilogue followed by TMA stores need `tlx.fence("async_shared")` between proxy boundaries.
- `accuracy/dtype_precision_debug.md` Pattern 4 + LN exception: keep FP32 for LN math, convert to bf16 ONLY at the final TMA store input.

**Expected.** 0.5-0.7 ms (8-12 % e2e). Single biggest remaining lever.

**Risk.** High. Requires non-trivial TLX changes; epilogue may need its own SMEM staging if 3 separate TMA stores can't issue in parallel. Numerics OK in theory (math is identical), but new bf16 cast points need adversarial-sweep validation.

**Skills referenced.** `optimization/multi_phase_continuous_pipeline.md`, `optimization/cross_proxy_fence.md`, `optimization/warp_specialization_producer_consumer.md`, `accuracy/dtype_precision_debug.md`.

### iter17 — Fold ln_stats + bf16 cast into TLX matmul prologue

**Hypothesis.** `ln_stats_and_bf16_cast` reads x fp32 (1.6 GB), writes mean/rstd/bf16 x (~0.81 GB write + 0.81 GB read at the matmul). Folding into matmul prologue eliminates the bf16 x write+read (1.62 GB).

**Method.**
1. TLX prologue producer task reads fp32 x via TMA into SMEM.
2. A reduction warpgroup computes mean/rstd per row (small per-tile reduce inside SMEM).
3. The bf16-cast happens inside the matmul A-staging path.
4. mean/rstd written to a tiny scratch buffer for the epilogue (iter16).

**Expected.** 0.4-0.6 ms (7-10 % e2e).

**Risk.** Very high. Combining a per-row reduction with the matmul producer pipeline is a major TLX rework.

**Skills referenced.** Same as iter16 plus `accuracy/reduction_stability_debug.md` (broadcast/axis rules — relevant since LN is a reduction).

### iter18 — All-in-one S1 (iter16 + iter17 combined)

**Hypothesis.** Combined, eliminate both `bf16 x` and `proj` intermediates; replace `ln_stats + matmul + fused_gate_ln` with a single kernel that reads fp32 x once and writes `lf, rf, og` only. Total HBM: 1.6 GB read + 0.81 GB write.

**Expected.** Matches iter16 + iter17 wins, possibly slightly better (1.0-1.3 ms, 17-22 % e2e). Approaches theoretical pipeline lower bound.

**Risk.** Very high — only attempt after iter16 and iter17 succeed individually.

### iter19 — CUDA Graph capture

**Hypothesis.** Static pipeline of 7+ kernels has a launch latency floor. Capture once, replay each call. Saves ~50 µs/iter on small shapes (5-10 %).

**Method.**
1. First call records into `torch.cuda.CUDAGraph` after warmup.
2. Subsequent calls call `graph.replay()`.
3. Need stable input pointers — use a workspace buffer and copy input in.

**Risk.** Low. Standard pattern. May not compose with the weight cache (graph captures specific tensor pointers); need to invalidate graph on cache miss.

### iter20 — Workspace pre-allocation

**Hypothesis.** `torch.empty(...)` per call adds CUDA caching-allocator overhead. Pre-allocate `mean, rstd, lf, rf, og, L, R, out_bmm, out` as a workspace tied to (id(x_in), shape).

**Expected.** < 2 % on small shapes, ~0 % on large.

**Risk.** Low. Need to handle workspace growing if shapes change.

## Stop conditions

- Reach within 10 % of theoretical lower bound (~3.5 ms on shape 6).
- 3 consecutive iters with < 1 % gain.
- iter16/17 failure with no clear path forward.
- **Adversarial sweep failure rate exceeds 1.5 %** on any iter — revert and re-evaluate before proceeding.

## Tracking

- Append to [`PROGRESS.md`](PROGRESS.md) after each iter.
- Adversarial sweep result table per iter, `(shape, seed) → fails/30 + max_err`.
- Commit per iter with bench delta and adversarial sweep delta in the message.
- Council second-opinion at iter15 (mid-batch) and iter20 (wave summary).
