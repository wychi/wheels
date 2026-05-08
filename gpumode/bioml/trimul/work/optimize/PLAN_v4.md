# TriMul `hopper_gemm_ws.py` — Optimization Plan v4

Builds on the stack from iter15 (current best: shape 6 **5.35 ms**, geo-mean **1.88×** over baseline). Acknowledges 5 hard ceilings from PLAN_v3 and proposes 8 concrete iters routing **around**, not through, those walls.

## Where we stand (iter15)

| Metric | Value | Notes |
|---|---|---|
| Shape 6 wall (B=1, S=1024, D=384) | 5.35 ms | 1.88× baseline → **target: 3.5–4.0 ms (50–60% further win)** |
| Geo-mean (all 7 shapes) | 1.88× baseline | vs 1.04× baseline at iter0 |
| Adversarial sweep fail rate | 0.0% (0/210) | Robust across 7 shapes, 30 trials, seed=731 |
| Pipeline HBM throughput (shape 6) | ~2.0 TB/s avg | Still ~60% idle bandwidth; L2 hit rates 50–84% per kernel |
| Bottleneck buckets | 5 kernels + 2 cuBLAS calls | Matmul 22% + fused_gate_ln 22% + ln_stats 20% + bmm 15% + rest |

**Critical observation:** iter15 discovered that the real win was **eliminating the post-bmm bf16→fp32 cast** (451 µs on shape 6), not beating cuBLAS at the matmul itself. This pattern — **look for implicit type-conversion bottlenecks** — should guide all future iters.

## 5 Hard ceilings to route around

1. **SMEM cap (228 KB)** — all aggressive fusions (iter18 Phase B with 5 accs, iter17 prologue-fold) hit register cliff first (80 KB consumer wg) or SMEM overflow. Any new fused kernel must fit entirely under 228 KB including **prologue scratch + matmul stages + epilogue staging**. Baseline matmul uses 98 KB at `BM=128, NS=3, NS_B=2`; only ~130 KB remain.

2. **uTLX wheel cluster/multicast ops** (`ttng.map_to_remote_buffer` not registered) — iter12 blocked. Multi-CTA TMA multicast requires Triton+uTLX patch or version bump. Defer to future Triton maintenance windows.

3. **uTLX multi-warpgroup conditional barrier MLIR crash** — iter18 Phase A attempted conditional barrier gates inside multi-task warpgroups; uTLX codegen emits invalid MLIR (`Builders.cpp:436` crash). Conditional barriers are safe in **single-task producers or single-consumer warpgroups**, not in multi-warpgroup forked control flow.

4. **Consumer warpgroup register file (~64 KB)** — iter18 discovered that keeping **5 independent fp32 accs of [32, 128]** (the cross-column multiply state for lv, rv, lg, rg, og) requires 80 KB, overflowing the ~64 KB consumer reg file. This caps all "all-5-in-registers" fusion designs.

5. **Round-robin schedule redundant reads** (`tile_id += NUM_SMS`) — iter17 never attempted, but analysis shows the TLX matmul's Group-M persistent scheduler reads x[B,S,S,D] redundantly when fusing the prologue. With `GROUP_SIZE_M=1` and `num_pid_n=5`, 5 consecutive consumer iterations **should** share x but the producer iterates `tile_id += 132`, decoupling M and N iteration order. A proper fix requires row-persistent scheduler rewrite.

## Tiered precision test methodology

Current full adversarial sweep: 7 shapes × 3 seeds × 30 trials = 630 runs, ~10 min.

**Quick-test (< 2 min): Precondition gate for all iters**
- Shapes: 1 (cauchy D=128), 4 (cauchy D=128), 6 (normal D=384)
- Seeds: 731 (primary), 17 (secondary)
- Trials: 3 per shape×seed combo = 18 runs total
- **Abort threshold:** if any shape > 2% failures on quick-test, do NOT proceed to medium.
- Rationale: D=128 shapes, especially cauchy distribution, historically fail precision first (see iter10b postmortem). Shape 6 is the perf target.

**Medium-test (3–4 min): Gating for medium/high-risk iters**
- Shapes: all 7
- Seeds: 731, 17
- Trials: 10 per shape×seed combo = 140 runs
- **Gate threshold:** < 1% failures (< 2 runs) across all 7 shapes, no single shape > 2%.
- Use: before committing any iter that touches precision or new cast points.

**Full-test (8–12 min): Final validation before rebase**
- 7 shapes × 3 seeds × 30 trials = 630 runs (existing `check_leaderboard_seeds.py`)
- **Gate threshold:** < 1% failures (< 7 runs), no single shape > 5%, worst max_err ≤ 0.040.
- Use: after stacking 2–3 iters, before merge to main.

**Why this tiered approach:** D=128 + cauchy is the **adversarial sensitivity floor**. If any iter survives quick-test (18 runs on the hard shapes), it's likely safe to expand. Saves ~6 min per iter in the gating loop.

## Iter-by-iter plan

### iter16 — L2 compression unlock (empirical tuning)

**Hypothesis.** NCU profiling on shape 6 shows L2 writes **0% compressed** across all kernels (matmul: 21% compression potential, fused_gate_ln: 33%, bmm: 24%). H100 L2 compression hardware is not activated. Config is likely either a Triton codegen default (compression disabled for correctness by default) or an NCU measurement artifact (disables compression in restricted sampling).

**Method.**
1. Run `nvprof --metrics l2_cache_write_bytes_compressed` on one shape (e.g., shape 6) baseline to confirm 0% or near-0%.
2. Search Triton + LLVM codegen for L2 compression disable flags; check if `clang -mllvm -amdgpu-enable-compact-kernel-frame` or equivalent H100 gating exists.
3. If toggle found: apply to matmul_kernel, fused_gate_ln, bmm_kernel in sequence and profile impact per kernel.
4. If no toggle exists: confirm via NVIDIA that H100 compress-on-write is automatic and NCU sampling is the artifact — no action needed.

**Expected gain.** 7–15% if toggle exists. Safe to spec-out first (no code change yet).

**Risk class.** Low. Pure config/codegen investigation. No precision risk.

**Preconditions.** None (pure profiling).

**Abort criteria.** If no compression toggle found OR if enabled compression breaks kernel outputs (silent data corruption from incorrect decompression).

**Acceptance gate.** NCU shows L2 writes ≥ 15% compressed on matmul/gate_ln; e2e ≥ 3% speedup on shape 6, no adversarial regression.

**Routes around ceiling.** None directly; is a "free" win orthogonal to all ceilings.

---

### iter17a — LN input normalization without x re-read (recompute-inside-LN)

**Hypothesis.** `ln_stats_and_bf16_cast` loads fp32 x once and writes bf16 x. Per-thread footprint is tiny. But per-row mean/rstd reductions are fully work-inefficient at small M-tile sizes. Consider **inlining x load + LN-norm into the matmul producer's first K-iter prologue** to amortize the x bandwidth over the full matmul K-loop. Skip the intermediate bf16 x buffer.

**Method.**
1. Modify `matmul_kernel_tlx_ws` producer: on the **very first K-iter per M-tile**, read `x_tile[BM, K_ITERS*BK]` into SMEM (replaces `B_tile` load for that first iter).
2. Reduction warpgroup (currently only used for TMA descriptor setup? Or unused?) computes `mean, rstd` over the `[BM, K]` reduction axis **in parallel with first B_tile load via second TMA pipeline**.
3. Consumers wait on `mean, rstd` ready (single named barrier), then apply `(x - mean) * rstd * scale + bias` at WGMMA input before the dot — bf16-cast only at that point.
4. SMEM layout: TMA prod produces A (fp32 x) into **dedicated 64 KB buffer**, reduction wg reads it and writes `mean, rstd [BM]` into a tiny `[BM, 2]` FP32 staging area. Both overlap with B_tile SMEM (which is unused during x prologue).
5. Exit prologue: zero out the x staging area and restore normal matmul mode.

**Expected gain.** 5–8% (eliminates the full `ln_stats_and_bf16_cast` kernel cost, **but** adds serialization overhead if x-read competes with B-read on the memory bus). Likely a wash or slight win on shape 6 (BW-saturated); bigger win on smaller shapes (shape 0 where `ln_stats` is proportionally larger).

**Risk class.** Medium-high. Introduces multi-purpose SMEM buffers and conditional control flow in the producer. Requires careful barrier coordination to avoid deadlock when K_ITERS=1 (D=128 shapes).

**Preconditions.** 
- Verify SMEM math: x prologue 128 (BM) × 384 (D, max K) × 4 = 196 KB, leaves 32 KB for mean/rstd + barriers. **Marginally fits if B staging is temporarily repurposed.** Likely requires `BM_SPLIT` logic: load x in two halves if BM=256.
- Confirm no uTLX syntax for "skip this warpgroup on iteration N" — the reduction wg should be **disabled after prologue** to avoid idle power. This is a new constraint.

**Abort criteria.** SMEM math doesn't fit under 228 KB even with BM_SPLIT. Or MLIR codegen crashes on conditional warpgroup enable/disable (likely, given iter18's multi-warpgroup barrier crash).

**Acceptance gate.** Quick-test passes. Median shape 6 speedup ≥ 1%, geo-mean ≥ 0.5%, zero adversarial regression.

**Routes around ceiling.** Avoids redundant M-reads by **fusing the prologue into the matmul pipeline itself**, which is where the M-reuse naturally lives (Group-M persistent). Does not hit SMEM or register cliffs (prologue is time-critical, not space-critical).

---

### iter17b — Fuse LN epilogue into fused_gate_ln_bmm_layout (LN-affine + gate moved earlier)

**Hypothesis.** `fused_gate_ln_bmm_layout` currently computes `proj[T,5H] → LN → (lv, rv, lg, rg, og)` then stores `lv, rv, og` in different layouts. What if we move the `LN(proj)` **computation into iter14's fused kernel** so it's done in-SMEM before the transpose to `[B*hd, N²]` layout? The LN-affine (scale + bias) are tiny (2×hd per row); the gate-mul (sigmoid) is already there.

**Method.**
1. Load `s1, s2` (LN weight & bias) as a small constant or precomputed per-batch.
2. In `fused_gate_ln_bmm_layout`, after computing raw `proj` in SMEM, apply `(proj - mu*s1)*rstd*s1 + s2` inline (in-place normalization using loaded `rstd` from upstream). Keep `proj[T, 5H]` in fp32 until the final cast to bf16.
3. Gate multiply `og * sigmoid(...)` stays in-kernel.
4. Transpose `(lv, rv, lg, rg)` → `[B*hd, N²]`, apply the cross-column multiply (lv*lg, rv*rg), write bf16.

**Expected gain.** 2–3% (eliminates the redundant `LN(proj)` reads — but the normalized data still needs to be materialized somewhere, likely in SMEM, before the transpose). The real win is **one fewer kernel launch** and potential for pipelined synchronization.

**Risk class.** Low-medium. No new precision hazards (LN math is unchanged, just moved). The transpose is already in the kernel, so coalescing shouldn't change. Risk is register pressure if trying to keep all 5 N-chunks in-flight simultaneously (already forbidden by iter18's discovery).

**Preconditions.** 
- Verify SMEM fits: `fused_gate_ln_bmm_layout` currently stagess `proj[TI, 5H]`, then transposes + stores. Adding in-SMEM LN-affine (no extra alloc, just a few extra FMA per element) should not overflow 228 KB. TI is typically 4–8; TI=4, hd=128, 5H=640 → SMEM ≈ 12 KB for proj, well under budget.

**Abort criteria.** Register pressure spills (visible in NVVP as spill_size > 0). Or if `fused_gate_ln_bmm_layout` SMEM is already at 180+ KB (check with `_get_kernel_smem_usage()`).

**Acceptance gate.** Quick-test passes. Shape 6 speedup ≥ 0.5%, no adversarial regression.

**Routes around ceiling.** Stays within existing fused_gate_ln_bmm_layout kernel's SMEM/register footprint; does not attempt to keep 5 accs for cross-column work (that failed in iter18). Exploits the **fact that LN is already part of the gate-LN kernel** — just reorders the operations.

---

### iter19 — D=128-specialized kernel dispatch (gate-LN epilogue fusion revisited)

**Hypothesis.** iter16 (-5 to -10% on D=128) won fnd on D=128 shapes but cratered on D=384 (SMEM cliff: needs NS_A=6 for K=384, which exceeds 228 KB). What if **only the D=128 code path runs the fused gate-LN epilogue**, while D=384 keeps the 2-kernel path (matmul + separate fused_invtr_ln_gate)? Gains back the D=128 wins without the D=384 regression.

**Method.**
1. Resurrect the iter16 gate-LN epilogue fusion kernel (`fused_matmul_gated_final_bf16_d128` or similar).
2. At the start of `forward()`, check `dim == hd` (D=128 only) and dispatch to a **specialized `trimul_forward_d128()`** function that:
   - Uses `matmul_kernel_tlx_ws_gated_final_d128` (iter16's fused version).
   - Otherwise identical pipeline.
3. For D=384 and D > hd, call the standard path (`forward()`).
4. Combine both paths in a single `submission.py` via an if-else.

**Expected gain.** Restore 5–10% on shapes 0, 1, 3, 4 (all D=128), 0% on D=384 (shapes 2, 5, 6).

**Risk class.** Low-medium. Code duplication is real (two forward paths), but the D=384 path is unchanged (known stable). D=128 path re-uses iter16's kernel, which passed adversarial sweep in the original commit log (though was reverted later — check if that was precision or perf).

**Preconditions.** 
- Re-examine iter16's commit message and adversarial results. If it was a **precision fault**, this iter is blocked. If it was a **perf cliff on D=384 only** (register/SMEM), proceed.
- Verify the D=128-specific kernel is alive in the `work/` directory or can be reconstructed from git history.

**Abort criteria.** iter16 was reverted due to precision fault. Or adversarial sweep on D=128 shapes shows > 2% failures (if precision is still marginal).

**Acceptance gate.** Quick-test passes (especially shapes 1, 4 in D=128). Geo-mean ≥ 2% speedup vs iter15 baseline. No regressions on D=384.

**Routes around ceiling.** Sidesteps the **SMEM register pressure ceiling** by **admitting defeat on D=384** and only applying deep fusion where it's safe (D=128, where K=D=128 is smaller than D=384, so the tile config is less constrained).

---

### iter20 — Custom Triton final linear (D=384 path only)

**Hypothesis.** The cuBLAS bf16 final linear (`F.linear(gated, W_out_bf16)`) costs 0.55 ms on shape 6. We control `gated` layout and could write a **persistent Triton kernel** that reads `gated [T, hd]` bf16 and outputs `[T, D]` fp32 in one pass, without dispatching through cuBLAS. For D=384, hd=128, this is a `[T, 128] @ [128, 384]` GEMM; should be faster than the dispatch overhead alone.

**Method.**
1. Write `final_linear_kernel_tlx` — persistent warp-spec GEMM over `[T, 128]` and weight `[128, D]` (D is fixed at 384 for this kernel variant).
2. 3D TMA descriptors for A (gated [T=1024, 128]), B (W_out_T [384, 128]), C_out [T, 384].
3. Tile config: `BM=128, BN=384 (or split to 192+192), BK=64, NS=2` (K is small: 128/64 = 2 K-iters). SMEM: 64×128×2 (A) + 384×64 (B) ≈ 16+24 = 40 KB — very comfy.
4. Epilogue: reduce fp32 with mixed bf16 math (bf16 input matmul, fp32 accumulate). Write output directly as fp32.
5. Dispatch from `forward()`: if D != hd, run custom `final_linear_kernel_tlx` instead of `F.linear(...)`.

**Expected gain.** 10–15% on D=384 shapes (0.55 ms → 0.47 ms). Smaller on D=128 (where final linear is cuBLAS's ≈0.25 ms on shape 4, likely already dispatch-limited and not critical).

**Risk class.** Low. Matmul design is vanilla Hopper warp-spec; no precision hazards (bf16 input, fp32 accum/output is the proven formula). Small K means the pipeline may not reach steady-state (2 K-iters), so watch for barrier overhead — may need `NUM_STAGES=1` or a direct-reduce epilogue.

**Preconditions.** 
- SMEM math ✓ (40 KB << 228 KB).
- Confirm `BN=384` does not cause register spillage. (If it does, try `BN=192` in sequence.)
- Verify Triton's bf16 WGMMA + fp32 accum + direct fp32 store in epilogue works (should; it's the matmul_kernel pattern).

**Abort criteria.** NVVP shows spill on `BN=384` and `BN=192` still fails. Or wall-time is actually worse than cuBLAS (unlikely, but dispatch overhead is not zero).

**Acceptance gate.** Quick-test passes. Shape 6 speedup ≥ 5%, geo-mean ≥ 2%, no adversarial regression.

**Routes around ceiling.** Orthogonal to all 5 ceilings. Custom kernel is independent, doesn't fuse with other stages, doesn't use multi-warpgroup barriers (single task).

---

### iter21 — "Pinch" the bmm matmul-ette tile config for shape-dependent speed

**Hypothesis.** iter15's `bmm_kernel_tlx_ws` uses a fixed config (`BM=128, BN=128, BK=64, NS=3`). This is tuned for the largest shapes (shape 6: 1024×1024 × 128 batch). Smaller shapes (shape 0: 256×256 × 2 batch, shape 2: 256×256 × 2 batch) might prefer smaller tiles (BM=64, NS=1) to reduce register pressure + occupancy loss. Per iter2's lesson, small shapes are already fast; the win is in reducing per-tile overhead when the grid is smaller.

**Method.**
1. Auto-compute optimal tile config based on total number of output tiles. If `num_tiles_total < 1000`, use "small-tile" config (`BM=64, NS=1`). Otherwise use the current config.
2. Or, profile bmm on shape 0 and shape 6 in isolation with a small sweep (1–2 configs each) and pick shape-specific constants.
3. Keep the single kernel, route the config at launch time (compile two versions if needed).

**Expected gain.** 2–5% on small shapes (shape 0, 2); 0% on large shapes (already optimal).

**Risk class.** Low. Pure config change, no code logic changes. Numerics identical.

**Preconditions.** 
- Profiling to validate hypothesis: does `BM=64, NS=1` actually win on shape 0 vs current `BM=128, NS=3`? (Quick NVVP run.)

**Abort criteria.** Profiling shows no improvement on small shapes.

**Acceptance gate.** Geo-mean ≥ 0.5% speedup. Quick-test passes.

**Routes around ceiling.** Orthogonal (config-only optimization). Does not hit any ceilings.

---

### iter22 — TF32 promotion in critical paths (alternative to fp32)

**Hypothesis.** Some operations (e.g., B_g projection matmul) use fp32 for precision but sacrifice TC throughput. What if we use **TF32 (Tensor Float 32: 8-bit exponent, 10-bit mantissa, rounds to 32-bit)** at the TMA descriptor level? TF32 matmul is bf16-like speed (~65% peak) with fp32-like accuracy (exponent range). Curie tradeoff: marginal speed (fp32 is fast enough, ~60% TC), risk of missing atol=2e-2 gate on cauchy shapes.

**Method.**
1. Check if `matmul_kernel_tlx_ws` TMA descriptor supports TF32 input mode. (Likely not; TLX defaults to bf16 input per the wheel's design.)
2. If not, this iter is blocked (would require uTLX TMA descriptor extension).
3. If yes: measure `B_g` matmul (5-proj) under TF32 input vs bf16. Expect ~5–10% speedup (TC jumps from 65% to 75%).
4. Validate adversarial sweep on all 7 shapes (TF32 may lose precision on cauchy inputs vs bf16).

**Expected gain.** 3–5% on the 5-proj matmul if TF32 works; 2–3% e2e.

**Risk class.** Medium. TF32 is less precise than fp32, more precise than bf16. Leaderboard tolerance atol=2e-2 is tight; cauchy shapes already borderline (shapes 1, 4). Could push them over the edge.

**Preconditions.** 
- Check TLX TMA descriptor source code for TF32 support. If absent, skip.

**Abort criteria.** TF32 not supported in TLX. Or adversarial sweep on D=128 cauchy shapes shows > 2% failures.

**Acceptance gate.** Quick-test passes, especially shapes 1 and 4 (the precision-sensitive ones). Geo-mean ≥ 1%, no regressions.

**Routes around ceiling.** Orthogonal. Doesn't fuse or change kernel architecture.

---

### iter23 — Profile-driven autotune for D=128 matmul (narrow config sweep)

**Hypothesis.** iter11 swept 4 matmul tile configs and found no win. But it was on the full 6.0 ms baseline (iter10b). Current stack (iter15) has tighter margin; a **narrower, shape-specific sweep** might reveal that D=128 shapes actually prefer a different config. Focus on NUM_STAGES and GROUP_SIZE_M, not BM/BN (too risky for SMEM).

**Method.**
1. Profile only D=128 shapes (0, 1, 3, 4) on the current stack with 2–3 matmul configs:
   - Current: `NUM_STAGES=3, GROUP_SIZE_M=1`
   - Thin: `NUM_STAGES=1, GROUP_SIZE_M=2` (collapses stages, spreads M across more CTA clusters)
   - Balanced: `NUM_STAGES=2, GROUP_SIZE_M=2`
2. Measure matmul wall time in isolation (via CUDA events, not full e2e).
3. Pick the best per-shape and hardcode it into the dispatcher for D=128 (separate from D=384).

**Expected gain.** 2–3% on D=128 (the smaller shapes), 0% on D=384.

**Risk class.** Low. Pure config, no logic change. One shape-specific constant per dimension.

**Preconditions.** 
- Profiling infrastructure ready (CUDA events already used in benchmark harness).

**Abort criteria.** All configs tie within noise.

**Acceptance gate.** D=128 geo-mean ≥ 1% speedup, or full geo-mean ≥ 0.5%.

**Routes around ceiling.** Orthogonal.

---

## Parallel dispatch matrix (git worktrees)

| Iter | Deps | Conflicts | Est. time | Priority |
|---|---|---|---|---|
| **iter16** | none | none (profiling only) | 0.5 h | P0 |
| **iter17a** | iter16 (optional) | matmul_kernel (TLX) | 2 h | P1 |
| **iter17b** | none | fused_gate_ln_bmm_layout | 1 h | P2 |
| **iter19** | none | fused_invtr_ln_gate (D=128 path) | 1.5 h | P1 |
| **iter20** | none | final_linear (custom kernel) | 1.5 h | P1 |
| **iter21** | iter15 (bmm baseline) | bmm_kernel_tlx_ws | 0.5 h | P2 |
| **iter22** | none (if TF32 avail) | matmul_kernel | 0.75 h | P2 |
| **iter23** | iter15 (baseline config) | matmul_kernel | 1 h | P2 |

**Safe concurrent iters (no code conflict):**
- iter16 + iter19 + iter20 (profiling + two independent kernels)
- iter17b + iter21 (different kernels)
- iter22 + iter23 (both touch matmul config, but iter22 is a **precondition check** — if blocked, iter23 still runs)

**Sequential: iter17a depends on matmul + iter16 result** (needs to know if LN-prologue is even worth the SMEM cost). Run after iter16 completes.

**Recommended execution order:**
1. Immediate (parallel): iter16 (30 min, is-it-possible check) + iter19 + iter20 + iter21.
2. After iter16: iter17a if promising, else skip.
3. Overlap with above: iter17b (independent), iter22 (optional, depends on TF32 availability).
4. Late-stage refinement: iter23 (shape-specific autotune, only if geo-mean is stuck after top 4–5 iters).

## Stop conditions

**Ship (declare victory) when shape 6 hits one of:**
- **3.5 ms (3.0× baseline)** — massive win, submit immediately.
- **4.0 ms (2.5× baseline)** after 6+ iters stacked, with geo-mean ≥ 2.1×, adversarial sweep < 0.5% fails — ship with high confidence.
- **4.5 ms (2.2× baseline)** after 8 iters, if each iter adds < 0.3% regression — ship as incremental improvement.

**Abort (diminishing returns) if:**
- After iters 16, 19, 20 (P0/P1) complete with combined gain < 1% on shape 6 and < 0.5% geo-mean — the architecture is locally optimal; pivot to clean-up (comments, submission polish).
- Adversarial fail rate creeps above 2% on any single shape (precision margin collapsing) — revert, re-baseline, halt new features.
- Any iter stalls (SMEM math fails, codegen crashes, uTLX blocker) without a clear workaround within 2 h — shelve and move to next iter.

**Confidence thresholds:**
- < 1.5× baseline: do NOT submit (below bar).
- 1.5–1.8×: submit as a contender, iterate on H100 variants.
- 1.8–2.2×: strong candidate; de-risk precision via full adversarial sweep + GPUMode server test.
- ≥ 2.2×: submission-ready if all shapes pass and geo-mean holds.

---

## Summary: key insight from iter15

The **post-matmul cast elimination** pattern (iter15 removed bf16→fp32 cast by writing fp32 directly in the bmm epilogue) should be your north star. Every kernel you write should ask: "**Am I implicitly converting types in a separate pass that should be fused into the epilogue?**" The hidden wins are not in TC utilization or SMEM pressure — they are in **eliminating redundant type-conversion kernels**.

