# PLAN_v5 — TriMul H100 Optimization (post-PLAN_v4 reset)

**Status:** drafted 2026-05-08, supersedes [`PLAN_v4.md`](PLAN_v4.md).
**Baseline (unchanged):** geo-mean 1.87×, shape 6 = **5.36 ms**, adversarial 0.95% @ atol 2e-2.
**Honest target:** **ship current best.** Stretch goal limited to small wins inside known wheel constraints.

---

## 1. PLAN_v4 post-mortem — 6 iters, 6 ABORTs, 0% e2e change

| Iter | Goal | Result | Root cause class |
|---|---|---|---|
| iter21 | L2 cache hints | ABORT | uTLX C++ binding has no `EvictionPolicyAttr` slot; Python wrapper silently drops kwarg. |
| iter22 | bmm SMEM bank-conflict pad | ABORT | NCU misread (LD conflict actually 0%, not 28%). Real 56.8% conflict is on stores; uTLX `local_alloc(layout=)` is silently overwritten. |
| iter23 | Custom S3 final linear | ABORT | Standalone TLX 8% slower than cuBLAS. K=128 too shallow for WGMMA pipe; TMA forbids the asymmetric tile cuBLAS picks. |
| iter24 | D=128 deep fusion | ABORT | Serializing 5 N-chunks adds 3 µs/chunk pipeline-fill; HBM saved (~430 µs) << overhead added. +38–48% slower. |
| iter25 | D=384 decomposed 2+2+1 | ABORT | `tl.split` segfaults Triton + `local_slice` binding mismatch → forced 2-WGMMA-per-K-iter; +20–24% slower. Even with wheel fixes, ceiling is +3-4%. |
| iter26 | Row-persistent scheduler | ABORT | Matmul itself +50% slower; 132 SMs lockstep contend on same B columns. Round-robin already at local optimum. |

**What three of the failures share:** the **wheel-pybind/codegen surface**. Cumulative blocked features:
- `eviction_policy` / `cache_modifier` on TMA loads (iter21).
- `local_alloc(layout=)` user override (iter22).
- `tlx.local_slice` C++ `create_memdesc_subslice` binding (iter24, iter25).
- `tl.split` codegen on warp-spec acc layouts (iter25).
- Multi-warpgroup conditional-barrier patterns (iter18 Phase A).
- Cluster / TMA-multicast (iter12, C1).

**What three of the failures share (separately):** the **multi-WGMMA fusion hypothesis** (iter18, iter24, iter25). All three independently confirmed: serializing chunks to save HBM costs more in pipeline-fill than the bandwidth saved.

**What iter26 disconfirmed:** the **L2-reuse-via-scheduler-rewrite hypothesis** (PLAN_v4 §2 C5). Round-robin is at a local optimum.

## 2. Corrected mental model

**PLAN_v4's wrong assumptions:**
1. ❌ "NCU headroom converts to e2e wins." — wheel-blocked or misread.
2. ❌ "Custom TLX can beat cuBLAS at K=128 by saving launch hop / L2 evict." — cuBLAS uses asymmetric tiles TMA can't replicate; standalone loss > L2 win.
3. ❌ "Fusion across N-chunks saves HBM bandwidth." — pipeline-fill cost dominates.
4. ❌ "Row-persistent scheduling captures pair-row L2 reuse." — destroys SM-level staggering across pid_n.
5. ❌ "Wheel features described in `tlx` Python API will work in MLIR." — three independent iters proved many are dead at the C++ binding.

**PLAN_v5 ground truth:**
1. ✅ The current `hopper_gemm_ws.py` (5.36 ms shape 6, 1.87× geo) is at a **local optimum within current wheel constraints**.
2. ✅ The **only structural lever left inside this wheel** is small wins inside existing kernels — tile config, launch overhead, CPU-side prep. Expected gain: ≤5% combined.
3. ✅ The **highest-leverage move available** is a wheel rebuild, which would unlock ≥5 dead optimizations. Expected gain: 10–20% if executed cleanly.
4. ✅ Precision/throughput tradeoff (TF32) is independent of all wheel issues but cuts adversarial margin from 0.95% toward the 1.5% gate.

## 3. Three tracks

### Track A — Wheel rebuild (high leverage, high effort)

Rebuild uTLX 0.1.0+gitcba4ef9a from source against current Triton, fixing:
1. `create_async_tma_copy_global_to_local` — add `EvictionPolicyAttr` operand, plumb through Python wrapper. Unlocks iter21.
2. `local_alloc` — respect user-supplied `layout=`. Unlocks iter22-class SMEM padding.
3. `create_memdesc_subslice` — fix binding signature mismatch. Unlocks iter24/iter25 acc-layout slicing.
4. `tl.split` codegen on warp-spec acc layouts. Unlocks iter25-class single-WGMMA designs.
5. `tritonGetPluginInfo` only export — add Python init for cluster/multicast. Unlocks C1.

After rebuild, replay iter21 (cheap +5-8%), iter25-fixed (+3-4%), and any new fusion designs that depend on the unblocked features. **Best-case combined: +15-20% e2e on shape 6 → ~4.4 ms, hits the realistic-ship bar.**

**Cost:** 1-2 days (LLVM is the long pole; Triton+uTLX rebuild ~30 min if LLVM cached). Build process documented in `wheels/CLAUDE.md` and `wheels/utlx/build_utlx_wheel.sh`. Devserver network restrictions need a workaround for the GitHub-hosted Triton wheel URL.

**Risk:** the wheel rebuild itself may surface its own breakage; the unblocked features may have additional wheel-internal bugs. Realistic post-rebuild gain probably half the best case.

### Track B — Conservative wins inside current wheel

Three small, low-risk plays that don't touch the wheel-blocked surface:

#### B1 — Fold LN-correction into ln_stats_and_bf16_cast (iter18 follow-up #1)
- Modify `ln_stats_and_bf16_cast` to emit `bf16((x-mu)*rstd)` instead of `bf16(x)` and drop the `mu*s1` term from `fused_gate_ln_bmm_layout`. Saves ~5% of post-pass compute (the LN math). HBM unchanged.
- Risk: very low (single Triton kernel, no warp-spec, identical numerics).
- Estimated gain: **+1-3% e2e shape 6.**

#### B2 — Reduce CPU-side launch overhead
- Profile shows ~10 kernel launches per call (`ln_stats`, `tlx_ws_matmul_fixed`, `fused_gate_ln_bmm_layout`, `bmm_kernel_tlx_ws`, `fused_invtr_ln_gate`, `F.linear`, plus multiple PyTorch ops). On shape 0 (smallest, 0.575 ms) launches are ~10% of wall.
- Use `torch.cuda.graph()` capture for the **post-prep** kernel sequence only (skip `_prep_weights` which is cached anyway). Avoids iter19's "input copy dominates" failure mode by capturing only after the input tensors are placed.
- Risk: medium — graph re-capture on shape change; semantic mismatch with `_W_CACHE` if the cache misses inside the graph.
- Estimated gain: **+2-5% on small shapes (0, 2), 0-1% on shape 6.**

#### B3 — Tile-config sweep on `bmm_kernel_tlx_ws`
- iter11 swept matmul; bmm wasn't swept exhaustively. Try `BM=128/BN=64` (narrower N may improve L2 reuse), `NUM_STAGES=4` (more pipelining if SMEM permits), `replicate=4` consumers (more parallelism if registers permit).
- Risk: low — pure config tuning, T0/T1 gates each candidate.
- Estimated gain: **+0-3% e2e shape 6.** Could be zero (iter11 found matmul was at local optimum; bmm may be too).

**Track B combined best case: +5-10% e2e shape 6 → ~4.85-5.10 ms.** Doesn't hit realistic-ship bar (4.5 ms) but gets closer to it. **All three can be parallel-dispatched** — disjoint code regions.

### Track C — Ship `trimul-v4-hold`

Tag current HEAD as `trimul-v4-hold`, submit. shape 6 = 5.36 ms, geo-mean 1.87×, adversarial 0.95% — under the leaderboard tolerance gate. Re-evaluate in a week.

**Cost:** ~10 min (tag + submit via popcorn-cli per `AGENTS.md`).

**Recommendation:** Track C is the safe floor. If you want non-zero progress, **Track B in parallel** is the cheapest next shot — three iters with low risk and low blast radius. Track A is the right answer if there's appetite for a wheel rebuild day, but it crosses out of "kernel optimization" into "tooling work" — different category of effort.

## 4. Iter-by-iter (Track B only — Tracks A and C are out-of-band)

| Iter | Track | Hypothesis | Risk | Est. gain | Wall | Parallel? |
|---|---|---|---|---|---|---|
| iter29 | B1 | Fold LN-correction into ln_stats_and_bf16_cast | Very low | +1-3% | 1 hr | yes |
| iter30 | B2 | CUDA-graph the post-prep kernel sequence | Medium | +2-5% (small shapes) | 2 hr | yes |
| iter31 | B3 | bmm tile-config sweep | Low | +0-3% | 1 hr | yes |

Tier gates per PLAN_v4 §3 still apply (T0 after every recompile, T1 sentinel, T2/T3 only if numerics touched — only B1 touches numerics).

**Stop conditions:**
- Any iter that regresses any shape >2% on T1: revert immediately.
- After all 3 iters, if combined gain <3%: ship `trimul-v4-hold` and stop.
- After all 3 iters, if combined gain ≥5%: ship as `trimul-v5`.

## 5. Out of scope for PLAN_v5

The PLAN_v4 "do not retry" list is inherited verbatim. Specifically NOT in PLAN_v5:
- Any multi-WGMMA-in-one-kernel fusion (iter18, iter24, iter25 each disconfirmed independently).
- Any scheduler rewrite of `matmul_kernel_tlx_ws` (iter26 disconfirmed).
- Standalone TLX GEMMs at K=128 (iter23 disconfirmed).
- Cache-residency / SMEM-padding wheel-API hacks (iter21, iter22 disconfirmed at wheel layer).
- iter27 (LN-prologue fold) — was conditional on iter26 working AND has the same multi-warpgroup design as iter18 (which crashed MLIR). Dead-on-arrival.
- iter28 (TF32) — H100 TF32 throughput is half bf16; matmul itself slows. Adversarial budget already at 0.95% / 1.5% gate. Risk/reward unfavorable.

## Files referenced

- `/home/wychi/oss/wheels/gpumode/bioml/trimul/work/hopper_gemm_ws.py` — current ship candidate.
- `/home/wychi/oss/wheels/gpumode/bioml/trimul/work/optimize/PROGRESS.md` — full iter1–26 log.
- `/home/wychi/oss/wheels/gpumode/bioml/trimul/work/optimize/PLAN_v4.md` — superseded; preserved with PLAN_v4 RESULT footer.
- `/home/wychi/oss/wheels/CLAUDE.md` + `wheels/utlx/build_utlx_wheel.sh` — Track A wheel-rebuild build process.
- `/home/wychi/oss/wheels/gpumode/bioml/trimul/AGENTS.md` — popcorn-cli submission workflow for Track C.
