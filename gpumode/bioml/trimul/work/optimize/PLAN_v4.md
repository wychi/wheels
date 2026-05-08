# PLAN_v4 — TriMul H100 Optimization

**Status:** drafted 2026-05-08, supersedes [`PLAN_v3.md`](PLAN_v3.md).
**Baseline:** 1.81× geo-mean, shape 6 = 5.55 ms, adversarial fail = 0.95% @ atol 2e-2.
**Stretch (P=10%):** shape 6 ≤ 4.0 ms, geo ≥ 2.4×.
**Realistic ship (P=60%):** shape 6 ≤ 4.5 ms, geo ≥ 2.1×.

Synthesized from a 4-agent council (2 Claude opus + 1 Codex GPT-5.2 + 1 Gemini 3 Pro, 2 layers, opus evaluator).

---

## 1. Where we stand

- **Floor:** PyTorch reference geo-mean = 1.0×. Current best (`work/hopper_gemm_ws.py` + iter13 + iter15-einsum) = **1.81×** geo-mean, **5.55 ms** on shape 6 (B=1, S=1024, D=384), adversarial fail rate **0.95%** (under the 1.5% gate, but only just).
- **Wall-time decomposition (NCU, shape 6):** matmul_kernel_tlx_ws 22%, fused_gate_ln_bmm_layout 22%, ln_stats_and_bf16_cast 20%, bmm_kernel_tlx_ws 14%, fused_invtr_ln_gate (D=384) 11%, cuBLAS bf16 final linear 10%. Five kernels each ≥10% — the perf surface is flat, no single dominant cost.
- **NCU headroom unconverted into wins:**
  - **L2 compression: 0% across all kernels.** Every read of operands and intermediates goes through DRAM. NCU estimates **10–15% e2e** if compression hit-rate moves to 30–40% on read-only operands.
  - **bmm SMEM bank conflict: 28% local** (5.4-way conflict on stores from `tlx.local_trans`). Estimated **~1% e2e** at ~50 µs.
  - **cuBLAS bf16 final linear (D=384)** still goes through cuBLAS — a custom Triton bf16-in/fp32-acc kernel can recover ~5-7% e2e by removing launch hop and L2 evict.
- **What did NOT work in PLAN_v3 (iters 16–20):** monolithic S1 (`iter18`, register + SMEM cliff), CUDA-graph capture (`iter19`, 1.6 GB input copy dominates), workspace pre-alloc (`iter20`, no win — CUDA caching allocator already optimal), cluster TMA-multicast (`iter12`, wheel pybind hole), gate-LN epilogue fold full (`iter16`, D=384 SMEM cliff), ln_stats prologue fold (`iter17`, scheduler redundancy + SMEM blocked).

---

## 2. Hard ceilings to route around

| ID | Ceiling | Demonstrated by | Route-around |
|----|---------|-----------------|--------------|
| **C1** | uTLX wheel pybind hole — `libutlx.so` only exports `tritonGetPluginInfo`; cluster/multicast Python API dead at runtime | iter12 | Do not call `tlx.cluster_*` / `tlx.tma_multicast`; treat as unavailable until wheel rebuild lands. |
| **C2** | MLIR crash on conditional barriers under `tl.if` inside multi-task warpgroups | iter18 Phase A | Keep barriers at top level of the kernel body; gate work via tile-shape predicates only. No `if pid_n == last: arrive`. |
| **C3** | Consumer-WG register file ~64 KB — exceeding it spills onto stack and serializes WGMMA. iter18 Phase B died at 5 fp32 accs of [32,128] = 80 KB | iter18 Phase B | Cap consumer regs ≤ 232/thread; max 2 fp32 accumulators per consumer warpgroup; offload extras to SMEM staging. |
| **C4** | Per-CTA SMEM cap 232,448 bytes — fully-fused S1 needs ~248 KB at D=384. iter16 hit the same wall: NS_A=6 needed for D=384 K_ITERS=6, falls back to NS_A=2 with 5× A-bandwidth penalty | iter16, iter18 | Decompose into 2+2+1 sub-fusions when D ≥ 256; full fusion only at D ≤ 128 where K_ITERS=2 fits. |
| **C5** | Round-robin scheduler `tile_id += NUM_SMS` with `GROUP_SIZE_M=1` creates redundant prologue reads (5× fp32-x on shape 6, num_pid_n=5) | iter17 static analysis | Switch to **row-persistent** schedule (one CTA owns a strip of `i`, sweeps all `j`) before any further fusion attempt. iter26 is the enabler. |

**Bonus precision ceiling — C6: bf16 cascade rounding** (see `memory/trimul_bf16_cascade.md`). `tl.store(fp32_buf, x.to(bf16))` silently rounds. Match reference's bf16 boundaries exactly; do not add new bf16 down-casts. Cauchy D=128 shapes are the sensitivity floor.

---

## 3. Tiered precision test methodology

PLAN_v3 wasted ~30% of dev wall on full-sweep waits to disprove obviously-broken kernels. T0/T1 cut that.

| Tier | Shapes | Seeds | Trials | Wall | Use |
|------|--------|-------|--------|------|-----|
| **T0** | shape 4 only (smallest cauchy D=128) | seed 731 | 1 | ~5 s | Compile + correctness smoke. After every recompile. |
| **T1** | {1, 4, 6} | {731, 17} | 5 | ~30 s | Per-iter speed sanity + adversarial sentinel. Catches >5% regressions on the 3 shapes that span (D=128 small / cauchy / D=384 largest). |
| **T2** | {0, 1, 4, 6} | 3 seeds | 8 | ~2 min | Mid-iter precision check. Required gate before merging any iter that touches numerics. |
| **T3** | All 7 shapes | 6 seeds | 30 trials | ~10 min | Final adversarial + perf validation. Required for ship. |

**Merge rule (NUMERIC, NOT VIBES):** no iter merges to `work/hopper_gemm_ws.py` without:
- T1 green AND
- if iter touches precision: T2 ≤ 1.0% AND T3 ≤ 1.5% adversarial fail.

Shape 4 (cauchy, S=1024, D=128) is the canary — historically the worst max_err (0.060+ on iter14 sweep). Always in T1.

---

## 4. Iter-by-iter plan (Waves)

Numbering continues at **iter21** to avoid clashing with PLAN_v3's iter16–20 historical record in `PROGRESS.md`.

### Wave 1 — NCU-headroom harvests (parallel, low risk)

#### iter21 — L2 cache-residency hints on read-only operands
| Field | Value |
|---|---|
| Hypothesis & gain | NCU shows L2 compression 0%; tagging `pair`, `mask`, weight TMA descriptors with `EvictLast` + cache hint should lift hit-rate to 30–40%. **+5–8% e2e.** |
| Approach | `tl.experimental_descriptor_load(..., cache_modifier=".ca")` on read-only TMA loads; switch CTA scheduler to a column-persistent strip so the same `pair` rows feed back-to-back CTAs. Allocator: try `cudaMallocAsync` mempool with `CU_MEM_ALLOCATION_COMP_GENERIC`. |
| Risk | Low. No numerics change. Worst case L2 thrash on small shapes. |
| Preconditions | None. |
| Abort | T1 shows ≥2% regression on any of {1, 4, 6}, OR `lts__t_sectors_op_write_compressed` still 0% after wiring. |
| Acceptance | T1 +3% on shape 6, T3 adversarial unchanged. |
| Ceiling routed | (precondition for) C5 |

#### iter22 — bmm SMEM bank-conflict pad
| Field | Value |
|---|---|
| Hypothesis & gain | NCU local 28% bank conflict on bmm B-fragment loads from `tlx.local_trans`. Pad LDS stride by 8 bf16 lanes OR store B as `[BK, BN]` directly via TMA swizzle. **+1.5% e2e (~50 µs).** |
| Approach | Add `pad_K=8` to bmm SMEM allocator OR use NVIDIA 128B swizzle on TMA descriptor for transposed load; verify NCU `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` drops below 5%. |
| Risk | Low. Adds 4 KB SMEM — within budget. |
| Preconditions | None. |
| Abort | SMEM occupancy drops below 2 CTAs/SM, OR bank-conflict counter doesn't drop ≥80%. |
| Acceptance | NCU bank-conflict ≤ 5% on bmm; T1 not slower; ≥1% e2e shape 6. |
| Ceiling routed | none (pure local fix) |

#### iter23 — Custom Triton final linear (replace cuBLAS bf16 in S3, D=384 only)
| Field | Value |
|---|---|
| Hypothesis & gain | cuBLAS bf16 final linear on shape 6 = 0.55 ms (10% wall), runs sequentially after `fused_invtr_ln_gate` with no L2 reuse of `gated`. A custom TLX warp-spec GEMM reading `gated` straight from L2 saves launch hop + L2 evict. **+3–6% on shape 6.** |
| Approach | Persistent TLX bf16-in/bf16-out GEMM, 3D over `[B, N², hd]→[B, N², D]`, BM=128, BN=128, BK=64, NUM_STAGES=3, MG=2, replicate=2. Same template as iter15-einsum bmm. Skip when `dim == hd` (iter13's fused path covers D=128). |
| Risk | Medium — cuBLAS hard to beat on K=128; win comes from L2-locality + launch saving. |
| Preconditions | None. |
| Abort | T2 adversarial > 1.0%, OR standalone within 5% of cuBLAS — gain eaten by launch noise. |
| Acceptance | T3 adversarial ≤ 1.3%, T1 ≥ 2% e2e on shape 6. |
| Ceiling routed | C6 (bf16 cascade) by keeping accumulator fp32 |

### Wave 2 — Architectural forks (parallel after W1)

These can run concurrently because they target **different shape buckets** (D=128 vs D=384) and ship as separate kernel entry points the dispatcher selects on input shape.

#### iter24 — D=128-only deep fusion (resurrect iter16 gate-LN epilogue, D=128-only dispatch)
| Field | Value |
|---|---|
| Hypothesis & gain | iter16 (epilogue fold) won 5–10% on D=128 but cliffed +13–15% on D=384 (C4). At D=128 the SMEM cap is not binding (K_ITERS=2, NS_A=4 fits). Removes the `proj [T, 5H]` 1.34 GB intermediate. **+5–10% on the 4 D=128 shapes (0,1,3,4) ≈ +3% geo.** |
| Approach | Re-instate iter16's epilogue-fused matmul kernel verbatim, gate dispatch on `dim == 128` (analogous to iter13's pattern). |
| Risk | Medium — already known to work numerically; mainly need to confirm dispatch boundary doesn't break D=384. |
| Preconditions | iter22 (bank-conflict pad freed up SMEM headroom). |
| Abort | D=128 adversarial regresses past 1.5%, OR consumer reg > 232. |
| Acceptance | T3 D=128 shapes ≥ 1.05× current, D=384 unchanged. |
| Ceiling routed | C4 (avoid cap by restricting to D=128); C3 (only 2 accs needed at D=128 BM size) |

#### iter25 — Decomposed 2+2+1 fusion @ D=384 (BM=64)
| Field | Value |
|---|---|
| Hypothesis & gain | iter18 died at 5-acc / 80 KB register cliff. Two independent matmul kernels — `L=lv·lg·m` and `R=rv·rg·m` — each only need 2 fp32 accs `[BM_split, hd]` ≈ 32 KB; fits the register file. Eliminates `proj [T, 5H]` (1.34 GB write+read = 2.68 GB HBM). **+8% on shape 6.** |
| Approach | New `matmul_L_kernel` and `matmul_R_kernel`, each takes a 2-projection slice of B_g (`[D, 2H]`), runs WGMMA, in epilogue does `sigmoid(gate) × value × mask`, writes directly into bmm-layout `[B*hd, N²]`. Reuse iter14's 2D-tiled write. og: trivial fused_gate_ln on the single remaining projection. Three persistent kernels chained via TMA store→load on a single workspace. |
| Risk | Medium-high — new bf16 cast in the matmul epilogue (sigmoid runs on fp32 acc, then bf16 store); B-bandwidth grows ~1.6×; needs full sweep. |
| Preconditions | iter21 cache-residency landed (workspace must be L2-resident across the chain). |
| Abort | Workspace alloc cache miss-rate > 5% over a 30-trial run, OR combined NCU DRAM TB/s on the new pair > current matmul + fused_gate_ln_bmm_layout sum. |
| Acceptance | T1 ≥ 5% improvement on shape 6, T3 adversarial ≤ 1.0%. |
| Ceiling routed | C3 (only 2 accs, not 5), C2 (uniform schedule, no conditional barriers), C4 (slimmer staging) |

### Wave 3 — Schedule rewrite (serial; both rewrite the matmul body)

#### iter26 — Row-persistent CTA scheduler
| Field | Value |
|---|---|
| Hypothesis & gain | C5 — round-robin re-fetches `pair` rows. Row-persistent reuses them in L2 across all `j` strips. **+4–6% e2e direct, AND enables iter27.** |
| Approach | Replace outer loop in `matmul_kernel_tlx_ws` with `for m_chunk in row_assignment[sm_id]: for pid_n in range(num_pid_n)`. Row assignment: `cdiv(num_pid_m, NUM_SMS)` chunks per SM. One TMA descriptor for `pair` row, reused. |
| Risk | Medium — schedule change; load balance for `num_pid_m % NUM_SMS != 0` shapes. |
| Preconditions | iter21, iter22 merged. Shape 6: num_pid_m = 8192, divisible by 132. Small shapes need a fall-through to round-robin. |
| Abort | T1 regressions on any shape > 2% (poor occupancy). |
| Acceptance | ≤1% regression on any shape (this is purely an enabler), NCU L2 hit-rate +20pp on matmul. |
| Ceiling routed | C5 |

#### iter27 — LN-stats prologue fold into iter26
| Field | Value |
|---|---|
| Hypothesis & gain | Once row-persistent, LN mean/rstd for the `i` strip can be computed once and held in SMEM across all `j`. Eliminates ln_stats_and_bf16_cast (1.10 ms, 20% of wall). **+6–10% e2e on shape 6.** |
| Approach | Producer warpgroup loads fp32 x once per `pid_m`; reduction warpgroup computes mean/rstd and bf16-normalizes into staging buffer; 2 consumer warpgroups consume bf16-staging for WGMMA. Phase tracking on `pid_m` advance only — no conditional barriers (uses iter26's deterministic schedule to know exactly when to arrive). |
| Risk | High — multi-warpgroup design, near uTLX codegen limits. SMEM math: fp32 x scratch [BM, D] for D=384 BM=128 = 192 KB ALONE exceeds cap. **Must use BM=64 for D=384** (96 KB) plus B/bf16 stages. |
| Preconditions | iter26 merged. Re-verify SMEM math before coding. |
| Abort | T2 adversarial > 1.0%, OR SMEM math doesn't fit BM≥64 for D=384, OR codegen crash recurs (1 day cap; fall back to pure iter26 + iter21). |
| Acceptance | T3 +4% e2e shape 6 vs iter26 baseline, T3 adversarial ≤ 1.5%. |
| Ceiling routed | C5 amortization (via iter26), C2 (no conditional barriers) |

### Wave 4 — Hail mary (solo, only if W1+W2+W3 < ship bar)

#### iter28 — TF32 promotion in bmm einsum (or matmul A operand)
| Field | Value |
|---|---|
| Hypothesis & gain | bmm + matmul together = 36% of wall in bf16. TF32 (~10b mantissa vs bf16 7b) may let us drop iter10b's fp32 promotions and re-enable bf16-output paths. **+5–10% e2e** if adversarial budget survives. |
| Approach | Swap `tl.dot(..., input_precision="tf32")` on bmm OR matmul A. Keep fp32 accumulator. |
| Risk | **High** — H100 TF32 throughput is half bf16; matmul itself may slow. Cauchy shapes already at 0.95% adversarial; precision math: TF32 mantissa = 10 bits is MORE than bf16's 7, but TF32 inputs cost 2× SMEM. |
| Preconditions | iter23 has reclaimed adversarial headroom (≤ 1.3%). NCU SoL on a tf32 prototype. |
| Abort | Matmul slows >5% standalone, OR T2 adversarial > 1.0%. |
| Acceptance | T3 adversarial ≤ 1.5%, T3 geo +0.2×. |
| Ceiling routed | none — pure precision/throughput bet |

---

## 5. Parallel dispatch matrix

| Wave | Iters | Parallel? | Why / Conflict |
|---|---|---|---|
| W1 | iter21, iter22, iter23 | **Yes — 3 worktrees** | Disjoint code regions: TMA descriptor flags / SMEM allocator / S3 kernel body. |
| W2 | iter24, iter25 | **Yes — 2 worktrees** | Orthogonal shape buckets (D=128 vs D=384), separate kernel entry points, dispatcher merges trivially. |
| W3 | iter26 → iter27 | **Serial** | Both rewrite the same persistent-kernel body; iter27 layered on iter26. |
| W4 | iter28 | **Solo** | Conditional on W1–W3 outcome. |

**Cross-wave conflict region:** the matmul kernel body is touched by iter23, iter24, iter25, iter26, iter27. Sequence is enforced by waves; merges into `work/hopper_gemm_ws.py` happen one wave at a time, in order.

**"Do not retry" list (codified):**
- Monolithic S1 fusion at D=384 (PLAN_v3 iter18 — register + SMEM cliff).
- Cluster / TMA-multicast paths (PLAN_v3 iter12 — wheel pybind hole).
- `tlx.barrier` under `tl.if` inside multi-task warpgroups (PLAN_v3 iter18 Phase A — MLIR crash).
- CUDA Graph capture (PLAN_v3 iter19 — input-copy dominates for fresh-tensor calling convention).
- Workspace pre-alloc as standalone iter (PLAN_v3 iter20 — CUDA caching allocator already optimal).
- fp8 anywhere on cauchy shapes (precision budget too thin).

---

## 6. Stop conditions

| Outcome | Probability | Trigger | Action |
|---|---|---|---|
| **Stretch ship** | ~10% | Shape 6 ≤ 4.0 ms AND geo ≥ 2.4× AND T3 ≤ 1.5% | Tag `trimul-v4-stretch`, submit. |
| **Realistic ship** | ~60% | Shape 6 ≤ 4.5 ms AND geo ≥ 2.1× AND T3 ≤ 1.5% | Tag `trimul-v4`, submit. |
| **Hold** | ~25% | After W1+W2+W3, shape 6 still ≥ 5.2 ms OR geo < 2.0× | Run iter28 (TF32). If still no win, ship current best as `trimul-v4-hold`. |
| **Plan abort** | ~5% | After all 8 iters, geo < 1.85× OR T3 adversarial > 1.5% | Escalate: rebuild uTLX wheel to unlock C1 (cluster/multicast) OR pivot to weights-side optimization. New PLAN_v5. |

**Per-iter abort:** T2 adversarial > 1.5% on any shape — abort that iter immediately, do not chase tuning. Reverted iters do not block subsequent waves.

**Wall-clock budget:** 5 working days. Day-end check-in on the realistic-ship bar; if W1 alone hasn't moved geo by ≥ 0.05×, re-evaluate W2 ordering.

---

## Files referenced

- `/home/wychi/oss/wheels/gpumode/bioml/trimul/work/hopper_gemm_ws.py` — current kernel stack (iter13 + iter14 + iter15-einsum)
- `/home/wychi/oss/wheels/gpumode/bioml/trimul/work/optimize/PROGRESS.md` — iter-by-iter log including PLAN_v3 iter16–20 failure post-mortems
- `/home/wychi/oss/wheels/gpumode/bioml/trimul/work/optimize/check_leaderboard_seeds.py` — adversarial-sweep harness (will need a CLI shim for tiered T0/T1/T2/T3 modes)
- `/home/wychi/oss/wheels/gpumode/bioml/trimul/work/optimize/profile/iter15_stack/` — NCU `--set full` reports for matmul, bmm, fused_gate_ln_bmm_layout
- `/home/wychi/oss/wheels/gpumode/bioml/trimul/work/optimize/reports/baseline_vs_iter15_stack.md` — per-kernel cost comparison vs baseline
