# TriMul `hopper_gemm_ws.py` — Optimization Plan v2 (iter11-20)

Picks up from iter10 (shape 6 = 5.10 ms, 1.97× over baseline). Driven by the
NCU memory-movement analysis in [`reports/iter10_ncu_analysis.md`](reports/iter10_ncu_analysis.md).

## Where the time goes (iter10, shape 6)

| Bucket | ms | %wall | Class | What's left |
|---|---:|---:|---|---|
| TLX matmul (5-proj GEMM) | 1.20 | 24 | TC at 65 % peak | tile/stage tuning, epilogue fusion |
| ln_stats + bf16 cast | 1.09 | 21 | DRAM 91 % | only fusable into matmul prologue |
| fused_gate_ln | 0.95 | 19 | DRAM 92 % | only fusable into matmul epilogue |
| cuBLAS einsum bmm | 0.54 | 11 | DRAM 80 % | candidate for custom Triton kernel |
| cuBLAS final linear | 0.50 | 10 | TC 82 % | already efficient (D=384) |
| tr_fwd_pair | 0.50 | 10 | DRAM 87 % | layout-only (rewrite gate_ln to skip) |
| fused_invtr_ln_gate | 0.36 | 7 | DRAM 91 % | already fused for D=128 |
| **Pipeline total HBM** | — | — | 10.5 GB / 5.1 ms ≈ 2.06 TB/s avg | **shrink the byte budget** |

Five kernels are at 86–92 % HBM; one (matmul) is TC-leaning. The headline:
**every Triton stage is HBM-saturated — further wins must reduce total bytes,
not improve per-kernel efficiency.** The single biggest tensor in flight is
`proj [T, 5H]` (1.34 GB) — eliminating it via matmul-epilogue fusion is the
~14 % e2e prize.

## Priority groups

**Group A — easy wins (mechanical, 1–2 hrs each, low risk):**
- iter11: TLX matmul tile/stage sweep with proper SMEM math
- iter12: Cluster size=2 + TMA multicast for TLX matmul
- iter13: Fuse fused_invtr_ln_gate + final linear for D=384 (extend iter8 to wide-dim)

**Group B — layout fixes (3–6 hrs, medium risk):**
- iter14: 2D-tiled fused_gate_ln writes `L, R` directly in bmm-friendly layout (kills `tr_fwd_pair`); SMEM transpose to preserve coalescing
- iter15: Custom Triton einsum bmm (replaces cuBLAS bf16 bmm — bandwidth-bound at 80 %; persistent + L2-aware tile schedule)

**Group C — deep fusions (1–2 days, high payoff, high risk):**
- iter16: Fold gate-LN epilogue into TLX matmul (eliminates `proj [T, 5H]` 1.34 GB intermediate)
- iter17: Fold ln_stats + bf16 cast into TLX matmul prologue (eliminates `x bf16 [T, D]` 0.81 GB intermediate)
- iter18: Combined "all-in-one S1" — LN + 5-proj + gate-LN as a single TLX kernel (combination of iter16 + iter17)

**Group D — opportunistic (cleanup):**
- iter19: cudagraph-style replay of the static pipeline (reduces launch latency on small shapes)
- iter20: Workspace pre-allocation (skip `torch.empty` per call) — already partly done via the weight cache

## Iter-by-iter plan

### iter11 — TLX matmul tile/stage tuning

**Hypothesis:** 65 % Compute SoL means stalls in the WGMMA pipeline. Standard
Hopper bf16 GEMM hits 75–85 %; we have 0.2–0.3 ms of headroom (4–6 % e2e).

**Method:**
1. Profile with NCU `--section WarpStateStats` to identify dominant stall
   reason (Long Scoreboard, Barrier, Wait, etc.).
2. Sweep `(BM, BN, BK, NUM_STAGES, NUM_MMA_GROUPS)`:
   - Try `BK=64 NUM_STAGES=4` if SMEM allows (228 KB cap).
   - Try `BM=128 NUM_MMA_GROUPS=2` (smaller per-warpgroup acc, more N-reuse).
   - Verify `BM × BK × 2 × NUM_STAGES × NUM_MMA_GROUPS + BK × BN × 2 × NUM_STAGES + barriers ≤ 228 KB`.
3. For shape 6 (M=1M, N=640, K=384), 6 K-iters per output is steady-state-friendly.

**Risk:** low. Pure config sweep; numerics unchanged. May be a wash if stalls
are barrier-bound (not config-fixable).

### iter12 — Cluster size 2 + TMA multicast for TLX matmul

**Hypothesis:** Two CTAs in a cluster can share the B-side TMA load (the
`B_g [D, 5H]` weight tile) — halves L2 pressure on B. The TLX kernel already
has the `NUM_CTAS == 2` codepath wired up.

**Method:**
1. Set `TLX_CONFIG["NUM_CTAS"] = 2`.
2. Verify cluster grid math (`NUM_SMS / 2` clusters of 2).
3. If the multicast load doesn't align with our N-tile sizes, fall back to
   plain L2-resident broadcast.

**Expected:** 3–5 % on shape 6 (B-side reads modest already). Stronger on
larger shapes if N grows.

**Risk:** medium — cluster launch can fail on certain (M, N) shapes; need
divisibility checks. Possible patch issues with uTLX cluster shims.

### iter13 — Extend `fused_invtr_ln_gate_proj` to D=384

**Hypothesis:** iter8 fused this only for D=128 (where the [TI, dim] fp32
accumulator fits in registers). For D=384 we need to **stream W_out from SMEM**
in BD chunks while keeping `gated [TI, hd]` resident. Saves the 0.27 GB
intermediate write+read.

**Method:**
1. Allocate `gated_smem [TI, hd]` in SMEM (TI=64 → 16 KB).
2. Load `W_out` chunk [hd, BD=128] into SMEM.
3. Loop over 3 dim chunks; reuse SMEM gated, swap W_out chunk via TMA pipeline.
4. Save: 0.5 ms (the cuBLAS H→D linear) − cost of the manual GEMM tile.

**Risk:** medium. cuBLAS H→D is 82 % Compute (efficient). Beating cuBLAS
requires careful WGMMA scheduling; may end up tied or slightly worse.

### iter14 — 2D-tiled `fused_gate_ln` (kill `tr_fwd_pair`)

**Hypothesis:** Make `fused_gate_ln` write `L, R` in `[B, hd, N²]` layout
directly. iter7a failed via uncoalesced strided writes; the fix is **SMEM
transpose**: build [TI, hd] of L/R in SMEM, then have one warp per d-row
write 32 contiguous ij values. Saves 0.50 ms (`tr_fwd_pair` entire cost) plus
0.27 GB of `lf, rf` intermediate.

**Method:**
1. Restructure grid to `(B, cdiv(N², TI))`, each program processes TI rows
   of the original `proj [T, 5H]`.
2. Compute lv, rv, og into SMEM tiles `[TI, hd]`.
3. Cooperative store: warp `w` writes `L[b·hd + (w·32 .. w·32+32), ij]` — 32
   d-rows × TI ij-cols, with consecutive threads on consecutive ij offsets.
4. `og` keeps the original [T, hd] layout (consumed downstream as such).

**Expected:** 0.4–0.5 ms (8–10 % e2e). The `tr_fwd_pair` 0.50 ms is the
ceiling.

**Risk:** medium. Coalescing math is fragile but verifiable with NCU
`L2 Sector Promotion Misses` and `Memory Throughput Tbyte/s`.

### iter15 — Custom Triton einsum bmm

**Hypothesis:** cuBLAS bf16 bmm runs at 80 % DRAM, 29 % Compute on shape 6 —
memory-bound. cuBLAS chose `192×192×64` tiles giving only 132 output CTAs
(B·hd = 128 batches × small tile counts). A custom persistent kernel with
L2-aware grouping (each SM does multiple consecutive output tiles within the
same batch, keeping L tiles in L2) could cut bandwidth by reusing L across
multiple R-tiles.

**Method:**
1. Write `bmm_kernel_tlx` with WGMMA, persistent grid (NUM_SMS), L2 grouping.
2. Tile M=128, N=128, K=64 for each per-batch matmul.
3. Per SM, walk a strip of consecutive (i,j) tiles within one batch before
   moving to the next — maximizes L2 hits on `L`.

**Expected:** 0.1–0.2 ms (2–4 % e2e). cuBLAS is hard to beat.

**Risk:** medium-high. Easy to ship something slower than cuBLAS.

### iter16 — Fold gate-LN epilogue into TLX matmul

**Hypothesis:** The `proj [T, 5H]` intermediate (1.34 GB write+read = 2.68 GB
of HBM) is the largest unwasted byte stream. Computing the LN-affine + sigmoid
+ mask + gate-mul as a TLX warp-spec **consumer epilogue** lets us write only
`lf, rf, og [T, hd]` (3 × 0.27 GB = 0.81 GB) and skip `proj` entirely.

**Method:**
1. After the WGMMA accumulator finishes for an output tile, perform 5
   per-output-column blocks of LN-correction in registers:
   - For each of 5 hd-wide projections: `(rs * (acc - mu * s1) + s2)`
   - Apply sigmoid to projections 2/3/4
   - Multiply lv·lg·m, rv·rg·m
2. Write 3 separate TMA stores instead of one 5-wide store.
3. Pass `mean, rstd, mask, s1, s2` as additional buffers (small).

**Expected:** **0.5–0.7 ms (10–14 % e2e)** — the biggest single remaining lever.

**Risk:** high. Requires non-trivial TLX changes; epilogue will need its own
SMEM staging if 3 separate TMA stores can't issue in parallel. Numerics OK
(the math is identical).

### iter17 — Fold ln_stats into TLX matmul prologue

**Hypothesis:** `ln_stats_and_bf16_cast` reads x fp32 (1.6 GB), writes mean,
rstd, and bf16 x (0.81 GB). The matmul then re-reads bf16 x. Folding ln_stats
into the matmul prologue eliminates the bf16 x write+read (1.62 GB).

**Method:**
1. TLX prologue producer task reads fp32 x via TMA into SMEM.
2. A reduction warpgroup computes mean/rstd per row (small per-tile reduce).
3. The bf16-cast happens inside the matmul A-staging path.
4. mean/rstd written to a tiny scratch buffer for the epilogue (iter16).

**Expected:** 0.4–0.6 ms (8–12 % e2e).

**Risk:** very high. Combining a per-row reduction with the matmul producer
pipeline is a major TLX rework. Plausibly the most ambitious iter.

### iter18 — All-in-one S1 (iter16 + iter17 combined)

**Hypothesis:** Combined, eliminate both `bf16 x` and `proj` intermediates;
replace `ln_stats + matmul + fused_gate_ln` with one kernel that reads fp32
x once and writes `lf, rf, og` only. Total HBM: 1.6 GB read + 0.81 GB write.

**Expected:** matches iter16 + iter17 wins or slightly better (1.0–1.3 ms,
20–25 % e2e). Approaches the theoretical pipeline lower bound.

**Risk:** very high — only attempt after iter16 and iter17 succeed individually.

### iter19 — CUDA Graph capture

**Hypothesis:** Static pipeline of 7+ kernels has launch latency floor. Capture
once, replay each call. Saves ~50 µs/iter on small shapes (5–10 %).

**Method:**
1. First call records into `torch.cuda.CUDAGraph` after warmup.
2. Subsequent calls call `graph.replay()`.
3. Need stable input pointers — use a workspace buffer and copy input in.

**Risk:** low. Standard pattern. May not compose well with the weight cache
(graph captures specific tensor pointers).

### iter20 — Workspace pre-allocation

**Hypothesis:** `torch.empty(...)` per call adds CUDA caching-allocator
overhead. Pre-allocate `mean, rstd, lf, rf, og, L, R, out_bmm, out` as a
workspace tied to (id(x_in), shape).

**Expected:** <2 % on small shapes, ~0 % on large.

**Risk:** low. Need to handle workspace growing if shapes change.

## Stop conditions

- Reach within 10 % of theoretical lower bound (~3.5 ms on shape 6 = 1.45×
  better than iter10).
- 3 consecutive iters with < 1 % gain.
- iter16/17 failure with no clear path forward.

## Tracking

Append to [`PROGRESS.md`](PROGRESS.md) after each iter, same format as
iter1-10. Commit per iter. Council second-opinion at iter15 (mid-batch) and
iter20 (wave summary).
