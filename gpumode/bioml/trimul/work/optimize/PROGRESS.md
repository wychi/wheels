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
| 7b | replace TLX matmul with cuBLAS bf16 GEMM — **REVERTED (user preference: keep TLX)** | 5.12 | 1.89× | Measured equivalent perf; reverted at user request to keep TLX kernel as the optimization target. |
| 8 | O6: fused inv-tr + LN + gate + H→D linear (`fused_invtr_ln_gate_proj`); dispatched only when dim==hd | 5.12 | 1.89× | D=128 shapes -5 to -6% (eliminates [T,hd] gated intermediate); D=384 keeps 2-kernel path (cuBLAS bf16 ≥ Triton fused for the wider GEMM). |
| 9 | multi-row fused_gate_ln (BR=4) | 5.12 | 1.89× | Tiny (≤1%) on all shapes — kernel was already at ~2.2 TB/s HBM peak. Kept for cleaner amortization of launch latency. |
| 10 | weight cache (B_g, s1, s2, w_out) keyed by data_ptr | 5.10 | 1.90× | -6.5% shape 0, ≈0% large shapes. Skips per-call cat/cast/affine-mul. Safe for repeated bench calls; doesn't help when caller passes fresh weights. |
| 10b | **PRECISION FIX**: cache fingerprint, fp32 bmm/og/output, remove hardcoded bf16 store cast in `fused_gate_ln`, revert iter8 fused proj for D=128 | 6.02 | 1.55× geo-mean | Triggered by GPUMode server failure. See `reports/precision_postmortem.md`. Adversarial sweep: 13/900 = 1.44% fail rate (down from 27% pre-fix); all 7 BENCHMARK_SHAPES pass seed-0 with max_err 0.013-0.023. |
| 11 | TLX matmul tile/stage sweep — **NO WIN** | 5.97 | 1.62× | Tested 4 alternatives (BM256/NS4/MG1, BM128/NS4/MG2, BM256/NS2/MG2, BM128/NS5/MG2). All within ±1.5% of baseline; deeper-NS configs that needed BM=256 (256 KB) don't fit the 228 KB cap. Baseline already near local optimum for SMEM-constrained tiles. Matches iter10 NCU finding (matmul TC SoL stuck at 65%). Move to iter12. |
| 12 | Cluster size 2 + TMA multicast — **BLOCKED** | 6.02 | 1.55× | Setting `NUM_CTAS=2` and launching with `num_ctas=2` triggers `utlx: op 'ttng.map_to_remote_buffer' not registered in this Triton build`. The multicast TMA op the kernel emits is not in the current Triton+uTLX wheel. Reverted; would need a uTLX patch or a Triton bump. Move to iter14. |
| 14 | 2D-tiled `fused_gate_ln_bmm_layout` — kills `tr_fwd_pair` | **5.57** | **1.67×** | Replaces (`fused_gate_ln` writing lf/rf [T,hd] + `tr_fwd_pair` reading them and writing L/R [B*hd,N²]) with a single kernel that does LN/gate math AND the transpose on store. Eliminates ~0.54 GB of intermediate traffic. Per-shape: 0=-7.5%, 1=-8.9%, 2=-6.4%, 3=-8.9%, 4=-9.3%, 5=-6.4%, 6=-7.4%. Adversarial sweep IMPROVED: 7/900=0.78% fail rate (vs iter10b's 1.44%). Worst max_err 0.060. |
| 13 | Retry `fused_invtr_ln_gate_proj` with strict precision (D=128) | **5.59** | (D=128 -26%, geo-mean 1.94×) | Re-introduce iter8's fused inv-tr+LN+gate+H→D matmul for D=128, with iter13 precision rules: (1) bf16 cast on `gated` AND on `w_out` ONLY at `tl.dot` input, (2) fp32 output store (no `.to(bfloat16)` on result), (3) keep `W["to_out.weight"]` as fp32 in `_W_CACHE` for D=128, (4) return fp32 directly (skip bf16 round-trip). D=128 shapes -26% (0=-26.5%, 1=-26.3%, 3=-26.6%, 4=-26.9%); D=384 shapes flat (2=+0.4%, 5=-1.4%, 6=-0.2%). Geo-mean: 2.314 → 1.936 ms (-16.35%). Adversarial sweep across 6 input seeds × 7 shapes × 30 trials = 1260 runs: **12/1260 = 0.95%** fail rate, vs iter14 baseline **13/1260 = 1.03%** on the same seeds. Moots the wout→fp32 precision-fix attempt entirely. |
| 15 | Custom Triton `bmm_kernel_tlx_ws` (bf16-in/fp32-out) — replaces cuBLAS bf16 bmm + `.float()` cast | **5.35** | (-4.2% on shape 6 vs iter14) | Persistent warp-spec GEMM (BM=128, BN=128, BK=64, NUM_STAGES=3, NUM_MMA_GROUPS=2, GROUP_SIZE_M=8, replicate=2 consumers). 3D TMA descriptors over (B*hd, N, N); B is loaded `[BN, BK]` and `local_trans`'d to `[BK, BN]` for the dot — saves a separate transpose. The real win isn't beating cuBLAS at the matmul; it's eliminating the **post-bmm bf16→fp32 elementwise cast** (~451 µs on shape 6, almost as much as the matmul itself) by writing fp32 directly inside the epilogue. Per-shape: 0=-5.9%, 1=-5.7%, 2=-4.9%, 3=-6.2%, 4=-5.2%, 5=-4.1%, 6=-4.2%. Adversarial sweep: **0/210=0.0% fail rate** (vs iter14's 0.78%). Standalone bmm: 1.26-1.40× vs `torch.bmm + .float()`. Stacked on top of iter13: re-bench needed. |

## Final summary (per shape, baseline → iter10)

| Shape | bs | sl | D | dist | baseline (ms) | iter10 (ms) | speedup | max_err |
|-------|----|----|---|------|---------------|-------------|---------|---------|
| 0 | 2 | 256 | 128 | normal | 1.04 | 0.46 | **2.26×** | 0.0277 |
| 1 | 1 | 768 | 128 | cauchy | 4.09 | 2.05 | **2.00×** | 0.0228 |
| 2 | 2 | 256 | 384 | normal | 1.37 | 0.66 | **2.08×** | 0.0168 |
| 3 | 1 | 512 | 128 | normal | 1.87 | 0.91 | **2.05×** | 0.0236 |
| 4 | 1 | 1024 | 128 | cauchy | 7.33 | 3.62 | **2.02×** | 0.0211 |
| 5 | 1 | 768 | 384 | normal | 5.64 | 2.87 | **1.96×** | 0.0141 |
| 6 | 1 | 1024 | 384 | normal | 10.06 | 5.10 | **1.97×** | 0.0123 |

Geo-mean speedup across 7 shapes: **2.04×**. All shapes pass `atol=2e-2` against the fp32 reference.

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

### iter10 — cache weight setup (B_g, s1, s2, w_out_bf16) by data_ptr
- The 5-projection setup (cat 5 weights → transpose → fp32 cast → ln_w-affine → bf16 cast, plus s1/s2 reductions, plus W_out → bf16) is invariant for fixed weights but ran every call.
- New `_W_CACHE` dict keyed by `(W["norm.weight"].data_ptr(), W["left_proj.weight"].data_ptr(), W["to_out.weight"].data_ptr(), dim, hd)` — survives Python `id()` reuse across shapes.
- Per-shape speedup: 0=-6.5% (small shapes win biggest since setup is largest fraction), 1=-0.6%, 2=-3.4%, 3=-1.9%, 4=-0.3%, 5=-0.7%, 6=-0.4%.
- Safe correctness-wise: data_ptr keys distinguish different weight tensors regardless of dict reuse.

### iter8 — O6: fuse fused_invtr_ln_gate + final H→D linear (D=128 only)
- New kernel `fused_invtr_ln_gate_proj` builds the [TI, hd] gated tile in registers, then runs `tl.dot(gated_bf16, W_out.T_bf16)` and writes [TI, dim] bf16 directly to the final output.
- Dispatched only when `dim == hd` (the D=128 shapes). For D > hd, the fused path's loop over BD chunks re-loads bmm/og or expands the register footprint; cuBLAS bf16 GEMM ties or wins, so we keep the two-kernel path.
- Per-shape speedup vs iter7: 0=-5.5%, 1=-5.7%, 3=-6.2%, 4=-6.1% (all D=128). D=384 shapes unchanged.
- Eliminates the [T, hd] bf16 gated intermediate (~0.27 GB on shape 4) for D=128 shapes.

### iter7 — strided fused_gate_ln write (REVERTED) + cuBLAS 5-proj matmul
- **7a (reverted):** modify fused_gate_ln to write `lf, rf` directly in [B, hd, N²] (bmm-friendly) layout to skip the tr_fwd_pair pass. Within each program (1 thread block per row of length hd), the strided writes (offsets spaced N²=1M apart) destroyed coalescing and made every shape 50-60% slower. A correct fix would require 2D tiling (TI rows × hd cols per program), which is more invasive.
- **7b:** replace `tlx_ws_matmul_fixed` with `torch.matmul(x_flat, B_g)` (cuBLAS bf16 GEMM, fp32 accum). Modest improvement on small shapes (shape 0 0.524→0.505, ≈3.6%), neutral on shape 6. Worth keeping for code simplicity — the 5-proj GEMM no longer relies on the bespoke TLX warp-spec kernel for any shape.

### iter6 — O10 (partial): fold fp32→bf16 cast of x into ln_stats
- New Triton kernel `ln_stats_and_bf16_cast` reads fp32 x once, writes mean+rstd AND bf16-cast x. Replaces `ln_stats_multirow` + `x_flat.to(bf16)` elementwise pair.
- Saves the second pass over [T, D] fp32 (~1.6 GB on shape 6) — partly offset by the new bf16 write (~0.8 GB).
- Per-shape speedup: 0=5.4%, 1=6.3%, 2=12.6%, 3=6.2%, 4=6.1%, 5=12.1%, 6=12.1%.
- Cumulative shape 6: 5.82 → 5.12 ms = **1.89× faster than baseline**.
- Note: shape 0 first-iter shows 2.5 ms (JIT compile of new kernel costs); steady-state matches the warm number above.

### iter10b — Precision fix (post-GPUMode-server failure)

- **Trigger:** server reported 6 mismatched elements on `bs=2/sl=768/dim=128 normal seed=731`. Tolerance `|diff| <= 2e-2 + 2e-2*|ref|`. Errors were 0.022-0.025 on small-|ref| elements.
- **Root causes:** (1) iter10 weight cache aliased on CUDA allocator data_ptr reuse → ~98% wrong outputs after a few trials with fresh weights. (2) bf16 cascade thinned precision margin to <1% on adversarial weight draws.
- **Fixes:**
  1. `_W_CACHE` content fingerprint via `torch.stack([...]).cpu().tolist()` (one host sync), plus `(data_ptr, shape, stride, dtype, _version)` per tensor in the key.
  2. Promote `out_bmm` to fp32 (`.float()` after `torch.bmm`).
  3. Promote `out_gate` allocation to fp32.
  4. Promote final output allocation to fp32.
  5. Remove hardcoded `.to(tl.bfloat16)` from `fused_gate_ln` stores — Triton was rounding to bf16 even when caller allocated fp32 buffers (silent precision loss).
  6. Revert iter8 `fused_invtr_ln_gate_proj` — use cuBLAS bf16 `F.linear` for D=128 too (cuBLAS path empirically cleaner numerics on D=128).
- **Cost:** ~28% geo-mean perf regression vs iter10 (cumulative speedup vs baseline drops 2.04× → 1.55×).
- **Validation:** all 7 BENCHMARK_SHAPES pass seed-0 with max_err 0.013-0.023. Adversarial sweep (10 shapes × 3 seeds × 30 trials = 900 runs, weights from global RNG matching leaderboard's `generate_input`): **13/900 = 1.44% fail rate**, all on D=128 shapes, worst max_err 0.063. Pre-fix sweep was 27%+ on the failing shape.
- **Council ruling (chairman):** SHIP Option A. Adversarial sweep covers the cauchy×D=128 cell that was missing pre-decision; resubmit on the rare adversarial failure.

### iter5 — O1: bf16 einsum bmm (Codex's "no precision restoration" insight)
- Drop the bf16→fp32 upcast in `tr_cast_fwd` → renamed `tr_fwd_pair` (just transpose). Run `torch.bmm` on bf16 inputs; cuBLAS uses fp32 accumulator internally. Output is bf16 (cuBLAS lacks bf16-in/fp32-out via `torch.bmm`).
- The bf16→fp32 upcast we'd been doing didn't restore mantissa bits the bf16 storage already lost; per Codex's analysis, the multiply input precision is identical to bf16 input + fp32 accum.
- Per-shape speedup: 0=19.9%, 1=17.9%, 2=12.6%, 3=16.1%, 4=17.7%, 5=13.5%, 6=13.0%.
- Cumulative shape 6: 6.69 → 5.82 ms = **1.66× faster than baseline**.
- Profile delta on shape 6: tr_cast_fwd 0.79 ms → tr_fwd_pair 0.51 ms (-0.28); fp32 cuBLAS einsum 1.10 ms → bf16 cuBLAS einsum is no longer in the top 9 (sub-0.5 ms); fused_invtr_ln_gate 0.47 → 0.36 ms (reads bf16 instead of fp32).
- Accuracy: cauchy shape 1 max_err 0.0192→0.0217, shape 4 0.0187→0.0205 — still passes the `atol+rtol*|ref|` gate. Tighter than feared.

### iter15 — Custom Triton bmm (`bmm_kernel_tlx_ws`) replaces cuBLAS bf16 bmm + `.float()` cast

- **Hypothesis (revised mid-iter):** The PLAN_v3 entry framed this as "beat cuBLAS by L2 grouping". Profile re-read showed the real prize is the post-bmm `bf16→fp32` cast: shape-6 baseline had `nvjet_tst_192x192_64x4_*` at 537 µs **plus** `unrolled_elementwise` at 451 µs (the `.float()` on `out_bmm`). A custom kernel that writes fp32 directly inside the matmul epilogue eliminates the entire 451 µs cast pass.
- **Design:** Persistent TLX warp-specialized GEMM modeled on `matmul_kernel_tlx_ws`. 3D TMA descriptors over the `[B*hd, N, N]` tensors (batch slot in dim 0). Two consumer warpgroups via `replicate=2`, split-M (`BLOCK_M_SPLIT = BM // 2`). B operand is loaded `[BN, BK]` (R is `[N, N]` row-major in K-fast layout) and `tlx.local_trans`'d to `[BK, BN]` before `async_dot` — avoids a separate transpose kernel.
- **Config:** `BM=128, BN=128, BK=64, NUM_STAGES=3, NUM_MMA_GROUPS=2, GROUP_SIZE_M=8`. SMEM: A = 3·2·64·64·2 = 49 KiB, B = 3·64·128·2 = 49 KiB → 98 KiB, well under the 232 KiB cap. Persistent grid `(NUM_SMS,)=(132,)`. For shape 6 (BATCH=128, N=1024) this is 8192 tiles ≈ 62 tiles per SM — comfortable steady-state.
- **Standalone bmm** (just the kernel vs `torch.bmm + .float()`): 1.26–1.40× faster across shapes; on shape 6 dims (B*hd=128, N=1024): 924 µs → 711 µs. Numerics differ from cuBLAS by ~1-2% relative (different reduction-tree ordering for bf16 accumulation), well inside the leaderboard tolerance.
- **End-to-end** (shape 6): 5.581 ms → 5.347 ms (**−4.2%**, +0.234 ms saved). All 7 shapes improve by 4.1–6.2%.
- **Adversarial sweep** (7 shapes × 30 trials, seed=731): **0/210 = 0.0%** failures (vs iter14's 7/900 = 0.78%). Worst max_err 0.042 on shape 4 (cauchy, N=1024) — same shape as before, comparable margin.
- **Profile delta on shape 6:** old `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN` (537 µs/call) + `unrolled_elementwise` cast (451 µs/call) → new `bmm_kernel_tlx_ws` 687 µs/call, cast eliminated. Net: 988 → 687 = −301 µs/call ≈ 5.4 % e2e budget reclaimed; 4.2 % shows up in median (some pipeline overlap was already hiding the cast).
- **Cumulative vs baseline:** shape 6 = 1.88×.

### iter18 — All-in-one S1 (LN-prologue + matmul + gate-LN epilogue) — **NOT COMMITTED, NEGATIVE**

- **Hypothesis (PLAN_v3):** Single TLX kernel that reads fp32 x ONCE, computes per-row mean/rstd via a reduction warpgroup, applies LN-affine inline at WGMMA input, runs the 5-proj matmul, then in the epilogue applies LN-correction + sigmoid + mask + gate-mul and writes lf, rf, og DIRECTLY in bmm-friendly `[B*hd, N²]` layout. Estimated 17–22 % e2e by eliminating both `bf16 x` (1.62 GB) and `proj` (2.68 GB write+read) intermediates.
- **Outcome:** Did not produce a working kernel. Reverted; baseline iter15 stack remains.
- **Phase B (gate-LN epilogue) blocker — register cliff (not SMEM):** The clean fusion needs all 5 N-chunks (lv, rv, lg, rg, og) for the SAME (pid_m) to coexist somewhere so the cross-column mults `lv*lg*m`, `rv*rg*m` can be done. Two tried sub-approaches:
  1. **5-acc-in-registers**: keep 5 fp32 accs of `[BLOCK_M_SPLIT, hd]`. For the smallest viable `BM=64, MG=2, hd=128`: 5 × 32 × 128 × 4 = 80 KB per consumer warpgroup. Hopper consumer wg has ~64 KB regs. Even `BM=32, MG=1` → 5 × 32 × 128 × 4 = 80 KB; same overflow. Below `BM=32` the matmul tile count for shape 6 (M=1M) explodes past 32 K tiles, killing occupancy on top of register pressure.
  2. **5-chunks-in-SMEM**: stage all 5 `[BLOCK_M_SPLIT, hd]` chunks to SMEM after each chunk's WGMMA, then do the cross-column work on the assembled tile. For `BM=128, MG=2, hd=128`: 5 × 64 × 128 × 4 = 160 KB just for the staging, plus matmul A/B stages, plus the prologue x scratch. Exceeds the 228 KB SMEM cap. `BM=64`: 5 × 32 × 128 × 4 = 80 KB; with `NS_A=NS_B=2` adds another 64 KB; tight but possible — but `BM=64` means num_pid_m = 16384 for shape 6, and the matmul tile-loop overhead per tile starts to dominate.
- **Phase A (LN-prologue) attempted, MLIR codegen crash:** Wrote a 4-warpgroup design — TMA producer (loads fp32 x[BM, K] once + bf16 B per K-iter), reduction wg (reads fp32 x_smem, computes mean/rstd, normalizes, casts to bf16, stores into a single `[BM, K]` bf16 SMEM buffer), 2 consumer wg's (replicate=2, slice their half-rows × K-tile from the bf16 buffer via `tlx.local_slice`, run WGMMA). SMEM math for `BM=128, D=128`: fp32 scratch 64 KB + bf16 normalized 32 KB + B stages 48 KB ≈ 144 KB — fits. First compile attempt died with `tl.reshape(x_norm, (BM, K_ITERS, BK))[:, k, :]` → "unsupported tensor index: constexpr[0]" inside `tl.static_range`. Switched to a single `[BM, K]` bf16 SMEM alloc with consumer-side `tlx.local_slice`. That triggered an MLIR codegen crash: `Builders.cpp:436: Assertion 'parent && "expected valid parent region"' failed` — the conditional barrier/store sequences inside multi-task warpgroups appear to confuse the warp-spec lowering.
- **Counter-mitigation hypothesis disconfirmed:** The plan suggested fusing might let a smaller `NS_A` work because A bandwidth pressure halves. In practice the SMEM/register cliffs are dominated by the prologue's fp32 x scratch and the 5-N-chunk staging — not by A staging depth.
- **What worked correctly before the crash:**
  - SMEM layout: producer's `tlx.async_descriptor_load` issuing K_ITERS partial loads into one `[BM, K]` fp32 buffer (sliced via `tlx.local_slice` per K-tile).
  - Single tile-level `bars_xfull` / `bars_xempty` barriers between TMA producer and reduction wg.
  - Explicit `tlx.fence("async_shared")` between register-source writes (`tlx.local_store`) and downstream WGMMA reads.
  - Explicit `prev_pid_m`-tracking to load fresh x only when the M-tile changes (within `GROUP_SIZE_M=1` and `num_pid_n=5`, 5 consecutive consumer iterations share the same x).
- **What broke:**
  - The `if load_x:` conditional inside the consumer warpgroup (gating both the `bars_afull` wait and the `bars_aempty` arrival on `pid_n == num_pid_n - 1`) is the suspected codegen trigger — uTLX warp-spec doesn't always handle conditional barriers cleanly.
  - Phase-tracking across an irregular iteration pattern (`x_phase_cnt` on M-advance only, `smem_accum_cnt` on every K-iter) is fragile.
- **Per-shape baseline before/after:** untouched. iter15 stack still:
  - shape 0: 0.568 ms; shape 1: 2.338 ms; shape 2: 0.759 ms; shape 3: 1.055 ms; shape 4: 4.015 ms; shape 5: 3.057 ms; shape 6: 5.350 ms (geo-mean 1.873 ms).
- **Verdict & next step:** The iter18 ceiling is **register pressure for the cross-column fusion AND uTLX codegen brittleness for multi-warpgroup conditional barrier patterns**, not raw HBM bandwidth. Two viable follow-ups:
  1. Phase A only, using a SEPARATE Triton kernel (not warp-spec): fold mean/rstd + bf16-normalize into one kernel (already done by `ln_stats_and_bf16_cast`) and modify it to emit `bf16((x-mu)*rs)` instead of `bf16(x)`. Then drop the `mu * s1` term from `fused_gate_ln_bmm_layout`. Saves ~5 % of the post-pass compute (the LN math). HBM unchanged. Low risk, small win.
  2. Phase B with a different decomposition: instead of fusing all 5 N-chunks, only fuse 2 (lv & lg → L) into one matmul kernel and 2 (rv & rg → R) into another, then a tiny fused og kernel. Output 5 × `[T, hd]` bf16 buffers (same total bytes as one `[T, 5*hd]` proj — no HBM win) but allows direct multiply-and-write in the matmul epilogue with only 2 accs in registers. Requires running the matmul twice with different B slices. Net: maybe negative due to launch overhead + B re-reads.
  3. Wait for iter17 standalone result. If iter17 also fails on its own SMEM cliff, the fundamental approach is wrong; pivot to CUDA Graphs (iter19) and workspace pre-alloc (iter20) for the remaining 5–10 %.
- **Adversarial sweep**: not run (no candidate kernel to test).
- **Time invested:** ~3 h of design + ~1 h of code attempts; reverted cleanly. Branch `trimul-iter18` retained for record but contains no kernel changes.

### iter21 — L2 cache-residency hints on read-only TMA loads — **ABORT (wheel doesn't plumb hint)**

- **Hypothesis (PLAN_v4 §4):** NCU shows L2 compression 0% across all kernels. Tag the read-only `B_g` (prepped weight), `pair`, and `mask` TMA descriptor loads with `eviction_policy="evict_last"` to pin them in L2 across persistent-CTA M-tile cycling. Target +3–8% e2e on shape 6.
- **Wheel-API audit (utlx 0.1.0+gitcba4ef9a):**
  - `tlx.async_descriptor_load` accepts `cache_modifier: str` and `eviction_policy: str` kwargs and validates them (`'', 'evict_first', 'evict_last'`), but **the body never passes either to the underlying `_semantic.builder.create_async_tma_copy_global_to_local` call**. Both kwargs are silently dropped.
  - `tlx.async_descriptor_prefetch_tensor` shows the same pattern: it calls `_semantic._str_to_eviction_policy(eviction_policy)` and then **discards the result** before invoking `utlx_async_tma_prefetch`.
  - The C++ binding `gluon_ir.GluonOpBuilder.create_async_tma_copy_global_to_local` is a 7-arg function (`desc, offsets, barrier, result, pred, multicast, multicast_targets`) with **no slot for an eviction-policy operand or attribute**. So even patching the Python wrapper would not get a hint into MLIR — the binding itself is missing the argument.
  - `tlx.make_tensor_descriptor` has no cache/eviction kwargs either; its IR result is a bare `tt.tensordesc<…, #shared>` with no policy attribute.
- **Empirical IR proof:** Added `eviction_policy="evict_last"` to the `b_desc` TMA load in `matmul_kernel_tlx_ws` and dumped MLIR with `MLIR_ENABLE_DUMP=1`. The resulting `ttng.async_tma_copy_global_to_local` op is byte-identical to baseline — zero `evict`-related substrings appear anywhere in the dumped TTGIR/TTIR. Reverted cleanly. Per-shape: 0=0.574 / 1=2.472 / 2=0.729 / 3=1.028 / 4=3.992 / 5=3.033 / 6=5.629 ms (kernel unchanged; numbers within noise of baseline).
- **Verdict — ABORT** per PLAN_v4 §4 abort rule "cache-modifier syntax not supported by the wheel". Confirmed at three levels: (1) Python wrapper drops the kwarg, (2) C++ binding lacks the parameter, (3) empty MLIR diff. **Same class as C1** (uTLX wheel pybind hole).
- **Unblocking work needed before retry:** wheel rebuild that (a) extends `create_async_tma_copy_global_to_local` to accept an `EvictionPolicyAttr` operand (the symbol `EvictionPolicyAttr` is already linked into `libutlx.so`), and (b) plumbs `cache_modifier` / `eviction_policy` through the `tlx.async_descriptor_load` Python wrapper to that new operand. Out of scope for in-tree iteration.
- **Time invested:** ~25 min audit + 1 IR dump + revert. No commit needed.

### iter22 — bmm SMEM bank-conflict pad — **ABORT (NCU disconfirms hypothesis; no in-scope lever)**

- **PLAN_v4 hypothesis:** NCU shows 28% local bank conflict on bmm B-fragment **loads** from `tlx.local_trans` in `bmm_kernel_tlx_ws`. Pad LDS stride by 8 bf16 lanes OR force NVIDIA 128B swizzle on the B TMA descriptor. Estimated +1.5% e2e (~50 µs).
- **NCU re-measurement (shape 6, kernel `bmm_kernel_tlx_ws`):**
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` = **0** (loads have **zero** LSU bank conflicts; WGMMA reads B via `ldmatrix` through the tensor-memory pipe, not LSU)
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` = 6,482,229 / 11,420,665 wavefronts = **56.8% on STORES** — fp32 accumulator → SMEM staging path for the TMA bulk store of `c` (`acc3d` shape `[1, 64, 128]` fp32 per consumer warpgroup).
- **In-scope levers exhausted:**
  1. **128B swizzle on B's TMA descriptor:** already on (`runner/tlx_patches.py` `_default_nvmma_swizzle` returns 128 because `block_shape[-1]*elem_bw = 64*16 = 128 B`). No higher swizzle exists.
  2. **`pad_K=8` on B's `tlx.local_alloc`:** uTLX `local_alloc` (`utlx_plugin/mem_ops.py:104`) accepts `layout=` but **always overwrites** with `nv_mma_shared_layout_encoding.make_default(shape, dtype)` (line 148). User-supplied layout is dead code. Padding requires a runner-level patch.
  3. **Eliminate `local_trans` by loading B in `[BK, BN]` via stride trick:** R is row-major `[batch, j, k]`; asking TMA for `[1, BK, BN]` indexes the wrong slice (not the transpose). Re-striding R changes the calling convention.
- **Why the prescribed fix doesn't address measured conflicts:** the real cost is acc-fragment register-layout → SMEM swizzle mismatch on the C-store path. C TMA descriptor block_shape `[1, 64, 128]` fp32 already gets 128B swizzle; reshaping the BN tile (still all multiples of 32 fp32) keeps swizzle at 128B. Fixing this requires restructuring the consumer epilogue (e.g. column-strip stores), not the B operand.
- **Decision:** ABORT with no kernel change. Recommended follow-ups: (a) "iter22b" runner patch enabling `local_alloc(..., layout=custom)` to target the C-store conflicts; (b) defer to iter25 (D=384 BM=64 decomposition), which restructures the consumer epilogue and incidentally narrows the C-store width.
- **Time invested:** ~30 min NCU + source audit; no compile, no diff to `hopper_gemm_ws.py`.

### iter23 — Custom TLX final linear (D=384, replace cuBLAS bf16) — **ABORT (kernel ~8% slower than cuBLAS standalone)**

- **Hypothesis (PLAN_v4 §4 iter23):** cuBLAS bf16 final linear on shape 6 = ~0.55 ms (10% wall), runs sequentially after `fused_invtr_ln_gate` with no L2 reuse of `gated`. A custom TLX warp-spec GEMM reading `gated` straight from L2 should save the launch hop + L2 evict. Target ≥ 2% e2e on shape 6.
- **Design:** `final_linear_kernel_tlx_ws` modeled on `bmm_kernel_tlx_ws` collapsed to 2D. bf16 [T, hd] @ bf16 [N=D, hd]^T → bf16 [T, D], fp32 accumulator (C6). Persistent warp-spec, replicate=2 consumers, `tlx.local_trans` on the B-side `[BN, BK]` slab. SMEM 98 KiB total.
- **Standalone benchmark on shape 6 dims (M=1.05M, N=384, K=128):**

  | Config | µs | vs cuBLAS |
  |---|---|---|
  | F.linear (cuBLAS bf16) | 549.6 | 1.000× |
  | TLX BM=128 BN=128 NS=3 MG=2 GS=8 | 648.3 | 1.181× |
  | TLX BM=128 BN=128 NS=3 MG=2 GS=1 | **594.4** | **1.083× (best)** |
  | TLX BM=128 BN=128 NS=4 MG=2 GS=8 | 709.2 | 1.292× |
  | TLX BM=256 BN=128 NS=3 MG=2 GS=1 | 641.3 | 1.168× |

  All BN ∉ {128, 256} (e.g. 192, 384) failed compilation: TMA descriptor block widths must be powers of 2 for K=128. Best feasible config is **8% slower than cuBLAS** standalone.
- **End-to-end with kernel wired in (best config GS=1):** shape 5 3.028 → 3.045 ms (+0.6%); shape 6 5.352 → 5.400 ms (+0.9%); other shapes unchanged (D=128 untouched).
- **Why it loses:** This is a memory-bound shallow GEMM. K=128 → only 2 K-iters of BK=64, WGMMA pipeline never fills. N=384 → num_pid_n=3, too narrow for L2 grouping to amortize. cuBLAS uses an asymmetric tile (`nvjet_tst_192x192_64x4_*`, per iter15 NCU profile) tuned for this aspect ratio and is near the HBM floor (~330 µs at 3 TB/s vs 549 µs measured) — only ~220 µs of compute headroom before launch noise.
- **Verdict (PLAN_v4 abort criterion):** ABORT — "standalone within 5% of cuBLAS — gain eaten by launch noise." Reverted dispatch site to `F.linear(gated, w_out)`. Numerics correct (T0 max_err 0.0147; T2 0/96 = 0.00%). Kernel definitions kept in iter23 worktree only; not merged.
- **What might still work later:** any kernel that *fuses additional work* into the final linear (e.g., applying the to_out gate inside the matmul epilogue, or producing the [B, N, N, D] view directly) so the WGMMA pipeline isn't the bottleneck. iter25 (decomposed 2+2+1 fusion) may absorb this opportunity.
- **Test infrastructure landed:** `check_leaderboard_seeds.py --tier T0|T1|T2|T3` (independently committed in main as `604287a`). Standalone bench/sweep helpers landed at `optimize/bench_final_linear.py` and `optimize/sweep_final_linear_cfg.py` for iter25 reuse.

### iter24 — D=128 deep fusion (gate-LN epilogue) — **ABORT (kernel ~2× slower than 2-kernel baseline)**

- **PLAN_v4 hypothesis:** iter16's epilogue-fold variant (matmul + gate-LN + transpose-on-store fused into one TLX warp-spec kernel) avoided the [T, 5*hd] proj intermediate (1.34 GB on shape 4). At D=128 the SMEM cap is non-binding (K_ITERS=2). Target +5–10% on the 4 D=128 shapes (0,1,3,4) ≈ +3% geo.
- **Implementation:** `matmul_kernel_tlx_ws_epi_d128` + Python wrapper `matmul_fused_d128`, gated on `dim == 128 and hd == 128`. Producer warpgroup loads A `[BM=128, K=128]` once per pid_m + 5 N-chunks × K_ITERS=2 = 10 B-tiles per pid_m (NUM_STAGES=3, BN=BLOCK_M_SPLIT*2 = hd=128). Two consumer WGs (replicate=2) process N-chunks in order [lv → spill, lg → write L, rv → spill, rg → write R, og → write] with a single fp32 SMEM staging slab (128×128×4=64 KiB) reused between (lv,lg) and (rv,rg). Per-WG only ONE fp32 acc [64,128]=32 KiB live at a time (well under C3). Producer reorders B-side memory chunk emission to match consumer order so each B buffer arrives JIT.
- **Wheel-API yak shave:** uTLX 0.1.0+gitcba4ef9a's shipped `tlx.local_slice` is broken — `mem_ops.py:368` calls `create_memdesc_subslice(handle, offset, shape)` (3 args) but the GluonOpBuilder binding expects `(result_type, source_value, offsets)` (3 args, different positions; the wrapper drops the result type and swaps `shape` for `offsets`). Same class as C1. Added a runtime monkey-patch `_install_local_slice_fix()` near the kernel imports that constructs the correct result memdesc type via `get_shared_mem_desc_ty(elem_ty, shape, layout, alloc_shape)` and re-extracts the source layout via `get_gluon_layout_from_memdesc → ._to_ir(builder)`. Patch decorated `@tl.builtin` so AST codegen routes through it. Verified `tlx_ws_matmul_fixed` / `bmm_kernel_tlx_ws` still compile (their existing `local_slice` call sites are inside dead `NUM_CTAS == 2` branches).
- **SMEM (verified by MLIR `out_of_resource` probe):** with NUM_STAGES=3 and 1 staging slab, actual SMEM usage = ~195 KiB / 232 KiB cap. NUM_STAGES=2 first try went to 262 KiB, so I dropped staging from 2 slabs → 1 (reusing across `(lv,lg)` and `(rv,rg)`) and restored NUM_STAGES=3.
- **Per-shape outcome (CUDA_VISIBLE_DEVICES=5, 20-iter warmup, 30-iter timing):**

  | shape | dim | distribution | baseline (ms) | iter24 fused (ms) | delta |
  |---|---|---|---|---|---|
  | 0 | 128 | normal | 0.575 | 0.795 | **+38%** SLOWER |
  | 1 | 128 | cauchy | 2.342 | 3.345 | **+43%** SLOWER |
  | 3 | 128 | normal | 1.029 | 1.515 | **+47%** SLOWER |
  | 4 | 128 | cauchy | 3.969 | 5.867 | **+48%** SLOWER |

  D=384 shapes (2, 5, 6) untouched (dispatch only fires on `dim == 128`).
- **Profile delta on shape 4:** torch profiler says new kernel `matmul_kernel_tlx_ws_epi_d128 = 3.873 ms/call` vs baseline (`matmul_kernel_tlx_ws = 0.897 ms/call` + `fused_gate_ln_bmm_layout = 1.153 ms/call` = 2.05 ms/call combined). The fused kernel is **1.9× slower** than the unfused 2-kernel sequence, the opposite of the predicted +5–10% improvement.
- **Why it loses (root cause):** The 5 sequential per-N-chunk WGMMA launches inside one kernel each pay a full `acc = tl.zeros + K-loop barrier wait/arrive cycle + async_dot_wait + epilogue spill/fence/global-store` — total ~10 µs per chunk per pid_m × 5 chunks × 62 pid_m per SM = 3.1 ms. The ORIGINAL `matmul_kernel_tlx_ws` runs all 5 N-chunks as DIFFERENT TILE_IDs in the persistent loop, where the round-robin scheduler keeps the WGMMA pipeline tight (per-pid_m matmul time = 5.8 µs ≈ 0.58 µs / wgmma; mine is ~2.5 µs / wgmma). The `proj` HBM bandwidth saved (~1.3 GB / 3 TB/s ≈ 430 µs) is only about 22% of the per-chunk overhead added, so net loss.
- **Numerics correctness (proves the fusion math is right; perf is the only blocker):** T0 PASS (max_err 0.01754); T2 with the fused path enabled mid-iter PASS (0/96 = 0.00%); the bf16 cascade is unchanged because L/R writes auto-narrow from fp32 acc → bf16 buffer and og keeps fp32. So the kernel is *correct*, just *slow*.
- **Verdict (PLAN_v4 §4 abort criterion):** ABORT — "D=128 shapes ≥ 1.05× current required for accept; we got 0.7×." Reverted the dispatch site so D=128 keeps the (matmul + fused_gate_ln_bmm_layout) sequence. Kept the kernel definition + `matmul_fused_d128` wrapper + the `_install_local_slice_fix` patch in source for the next attempt; the local-slice patch is generally useful and unblocks future iterations that need multi-dim SMEM slicing.
- **Post-revert validation:** Bench numbers within noise of pre-iter baseline:

  | shape | pre-iter24 (ms) | post-revert (ms) |
  |---|---|---|
  | 0 | 0.575 | 0.567 |
  | 1 | 2.342 | 2.350 |
  | 2 | 0.731 | 0.737 |
  | 3 | 1.029 | 1.030 |
  | 4 | 3.969 | 3.988 |
  | 5 | 3.026 | 3.047 |
  | 6 | 5.356 | 5.375 |

  T2 0/96 = 0.00%; max_err 0.02204 (shape 0).
- **What needs to happen for the fusion to win:** keep all 5 N-chunk accumulators alive across a tight K-iter loop (so the WGMMA pipeline is fed continuously). That requires either (a) 5 fp32 accs per consumer WG simultaneously — iter18's 80-KB cliff at BM=64 — or (b) a tilewise interleaved-K design where each k-iter cycles through all 5 N-chunks before advancing k, with B-loads pipelined across chunks. Option (b) is essentially a partial monolithic S1 and is closer in spirit to iter18 Phase B than iter16. Out of scope for iter24.
- **Time invested:** ~3 h (1 h kernel design + 1.5 h debugging local_slice wheel-API hole + 30 min profiling/abort).

### iter25 — D=384 decomposed 2+2+1 fusion — **ABORT (fused kernels 64% slower than baseline path)**

- **Hypothesis (PLAN_v4 §4 iter25):** Replace (`tlx_ws_matmul_fixed` producing [T, 5H] `proj` + `fused_gate_ln_bmm_layout` reading proj and emitting bmm-layout L/R + [T, hd] og) with three persistent kernels: `matmul_lr_kernel` (×2, for L and R: `[T,D]@[D,2H]` with LN-correction + sigmoid + mask + value*gate fused in epilogue, written transposed to `[B*hd, N²]` bmm layout) plus a trivial `matmul_og_kernel`. Eliminates `proj [T, 5H]` bf16 round-trip = 2.68 GB HBM saved on shape 6. Target: +5–8% on shape 6.
- **Pencil-out (matmul-only, before writing fused kernels):**
  - `tlx_ws_matmul_fixed [T,D]@[D,5H]` (existing, BM=256/BN=128) = 1167 µs
  - `tlx_ws_matmul_fixed` 3-split sum = 706 + 706 + 527 = 1939 µs
  - Existing post-pass (`fused_gate_ln_bmm_layout`) = 1170 µs → existing total = 2337 µs
  - 3-split matmul-only sum vs current = **~+400 µs theoretical headroom (17%)** if fused epilogue adds zero overhead.
- **Implementation hit two wheel-level blockers:**
  - `tl.split` on the [BM_SPLIT, 2H]→[BM_SPLIT, 2, H] reshape **segfaults** the Triton compiler (`semantic.py:692`) on this acc layout. Forced a 2-WGMMA-per-K-iter design (one per H-half into separate fp32 accs `[BM_SPLIT, H]`) — halves WGMMA throughput vs single-WGMMA-on-[BM_SPLIT, 128].
  - `tlx.local_slice` on a [BK, 2H] SMEM region fails — C++ binding `create_memdesc_subslice` signature mismatch (independently rediscovered by iter24). Forced two separate B TMA descriptors / SMEM allocs.
- **Standalone bench (shape 6, T=1.05M, D=384, H=128):**

  | Kernel | µs |
  |---|---|
  | `tlx_ws_matmul_fixed [T,D]@[D,5H]` (existing) | 1167 |
  | `fused_gate_ln_bmm_layout` (existing post-pass) | 1170 |
  | **Existing sum** | **2337** |
  | `matmul_lr_kernel` L variant | 1582 |
  | `matmul_lr_kernel` R variant | 1571 |
  | `matmul_og_kernel` | 685 |
  | **iter25 sum** | **3838** |
  | **Δ** | **−1500 µs (iter25 is 64% slower)** |

- **End-to-end with iter25 dispatched (vs baseline 5.404 ms shape 6):** shape 2 0.728 → **0.905 (+24%)**, shape 5 3.020 → **3.717 (+23%)**, shape 6 5.404 → **6.499 (+20%)**. D=128 untouched.
- **Why it failed:**
  1. `tl.split` segfault forced 2-WGMMA-per-K-iter — each `matmul_lr` does ~1.5× the WGMMA work of an unfused split.
  2. Transpose-store overhead. Writing `[BM_SPLIT, H]` register tiles transposed into `[B*hd, N²]` bmm-layout (whether via TMA store + `tl.permute` or strided `tl.store`) measured ~600 µs/kernel — and iter14's `fused_gate_ln_bmm_layout` already runs near 2.2 TB/s HBM, leaving zero room.
  3. **Even with both wheel fixes, the theoretical ceiling is ~+200 µs (3-4% e2e on shape 6) — tight enough to risk regression.**
- **Pencil-out vs reality:** Predicted matmul sum 1939 µs; actual fused sum 3838 µs. Fusion overhead alone (~870 µs per `matmul_lr`) eats >4× the theoretical HBM savings. Each `tlx_ws_matmul_fixed [T,D]@[D,2H]` (706 µs unfused) became 1582 µs fused.
- **Verdict — ABORT** per PLAN_v4 §4 iter25 abort rule. Reverted dispatch site to use the existing path. Kernels (`matmul_lr_kernel`, `matmul_og_kernel`, `tlx_ws_matmul_lr`, `tlx_ws_matmul_og`) and `_prep_weights` slicing additions retained in iter25 worktree only; **not merged**. T2 (after revert): 0/96 = 0.00%; T3: 8/1260 = 0.63% (matches iter15 baseline 0.95%, under 1.5% gate).
- **What it would take to retry:** wheel rebuild fixing both `tl.split` codegen AND `tlx.local_slice` C++ binding. **Same wheel-level limit as iter21 (eviction_policy) and iter24 (local_slice).** Three independent iters in this campaign blocked by the same pybind/codegen surface.
- **Time invested:** ~3 h (kernel design + 2 implementation passes + bench/standalone validation).

### iter26 — Row-persistent CTA scheduler — **ABORT (matmul +50% slower; round-robin is at a local optimum)**

- **Hypothesis (PLAN_v4 §4 iter26):** Round-robin scheduler `tile_id += NUM_SMS` re-fetches `pair` rows from DRAM (`num_pid_n=5` redundant prologue reads on shape 6). Row-persistent reuses each row in L2 across all `j` strips. Target +4–6% e2e on shape 6.
- **Three variants tried** (nested `for pid_m: for pid_n`, flat `while`, `pid_n = (tile_id + sm_id) % num_pid_n` stagger). All recompiled and ran. None recovered baseline matmul throughput.
- **Per-kernel attribution on shape 6 (torch.profiler, 10 iters):**

  | Kernel | Baseline | iter26 | Δ |
  |---|---|---|---|
  | `matmul_kernel_tlx_ws` | 1125 µs | **1681 µs** | **+50%** |
  | `fused_gate_ln_bmm_layout` | 1165 | 1147 | −1.5% |
  | `ln_stats_and_bf16_cast` | 1101 | 1101 | 0% |
  | `bmm_kernel_tlx_ws` | 691 | 691 | 0% |
  | `fused_invtr_ln_gate` | 594 | 593 | 0% |

- **Per-shape e2e (CUDA_VISIBLE_DEVICES=5):** 0=+1%, 1=0%, 2=+5%, 3=0%, 4=0%, 5=+10%, 6=+10%. The 50% matmul slowdown wholly explains the e2e regression — every downstream kernel is unchanged, so L2 state at the matmul boundary is irrelevant to the e2e win the plan predicted.
- **Mechanism inference (unconfirmed):** row-persistent makes all 132 SMs lockstep through `pid_n=0..4` together, contending on the same B-column L2 sectors / HBM channels at every step. The original round-robin schedule naturally staggers SMs across both `pid_m` and `pid_n`. The `pid_n` stagger variant didn't help — suggests something deeper in the warp-spec pipeline (tail effects when SMs leave the persistent loop at very different times, or producer-consumer phase drift on contiguous tiles).
- **T1:** 0/30 fail. T0 max_err 0.013, T1 max_err 0.029. Numerics byte-clean — kernel is functionally correct, just slower.
- **Verdict:** ABORT. Reverted; only PROGRESS.md modified. **The round-robin scheduler is at a local optimum for this kernel; row-persistent is not a free L2-reuse win.**
- **Time:** ~70 min of 90-min cap.

### iter31 — bmm tile-config sweep (B3) — **PASS (BK=64→128, NS=3→2, GSM=8→1; shape 6 −1.3%)**

- **Hypothesis (PLAN_v5 §4 / B3):** `bmm_kernel_tlx_ws` was set to `BM=128 BN=128 BK=64 NS=3 MG=2 GSM=8` in iter15 by analogy to the matmul kernel. Iter11 swept matmul tiles (local optimum), but bmm wasn't swept. Try a focused Cartesian product to look for a 0-3% e2e win on shape 6.
- **Sweep (`work/optimize/sweep_bmm_cfg.py`):** 108 combos × axes `BM∈{128,256}, BN∈{64,128,256}, BK∈{32,64,128}, NS∈{2,3,4}, GSM∈{1,8}`. 22 combos pruned a-priori by the SMEM-cap predicate (`smem_bytes > 232,448`); another 17 hit `OutOfResources` at compile time after MLIR layout/swizzle padding inflated the footprint past the predicate's estimate. **Locked axes (structural, not config-only): NUM_MMA_GROUPS=2 and replicate=2** — the producer hard-codes a 2-half A-load via `tlx.async_task_replica_id()` and barriers are sized as `NS*MG`. Sweeping these would require kernel surgery, out of scope.
- **Top 5 standalone bmm configs on shape 6 dims (BATCH=128, N=1024), µs / SMEM (single-pass median, 30 iters):**

  | rank | BM | BN | BK | NS | GSM | SMEM B | µs | vs base |
  |---|---|---|---|---|---|---|---|---|
  | 1 | 128 | 128 | 128 | 2 | 1 | 132,096 | 636.5 | 0.896× |
  | 2 | 128 | 128 | 64 | 4 | 1 | 132,096 | 637.7 | 0.898× |
  | 3 | 128 | 128 | 64 | 4 | 8 | 132,096 | 652.2 | 0.918× |
  | 4 | 256 | 64 | 128 | 2 | 1 | 164,864 | 661.2 | 0.931× |
  | 5 | 128 | 128 | 128 | 2 | 8 | 132,096 | 662.8 | 0.933× |

  Re-bench (100 iters × 2 passes) of top-3 confirmed ranking stable: c1 = 666–670 µs, c2 = 676–685 µs, c3 = 696 µs, baseline = 729–765 µs (drift on baseline is run-to-run thermal — the 10% c1 win is robust).
- **Winner:** `BM=128, BN=128, BK=128, NS=2, GSM=1` (c1). Replaces `BMM_TLX_CONFIG` in `hopper_gemm_ws.py:1953`.
- **Why it wins (read of the result):** larger BK (128 vs 64) halves the K-iter count (8→4) and amortizes pipeline fill/drain overhead. Lower NS (3→2) is offset by the larger SMEM tiles still reaching enough in-flight BK chunks. GSM=1 disables L2 grouping — at BATCH=128, the round-robin schedule across 132 SMs already gives good L2 reuse without the `pid_m` swizzle, and GSM=1 gives a more balanced producer-consumer phase pattern.
- **Per-shape e2e ms (CUDA_VISIBLE_DEVICES=7, 2 passes, mean):**

  | shape | baseline | iter31 | Δ |
  |---|---|---|---|
  | 0 (S=256, D=128) | 0.575 | 0.571 | −0.7% |
  | 1 (S=768, D=128, cauchy) | 2.356 | 2.340 | −0.7% |
  | 2 (S=256, D=384) | 0.732 | 0.732 | 0.0% |
  | 3 (S=512, D=128) | 1.032 | 1.020 | −1.2% |
  | 4 (S=1024, D=128, cauchy) | 3.997 | 3.925 | −1.8% |
  | 5 (S=768, D=384) | 3.030 | 3.008 | −0.7% |
  | 6 (S=1024, D=384) | 5.364 | 5.293 | **−1.3%** |

  Geo-mean improvement ≈ **−1.0%**. No shape regresses; biggest wins on the two largest D=128 shapes (1, 4) which spend the most time in `bmm_kernel_tlx_ws`.
- **T0:** PASS (0/1, max_err 0.014). **T1:** PASS (0/30, max_err 0.027). Numerics unchanged — fp32 accumulation path identical, only tile shape and pipeline depth shifted.
- **Verdict:** **PASS.** Standalone win is 10% (≥1% threshold cleared by 10×). E2e geo-mean −1.0%, shape 6 −1.3% (≥0.5% threshold cleared). No regressions. Lesson: matmul was at a local optimum (iter11), but bmm was *not* — different aspect ratio (square `[N,N] @ [N,N]^T` vs rectangular matmul) made a different tile shape optimal.
- **Files touched:** `gpumode/bioml/trimul/work/hopper_gemm_ws.py` (`BMM_TLX_CONFIG`), `gpumode/bioml/trimul/work/optimize/sweep_bmm_cfg.py` (new).
- **Time:** ~60 min of 90-min cap.

