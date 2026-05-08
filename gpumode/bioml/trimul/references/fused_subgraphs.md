# TriMul Fused Subgraphs

Manual extraction of fusible subgraphs from `TriMul.forward` (`submission.py:28-70`). Boundaries chosen where dtype changes, parallel ops share an input, or epilogues fold into a matmul.

Notation: `B = batch_size`, `S = seq_len`, `D = dim`, `H = hidden_dim`. From `trimul_problem.md`: `H=128` always, `D ∈ {128, 384}`, `S ∈ {256, 512, 768, 1024}`, `B ∈ {1, 2}`.

---

## Subgraph 1 — `norm_5proj_gate_mask` (input head)

| | |
|---|---|
| **Input** | `x: [B, S, S, D]` fp32, `mask: [B, S, S]` fp32 (may be all-ones) |
| **Output** | `left: [B, S, S, H]` bf16, `right: [B, S, S, H]` bf16, `out_gate: [B, S, S, H]` fp32 |
| **Source** | `submission.py:38-53` (+ bf16 cast at line 55) |
| **Modules touched** | `norm`, `left_proj`, `right_proj`, `left_gate`, `right_gate`, `out_gate` |

### Ops
1. `LayerNorm(x)` over last dim — `submission.py:38`
2. **5 parallel Linear** (D → H, no bias) on the *same* normalized x — `submission.py:41-50`
3. `sigmoid` on the 3 gates — `submission.py:48-50`
4. `left  = left_proj(x_norm)  * mask.unsqueeze(-1) * sigmoid(left_gate(x_norm))`  — `:45,52`
5. `right = right_proj(x_norm) * mask.unsqueeze(-1) * sigmoid(right_gate(x_norm))` — `:46,53`
6. Cast `left, right → bf16` for the einsum — `:55`

### Fusion rationale
All 5 linears read the same `x_norm` of size `B·S²·D`. Concatenating weights into one `W: [5H, D]` matmul cuts read traffic 5×. Sigmoid + mask + gate-mul are pure elementwise on the projection outputs and have no reuse — folding them into the matmul epilogue avoids three extra HBM round-trips for `left`/`right`/`out_gate`.

When `nomask=true` (4 of 7 benchmark shapes), the `* mask` op can be specialized away.

### Weights fused
- `W_in: [5H, D]` — stack of `left_proj`, `right_proj`, `left_gate`, `right_gate`, `out_gate`
- LayerNorm γ, β: `[D]`

### Memory traffic (per batch element)
- Reads: `x` once (`S²·D` fp32) + `W_in` (`5H·D`)
- Writes: `left`, `right` (`S²·H` bf16 each) + `out_gate` (`S²·H` fp32)

---

## Subgraph 2 — `trimul_einsum`

| | |
|---|---|
| **Input** | `left: [B, S, S, H]` bf16, `right: [B, S, S, H]` bf16 |
| **Output** | `out: [B, S, S, H]` fp32 |
| **Source** | `submission.py:55` |

### Op
```
out[b, i, j, d] = Σ_k left[b, i, k, d] * right[b, j, k, d]
```

### Shape semantics
Per `(b, d)`, this is a `[S, S] @ [S, S]ᵀ` matmul where rows of left are indexed by `i` and rows of right by `j`. So it's `B · H` independent S×S×S matmuls. Two valid mappings:
- **Per-(b,d) standard batched gemm** — grid `(B, H, ⌈S/BLOCK_I⌉, ⌈S/BLOCK_J⌉)`, each work-item is a textbook tiled matmul. Simple and well-tuned by Triton.
- **Batched over (b,d) with H packed into the load width** — amortizes the `left`/`right` row loads across H. Win is small in practice because `H=128` is already a single 256 B bf16 vector load; not worth the complexity.

### FLOPs
`2 · B · S³ · H` bf16 multiply-adds.

For the largest benchmark (`B=1, S=1024, H=128`): **0.275 TFLOP** (≈0.28 ms at H100 bf16 peak 989 TFLOP/s). Memory traffic is ≈1.0 GB (≈0.30 ms at 3.3 TB/s HBM). **Near the roofline** — neither cleanly compute- nor bandwidth-bound at this shape.

| Shape | TFLOP | compute @ 989 TF/s | IO (GB) | mem @ 3.3 TB/s |
|-------|-------|--------------------|---------|-----------------|
| B=1,S=1024 | 0.275 | 0.28 ms | 1.0 | 0.30 ms |
| B=1,S=768  | 0.116 | 0.12 ms | 0.58 | 0.17 ms |
| B=1,S=512  | 0.034 | 0.03 ms | 0.26 | 0.08 ms |
| B=2,S=256  | 0.0086 | 0.01 ms | 0.13 | 0.04 ms |

### Hard boundary justification (vs subgraph 1)
- Every `out[b,i,j,d]` reuses both `left[b,i,*,d]` and `right[b,j,*,d]`. The reuse pattern requires materializing `left`/`right` so they can be tiled across `(i, j)` blocks — they cannot stream from subgraph 1.
- Dtype change (fp32 → bf16) at the input edge; (bf16 → fp32) at the output edge.

### Tile sizing (B-only, when not fused with subgraph 3)
- **Do not** instantiate `BLOCK_I × BLOCK_J × H` fp32 accumulators ("pack full H") — register and SMEM pressure explodes.
- Suggested: `BLOCK_I=128, BLOCK_J=128, BLOCK_K=64`, `num_warps=8`, `num_stages=3`, bf16 inputs with fp32 accum. Staging is `(128×64 + 64×128) × 2 B × 2 buffers ≈ 96 KB` SMEM, safely under H100's 228 KB/SM.
- For `S=256` use `64×128` or `128×64` to keep enough blocks for occupancy.
- Use a small `VEC_H ∈ {1, 4, 8}` purely for vectorized loads on the H axis.

---

## Subgraph 3 — `norm_gate_proj` (output epilogue)

| | |
|---|---|
| **Input** | `out: [B, S, S, H]` fp32 (from subgraph 2), `out_gate: [B, S, S, H]` fp32 (from subgraph 1) |
| **Output** | `[B, S, S, D]` fp32 |
| **Source** | `submission.py:67-70` |
| **Modules touched** | `to_out_norm`, `to_out` |

### Ops
1. Cast `bf16 → fp32` (already done in einsum output) — `:67`
2. `LayerNorm(out)` over last dim (size H=128) — `:68`
3. Elementwise `* out_gate` — `:69`
4. `Linear(H → D, no bias)` — `:70`

### Fusion rationale
LayerNorm + gate-multiply are pure epilogue on the H axis — both fold into the loading of A in the `[H] @ [H, D]` matmul. The norm needs one pass to compute mean/var over H per `(b,i,j)`; if the kernel keeps a full H-vector in registers/SMEM per tile, the matmul reduction over H runs in the same pass.

Since `H=128` always, the full H-vector easily fits in SMEM/registers per `(i,j)` tile — no need for two-pass norm.

### Weights fused
- LayerNorm γ, β: `[H]`
- `to_out.W: [D, H]`

---

## Summary

| # | Subgraph | Lines | I/O shapes | Roofline character (largest shape) |
|---|----------|-------|------------|-------------------------------------|
| 1 | `norm_5proj_gate_mask` | 38-53 | `[B,S,S,D]+[B,S,S]` → 2× bf16 + 1× fp32 of `[B,S,S,H]` | **bandwidth-bound** (~1.5 GB → ~0.45 ms; ~2.5 GB at D=384 → ~0.75 ms) |
| 2 | `trimul_einsum` | 55 | 2× `[B,S,S,H]` bf16 → `[B,S,S,H]` fp32 | **near-roofline** (compute ≈ memory time, both ≈ 0.3 ms) |
| 3 | `norm_gate_proj` | 67-70 | 2× `[B,S,S,H]` → `[B,S,S,D]` | **bandwidth-bound** (~1.0 GB at D=128 → 0.30 ms; 2.5 GB at D=384 → 0.75 ms) |

Wall-time ranking on the largest D=384 shapes is roughly **A ≈ C > B**. Subgraph 2 is *not* the wall-time dominator — the input head and output epilogue carry comparable or larger bandwidth cost.

### Recommended cross-subgraph fusion: **fuse B + C**

The `[B,S,S,H]` fp32 intermediate between B and C costs ~1 GB of write+read for the largest shape. Fusing eliminates this round-trip.

| Aspect | Detail |
|---|---|
| Saves | ~1 GB HBM traffic per call → ~0.30 ms at peak BW; **15–30% end-to-end** on largest shapes (more for D=384) |
| Feasibility | `H=128` fits on-chip — per `(i,j)` H-vector buffer for the LayerNorm + matmul epilogue is ~98 KB SMEM with `BLOCK_I=16, BLOCK_J=12 or 16` |
| Algorithm | Accumulate H-vector into SMEM during K-loop → reduce-over-H for LN mean/var → multiply by `out_gate` (loaded once) → `(BLOCK_I·BLOCK_J, H) @ (H, D)` epilogue → store final D-vector once. For D=384 loop the small H→D gemm 3× with `N_TILE=128`. |
| Tile recipe | `BLOCK_I=16, BLOCK_J=16, BLOCK_K=64`, fp32 accum. Buffer `BLOCK_I·BLOCK_J·H·4 = 16·16·128·4 ≈ 128 KB`, with double-buffered A/B staging tiles. |

A↔B fusion is **not** worthwhile: it would double `x` reads (left needs rows `(i,k,*)`, right needs rows `(j,k,*)`).

### Specialization opportunities (per-shape)

| Shape variant | Opportunity |
|---------------|-------------|
| `nomask=true` (4/7 cases) | Skip mask load + multiply in subgraph 1 — emit a no-mask fastpath kernel |
| `D=384` (3/7 cases) | Subgraph 1: 5×384=1920 output channels — wider concat matmul; subgraph 3: H=128 → D=384 (loop epilogue 3× with N_TILE=128) |
| `D=128` (4/7 cases) | Subgraphs 1 and 3 are square matmuls (D=H=128) |
| `H=128` always | Per-(i,j) H-vector fits in SMEM — enables the B+C fusion above |
| `S=1024, B=1` (2/7 cases) | Persistent kernel for B+C reduces launch overhead and keeps `left`/`right` tiles in L2 |
| Cauchy distribution (2/7 cases) | Heavy tails risk overflow on the bf16 cast at `submission.py:55`. Consider per-batch input scaling (e.g., divide by σ_estimate, multiply back at the end) or clamp before the cast. Keep einsum accumulator in fp32. |
| All shapes | Pre-transpose `W_in: [5H, D]` and `W_out: [D, H]` once into tensor-core swizzled layout to reduce L2 misses |

---

## Multi-Agent Council Review

This decomposition was reviewed by GPT-5 (via Codex CLI) and Gemini 2.5 Pro (via Gemini CLI) on 2026-05-07. Key corrections to the original draft are folded into the sections above; this section captures the council's findings, the points of disagreement, and the rationale for the resolution.

### Corrections accepted from both reviewers (high confidence)

| Original claim | Correction | Source |
|---|---|---|
| Einsum is **2.7 TFLOP**, ~2.7 ms at largest shape | Off by 10×: actually **0.275 TFLOP**, ~0.28 ms | Both |
| Einsum is the **dominant** wall-time cost | Wrong — einsum is near-roofline; **subgraphs A and C are bandwidth-bound** and likely larger wall-time contributors | Both |
| Tile hint: "pack the full H=128 along the load width" | Bad — `BLOCK_I × BLOCK_J × H` fp32 accumulators blow registers/SMEM. Use small `VEC_H` for vectorized loads only | Both |
| Cauchy inputs not mentioned | 2/7 shapes use cauchy distribution; heavy tails risk overflow on the bf16 cast for the einsum | GPT-5 only |
| Persistent kernels & swizzled weight layouts | Worth specializing for largest shapes | GPT-5 only |

### Disagreement: should B+C be fused?

| Reviewer | Position | Reasoning |
|---|---|---|
| GPT-5 | **Fuse them** | Saves ~1 GB HBM traffic per call (~0.30 ms at peak BW). Since A and C are bandwidth-bound, eliminating the B↔C intermediate is exactly the right lever. H=128 makes the per-(i,j) H-vector buffer fit on-chip cleanly (~98 KB SMEM). Estimated 15–30% end-to-end win on largest shapes. |
| Gemini 2.5 Pro | **Don't fuse** | Subgraphs B and C are not the bottleneck — the effort is better spent elsewhere. Fused kernel is complex to get right. |

**Resolution: GPT-5's argument wins.** Gemini's reasoning hinges on a separate (incorrect) claim that subgraph 1 takes ~6.4 ms in pure FP32 cuda cores — but on H100 with TF32 (the PyTorch default) it takes ~0.17–0.52 ms, so subgraph 1 is *not* a 20× outlier dominating everything else. With the actual roofline numbers, B+C fusion is the highest-impact single optimization available.

### Tile-sizing concrete recommendations from the council

| Kernel | BLOCK_I | BLOCK_J | BLOCK_K | num_warps | SMEM | Source |
|---|---|---|---|---|---|---|
| B-only (einsum) | 128 | 128 | 64 | 8 | ~96 KB | GPT-5 |
| B-only alternative | 64 | 64 | 32 | — | smaller | Gemini |
| **B+C fused** | 16 | 12–16 | 64 | — | ~98–128 KB | GPT-5 |

### Notes on reviewer credibility

- GPT-5's analysis was grounded in concrete roofline math (HBM bytes vs FLOPs at 989 TF/s and 3.3 TB/s) and reproduced the 10× FLOP error to the gigaFLOP. Numbers in this document were spot-checked against an independent Python calculation and matched.
- Gemini caught the same FLOP error and the "pack full H" tile mistake, but its `~6.4 ms` Subgraph 1 estimate assumed pure FP32 cuda cores, which the developer would not actually use in a Triton kernel. This led it to over-prioritize subgraph 1 and under-value B+C fusion.
