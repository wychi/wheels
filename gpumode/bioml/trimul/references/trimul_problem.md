---
name: TriMul GPUMode problem spec
description: Problem description and benchmark shapes for the GPUMode "trimul" leaderboard task — AlphaFold3 outgoing Triangle Multiplicative Update, forward only, on H100
type: project
originSessionId: bb120acb-0303-4cb4-bbb0-d080cfd578ba
---
GPUMode leaderboard task: implement the **outgoing** TriMul (Triangle Multiplicative Update) from AlphaFold3, used in Chai, Protenix, and other BioML structure-prediction models. Forward pass only — no gradients. GPU: H100. Reference impl: `/home/wychi/oss/wheels/gpumode/bioml/trimul/submission.py`.

**Why:** TriMul is a core BioML primitive; the einsum `'... i k d, ... j k d -> ... i j d'` dominates runtime (`2·B·S³·H` FLOPs). Optimization target is end-to-end TriMul speed across the 7 benchmark shapes below.

**How to apply:** When designing kernels or tile sizes, plan for these exact shapes — `H=128` is constant, `D ∈ {128, 384}`, `S ∈ {256, 512, 768, 1024}`, `B ∈ {1, 2}`, mask sometimes absent. The largest case is `B=1, S=1024, D=128`.

**Input:** `(input, mask, weights, config)`
- `input: [bs, seq_len, seq_len, dim]`
- `mask: [bs, seq_len, seq_len]` (may be all-ones when `nomask=true`)
- `weights`: dict of `norm.{weight,bias}`, `left_proj.weight`, `right_proj.weight`, `left_gate.weight`, `right_gate.weight`, `out_gate.weight`, `to_out_norm.{weight,bias}`, `to_out.weight`
- `config`: `{dim, hidden_dim}`

**Output:** `[bs, seq_len, seq_len, dim]` fp32

**Benchmark shapes (B, S, D, H, mask):**
| bs | seq_len | dim | hidden_dim | mask | distribution |
|----|---------|-----|------------|------|--------------|
| 2  | 256     | 128 | 128        | none | normal       |
| 1  | 768     | 128 | 128        | none | cauchy       |
| 2  | 256     | 384 | 128        | yes  | normal       |
| 1  | 512     | 128 | 128        | none | normal       |
| 1  | 1024    | 128 | 128        | none | cauchy       |
| 1  | 768     | 384 | 128        | yes  | normal       |
| 1  | 1024    | 384 | 128        | none | normal       |

**Spec link:** https://tinyurl.com/gpumode-trimul
