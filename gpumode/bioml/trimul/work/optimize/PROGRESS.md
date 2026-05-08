# Optimization Progress

Source kernel: `work/hopper_gemm_ws.py`. Target shape: #6 (B=1, S=1024, D=384). All times = `--bench` median, 10 iters, warmup=3, on H100 (CUDA_VISIBLE_DEVICES=4).

## Benchmark Table (shape 6 unless noted)

| Iter | Variant | shape6 ms | speedup | Notes |
|------|---------|-----------|---------|-------|
| 0 | baseline (`work/hopper_gemm_ws.py`) | 9.66 | 1.00× | profile in `profile/e2e_results.md` |

## Per-iteration log

(filled in below as iterations complete)
