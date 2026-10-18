# sglang-triton-fusion

Triton kernel fusion experiments for **SGLang** on consumer GPUs, targeting
**Qwen3-4B/8B-FP8** served on an **RTX 5060 Ti 16GB** (Blackwell sm_120,
WSL2). Everything here was measured end-to-end on a live server, not just
micro-benchmarks.

**Result: 27.7 → 81.7 tok/s decode (3×) on Qwen3-8B-FP8**, achieved by
stacking fused FP8 quant+GEMV kernels, CUDA graphs, and NGRAM speculative
decoding. Parallel throughput (4 requests): 116.7 → 281.4 tok/s (2.4×).

## What's inside

| Path | Contents |
|---|---|
| `kernels/` | Standalone Triton kernels (quant+GEMV, SwiGLU, RoPE, QK-norm, sampling, argmax) |
| `patches/` | Monkeypatch that wires fused quant+GEMV into sglang's FP8 linear path (M==1 decode only) |
| `benchmarks/` | Micro-benchmarks, accuracy checks (real-logits), e2e server benchmarks |
| `scripts/` | Server launch configs (production = `start_sglang_8b_ngram_w8.sh`, plus experiment/negative-result variants) |
| `tools/` | Utilities (e.g. NGRAM external-corpus prep) |
| `docs/` | Full experiment journal (`TRITON_KERNELS.md`), morning report (`MORNING_REPORT.md`), original plan |

## Headline results

### Server e2e (Qwen3-8B-FP8, decode bs=1)

| Config | tok/s | Notes |
|---|---:|---|
| Baseline (vanilla sglang) | 26.8 | |
| + fused quant+GEMV (M==1) | **37.2** | +39% |
| + CUDA graph | 42.1 | |
| + NGRAM spec-dec (s4/d5) | 50.2 | |
| **+ NGRAM (s7/d8) = production** | **81.7** (warm) | **3× baseline** |
| Parallel 4-req (production) | **281.4** (warm) | 2.4× baseline |

### Kernel micro-benchmarks (bs=1, PASS on both 4B and 8B real logits)

| Kernel | Speedup |
|---|---:|
| Fused quant+GEMV (`gate_up`) | **4.9×** |
| Fused quant+GEMV (`q`) | 3.4× |
| Fused QK-norm+RoPE+GQA | **10.9×** (already fused upstream; not wired) |
| Fused SwiGLU+quant+GEMV | 1.95× |
| Fused sampling | 2–8× |

### Negative results (kept for honesty)

- SwiGLU-chain integration on top of quant+GEMV: **no gain**, reverted.
- NGRAM draft window 16 (vs 8): accept length up, throughput **down** (71.3 vs 81.7 tok/s).
- External NGRAM corpus (Wikipedia ID): budget cap ≤ draft−1 hurts more than it helps.
- quant+GEMV at batch 16 without TM-blocking: 0.06–0.5× (GEMV-fusion ≠ GEMM-fusion).

## Production recipe

```bash
# server (RTX 5060 Ti, ~13.4 GB VRAM used)
scripts/start_sglang_8b_ngram_w8.sh
# = Qwen3-8B-FP8 + attention-backend triton
#   + fused quant+GEMV monkeypatch (patches/patch_fp8_fused_tail.py)
#   + CUDA graph (decode bs 1..8)
#   + NGRAM speculative decoding, steps=7 draft=8
```

Key insight from the roofline analysis: 8B decode streams ~8.5 GB of
weights per step and was already at the DRAM ceiling (~357 GB/s effective
vs 358 GB/s measured). Kernel fusion helped by cutting per-step overhead;
beyond that, the only lever left is **reading fewer weight bytes per
output token** — i.e. speculative decoding, where a wider verify pass is
nearly free under bandwidth-bound decode.

## Requirements

- NVIDIA GPU with CUDA 12.8+ (tested: RTX 5060 Ti, sm_120)
- Python 3.12, PyTorch 2.x, `sglang` (tested against sglang with
  flashinfer/cutlass FP8 block-linear paths), `triton`
- Model: [Qwen3-8B-FP8](https://huggingface.co/Qwen/Qwen3-8B-FP8)
  (4B-FP8 also validated for kernel accuracy)

## Docs

- [`docs/TRITON_KERNELS.md`](docs/TRITON_KERNELS.md) — full experiment
  journal, section by section (§1–§21), including all failed attempts.
- [`docs/MORNING_REPORT.md`](docs/MORNING_REPORT.md) — summary report
  with scoreboard and recommendations.
- [`docs/PLAN.md`](docs/PLAN.md) — original environment/setup plan.

## License

MIT
