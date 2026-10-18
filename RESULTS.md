# Results — SGLang Triton Fusion on RTX 5060 Ti 16GB

Plain-language scoreboard for every experiment in this repo. Full
technical detail (methodology, failure modes, dead ends) lives in
[`docs/TRITON_KERNELS.md`](docs/TRITON_KERNELS.md).

## The hardware problem

An RTX 5060 Ti has 16GB VRAM and one memory controller. Serving a
7–9B model means streaming every weight once per decode step — the
GPU spends most of its time **waiting on VRAM**, not computing. So
the only two levers that matter:

1. **Shrink the weights** (quantize) → fewer bytes per step.
2. **Emit more tokens per weight read** (speculative decoding + CUDA
   graphs) → same bytes, more output.

## Headline journey (Qwen3-8B-FP8 decode bs=1)

```
vanilla sglang .................. 26.8 tok/s
+ fused quant+GEMV (§13) ........ 37.2   (+39%)
+ CUDA graph (§16) .............. 42.1
+ NGRAM spec-dec s4/d5 (§18) .... 50.2
+ NGRAM s7/d8 (§19, production) . 81.7   (3.0×)
```

Parallel throughput (4 concurrent requests): 116.7 → **281.4 tok/s**.

## Model comparison — which model on 16GB?

All numbers on the same card, same optimized stack (triton attention
+ CUDA graph + NGRAM s7/d8; prefill graph off for hybrid models).

| | Qwen3-8B-FP8 (§19) | Qwen3.5-9B-FP8 (§22) | Qwen3.5-9B-INT4 (§23) |
|---|---:|---:|---:|
| decode bs=1 | **81.7** tok/s | **57.3** tok/s | 49.2 tok/s |
| prefill | 2314 tok/s | **4056** tok/s | 2524 tok/s |
| parallel-4 aggregate | **281.4** | 68.0 (fake — serialized) | 145.6 |
| parallel-8 aggregate | — | impossible (max_running=1) | **192.4** |
| concurrent requests | 4+ | **1** | **8** |
| KV cache capacity | 18.8k tokens | 36.6k | **106.6k** |
| weights in VRAM | ~8.5 GB | 11.4 GB | **7.2 GB** |
| NGRAM accept len | 1.24–2.72 | 1.15–2.75 | 1.00–1.20 |
| best for | raw single-user speed | long-context single user | **multi-user serving** |

### How to read this table

- **8B-FP8 is still the speed king** (81.7 tok/s). Smallest weights →
  fastest weight streaming.
- **9B-FP8 is the long-context single-user pick**: fastest prefill by
  far (4056 — the hybrid linear-attention layers make prefill cheap),
  but its mamba state cache only fits 1 concurrent request on 16GB.
- **9B-INT4 is the multi-user pick** — this is the §23 experiment.
  GPTQ weights shrink 11.4 → 7.2 GB; the freed VRAM buys:
  - mamba slots 9 → **41** → `max_running_requests` 1 → **8**
  - KV cache 36.6k → **106.6k tokens**
  - real 8-way batching: **192.4 tok/s aggregate** vs FP8's ceiling
    of ~57 (it can only ever run one request at a time).
- Trade-off: INT4 single-stream is 14% slower than FP8 (49.2 vs
  57.3) and prefill is 38% slower — GPTQ's marlin kernels are less
  optimized on Blackwell than the native FP8 cutlass path, and our
  fused quant+GEMV patch (§13) does not apply to GPTQ.
- NGRAM still works on INT4 but accepts fewer draft tokens
  (1.00–1.20 vs 1.15–2.75) — quantization noise flips borderline
  tokens, so the verifier rejects more drafts.

## What transferred to INT4 (§23) — the architecture test

The question was: does the optimization stack we built for FP8 still
help once the weights are GPTQ-INT4?

| Lever | Transfers? | Evidence |
|---|---|---|
| NGRAM speculative decoding | **yes** | fires, accept 1.00–1.20 (lower, see above) |
| CUDA graph (decode) | **yes** | bs 1–8 captured, `cuda graph: True` in decode logs |
| Triton attention backend | **yes** | active, no regression |
| Fused quant+GEMV patch (§13) | **no** (predicted) | gate requires FP8 block-128 format; GPTQ runs through marlin |
| Prefill CUDA graph | **no** (same as §22) | hybrid GDN capture hangs — stays disabled |
| `--mamba-ssm-dtype bfloat16` | **required** (same as §22) | float32 default → 0 mamba slots → crash |

**Verdict: yes, the architecture transfers.** The quant-agnostic
stack (NGRAM + graph + triton attn) is what makes 8-way concurrent
serving possible on INT4; without it, freeing 4GB of weights would
not have turned `max_running_requests=1` into 8 usable slots.

## Experiment log (one line each)

| § | Experiment | Result |
|---|---|---|
| 1 | Argmax kernel | NEGATIVE — torch wins |
| 2 | Fused sampling | win 2–8× |
| 3 | Fused add+RMSNorm | win at decode |
| 4 | Fused SwiGLU+FP8 quant | big win |
| 6 | Fused RoPE | win 3.3–8× |
| 9 | Fused QK-norm+RoPE+GQA | win 10.9× (not wired — already fused upstream) |
| 10 | Fused quant+GEMV | win 1.8–4.9×, PASS on real logits |
| 11 | Batch decode bs=16 | NEGATIVE without TM-blocking |
| 12 | 4B → 8B server upgrade | SAFE, tok/s unchanged |
| 13 | **Wire quant+GEMV into server** | **+39% decode** |
| 15 | SwiGLU chain on server | NEGATIVE, reverted |
| 16 | **CUDA graph** | **42.1 tok/s, new config** |
| 17 | Attention-side fusion | rejected (roofline: decode is bandwidth-bound) |
| 18 | **NGRAM s4/d5** | **+19%** |
| 19 | **NGRAM s7/d8** | **81.7 tok/s = production** |
| 20 | External Wikipedia-ID corpus | NEGATIVE, reverted |
| 21 | Draft window 16 | NEGATIVE, reverted (draft 8 stays the sweet spot) |
| 22 | Qwen3.5-9B-FP8 hybrid | fits, 57.3 tok/s, but max_running=1 |
| 23 | **Qwen3.5-9B-GPTQ-INT4** | **8 concurrent, 192.4 tok/s aggregate** |

## Which config should I run?

- **Single user, max speed** → 8B-FP8
  `scripts/start_sglang_8b_ngram_w8.sh` — 81.7 tok/s decode.
- **Single user, long context / fast prefill** → 9B-FP8
  `scripts/start_sglang_35_9b_fp8.sh` — 4056 prefill tok/s.
- **Several users on one GPU** → 9B-INT4
  `scripts/start_sglang_35_9b_int4.sh` — 8 concurrent requests,
  192.4 tok/s aggregate, 106k-token KV cache.

## Reproduce

```bash
scripts/start_sglang_8b_ngram_w8.sh   # then:
benchmarks/bench_e2e.py               # decode bs=1
benchmarks/bench_prefill.py           # prefill tok/s
benchmarks/bench_parallel.py          # 4-way aggregate
benchmarks/bench_parallel8.py         # 8-way aggregate (§23)
```
