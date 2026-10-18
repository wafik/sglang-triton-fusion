# Morning Report — Night Session 17–18 Oct 2026

Summary of all experiments from this session. Full technical details:
[`TRITON_KERNELS.md`](TRITON_KERNELS.md) §8–§21.

## TL;DR

1. **Fused quant+GEMV PASS** (§10) — wins 1.8–4.9× at bs=1; the rel-13.17
   debug saga ended in 3 bugs/insights (different quant formula, broken
   maxrel metric, tolerance must be computed).
2. **Lost at batch bs=16** (§11a) → root cause: row-parallel without
   weight reuse (lm_head 15.3ms = 16×973µs) → **TM-blocking fix** (§11c):
   lm_head M=16 back to 960µs. Remaining small layers: cutlass still wins —
   decision: fusion = bs=1 only.
3. **Fused SwiGLU+quant+down-GEMV** (§11b): 1.95× at bs=1 → projection
   saves ~2.3ms/step.
4. **Server upgraded 4B → Qwen3-8B-FP8** (§12, user request): **SAFE** —
   VRAM 12.8/16.3GB, RAM 4.5Gi, and tok/s nearly identical
   (26.9 vs 27.7 single; 120.9 vs 116.7 parallel-4). **8B = free upgrade.**
5. **★ E2E INTEGRATION SUCCESS** (§13): quant+GEMV wired into the real
   server via monkeypatch → **26.8 → 37.2 tok/s (+39%), decode
   −28% ms/token (37.4→26.9)**, parallel batch no regression (120.3≈120.9),
   coherent output. Early milestone — surpassed by §16/§18/§19 below.
6. **SwiGLU chain (§15) NEGATIVE** — projection −2.3ms from vanilla
   baseline; on top of §13 the remaining saving was only silu (~0.5ms,
   noise). Chain fired 100% but 27.14 vs 26.91 ms/token. **Reverted.**
   3-way A/B quality test proved echo = raw Qwen3 greedy behavior, not
   the patch.
7. **★ CUDA graph + §13 = NEW CONFIG** (§16): decode **23.78 ms/token
   / 42.1 tok/s** (graph ON), parallel-4 **176.1 tok/s (+46%)**.
   Total vs the 4B baseline: **27.7 → 42.1 tok/s decode**, **116.7 →
   176.1 parallel**.
8. **Roofline discovered** (§17): 8B decode = **≈357 GB/s effective vs
   358 GB/s measured D2D** → already at the DRAM ceiling. QK-norm turned
   out to be already fused by sglang; attention-side fusion rejected
   (KB-scale savings vs 8.5GB weights). **Ceiling = weight reads; the way
   to win = speculative decoding** → tried (§18 NGRAM).
9. **★ NGRAM spec-dec WINS** (§18): **50.2 tok/s decode (+19%)**,
   parallel **244.6 (warm 280.7, +39%)**, accept len 1.5–2.5 per verify-5.
   No draft model needed.
10. **★★ Wider window (§19, steps7/draft8)**: single **63.7 cold →
    81.7 tok/s warm**, parallel warm 281. **Total vs 4B baseline:
    27.7 → 81.7 tok/s (+195%, 3×); parallel 116.7 → 281.4 (2.4×).**
    Wider verify is nearly free when bandwidth-bound.
11. **External Wikipedia-ID corpus (§20) NEGATIVE** — SAM budget capped
    at ≤ draft−1 (7 nodes) → wiki barely used and actually displaced the
    self-ngram: accept 1.5 vs 2.7, 54.6 vs 81.7 tok/s. Reverted.
12. **Draft>8 (§21, candidate 9b) NEGATIVE** — steps15/draft16: accept
    len rose (1.9–3.1) but tok/s dropped (71.3 vs 81.7 single, 220.6
    vs 281.4 parallel). Verify graph at 16 was fine (+0.11GB). Sweet-spot
    draft remains 8. Reverted.
13. Fused sampling revalidated on real 8B logits: TV 0.0012 PASS.

## Kernel proof — 4B & 8B (summary)

**1. Micro-benchmarks (bs=1, µs per call) — kernel logic proven correct:**

| Kernel | Baseline | Fused | Speedup | Accuracy |
|---|---:|---:|---:|---|
| quant+GEMV `q` | 138 µs | 41 µs | **3.38×** | PASS 4B & 8B |
| quant+GEMV `gate_up` | 271 µs | 55 µs | **4.91×** | PASS 4B & 8B |
| quant+GEMV `lm_head` | 953 µs | 937 µs | 1.02× | PASS |
| SwiGLU+quant+GEMV (§11b) | 132 µs | 68 µs | **1.95×** | PASS |
| QK-norm+RoPE+GQA (§9) | 112 µs | 10 µs | **10.85×** | PASS 4B & 8B |
| Sampling real-logits | 8–16 µs | 1–2 µs | **2–8×** | TV 0.0211/0.0012 |

All fused kernels were tested with **real logits from both checkpoints**
(not random tensors).

**2. Real e2e server — user-facing impact (tok/s decode bs=1):**

| Stage | 4B | 8B | Proof |
|---|---:|---:|---|
| Vanilla | 27.7 | 26.8 | baseline |
| + fused quant+GEMV (§13) | — | **37.2** | **+39%** = kernel wins on server |
| + graph (§16) | — | 42.1 | kernel still active inside graph capture |
| + NGRAM w8 (§19) | — | **81.7** | final production |

Micro 5× → e2e +39% is expected: on the server, GEMV is only ~40% of
decode time (the rest is attention/sampling/graph). What matters is
**the direction went up in an A/B on the same server** = proven, not a
paper benchmark. The 11× QK-norm does not count for e2e because sglang
already fuses it elsewhere (§17).

## Kernel scoreboard (bs=1 decode unless stated otherwise)

| Experiment | Result | Status |
|---|---|---|
| Fused QK-norm+RoPE+GQA (§9) | decode **10.85×**, prefill 4.06× | big win |
| Fused sampling real-logits 4B/8B | TV 0.0211 / 0.0012 | validated |
| Fused quant+GEMV (§10) | q 3.38× · gate_up **4.91×** · lm_head 1.02× | wins at bs=1 |
| Fused SwiGLU+quant+GEMV (§11b) | **1.95×** (132→68µs) | wins at bs=1 |
| quant+GEMV M=16 row-parallel (§11a) | 0.06–0.5× | **NEGATIVE** |
| quant+GEMV TM-16 blocked (§11c) | lm_head M=16 15.8× fix; middle mixed | partial fix |
| **E2E integration on 8B server (§13)** | decode **37.4→26.9 ms/token (+39% tok/s)** | **big win** |
| SwiGLU+quant+chain on server (§15) | 27.14 vs 26.91 ms/token | **NEGATIVE** (reverted) |
| **CUDA graph + §13 (§16)** | **42.1 tok/s · parallel 176** | **new config** |
| QK-norm+RoPE fusion (§17) | rejected: already fused in sglang + KB vs 8.5GB | roofline analysis |
| 8B decode roofline (§17) | 357 GB/s effective ≈ 358 GB/s measured | **DRAM ceiling** |
| **NGRAM spec-dec (§18)** | **50.2 tok/s · parallel 244.6** | **big win** |
| **NGRAM wide window s7/d8 (§19)** | **63.7–81.7 tok/s single** | **big win** |
| External Wikipedia-ID corpus (§20) | 54.6 tok/s · accept 1.5 · parallel 199 | **NEGATIVE** (reverted) |
| Draft>8 w16 (§21) | 71.3 tok/s · accept 1.9–3.1 · parallel 220.6 | **NEGATIVE** (reverted) |

## Server numbers

| | 4B-FP8 | 8B-FP8 |
|---|---|---|
| VRAM | 7.8 GB | ~13.4 GB + graph/ngram (~2.6 left) |
| KV tokens | ~30k | 18.8k |
| decode bs=1 | 27.7 tok/s | **81.7 tok/s warm** (w8 ngram+graph+§13; vanilla 26.8) |
| 4-way parallel aggregate | 116.7 | **281.4** (warm) |

## Lessons learned (most valuable)

1. **Every constant must be an argument/constexpr** — QMAX/EPS cost 4× in total.
2. **maxrel is a broken metric** for GEMV (cancellation → tiny
   denominator); use maxabs in units of bf16 ulp at the max magnitude.
3. **GEMV-fusion ≠ GEMM-fusion** — winning at M=1 can become a 16× loss
   at M=16 if the program doesn't reuse weights; TM-blocking is mandatory.
4. **Tolerance is computed from measured components** (self-vs-exact,
   cutlass-vs-exact, double-rounding), not guessed.
5. **8B is free on the 5060 Ti** — step time is dominated by overhead that
   doesn't scale with weight size.

## Next steps (recommendations)

1. ~~**E2E integration**~~ ✅ **DONE (§13)**: +39% tok/s on 8B, batch safe.
2. ~~**SwiGLU chain**~~ ✅ **TRIED (§15)**: gain 0 on top of §13, reverted.
3. ~~8B prefill~~ ✅ **MEASURED (§14)**: **2314 tok/s input** — an 8k-token
   doc ≈ 3.5s TTFT; the patch doesn't touch prefill (M>1 by design).
4. ~~**CUDA graph**~~ ✅ **WIN (§16)**: 42.1 tok/s + parallel 176.
5. ~~Attention-side fusion~~ ✅ **REJECTED (§17)**: already fused in sglang;
   decode is at the DRAM roofline (357 ≈ 358 GB/s).
6. ~~**NGRAM spec-dec**~~ ✅ **WIN (§18)**: 50.2 tok/s + parallel 244.6.
7. ~~**Wider ngram window**~~ ✅ **WIN (§19)**: steps7/draft8 →
   **63.7–81.7 tok/s**. Final production = `start_sglang_8b_ngram_w8.sh`.
8. ~~**External corpus**~~ ✅ **NEGATIVE (§20)**: SAM budget ≤7 nodes
   displaced the self-ngram → harmful. Reverted.
9. Next-tick candidates (by value):
   a. EAGLE draft (download ~0.6B ckpt) — accept-rate much smarter than
      ngram; but check first: sglang has no official qwen3+eagle path
      (grep empty), no ckpt in /root/models → research before committing.
   b. ~~Draft window >8~~ ✅ **NEGATIVE (§21)**: w16 → 71.3 tok/s
      (vs w8 81.7). Draft sweet-spot = 8.
   c. ~~Polish documentation~~ ✅ **DONE**: 4B/8B kernel-proof tables in
      MORNING_REPORT; §1–§21 complete in TRITON_KERNELS.md.

## System state

- 8B server **patched + graph + NGRAM w8** running on :30000
  (`start_sglang_8b_ngram_w8.sh`, pid in `/root/sglang.pid`). Fused
  quant+GEMV (§13) + CUDA graph + ngram steps7/draft8 active.
  Fallbacks: `start_sglang_8b_ngram.sh` (s4/d5), `start_sglang_8b_graph.sh`
  (no spec, minimal TTFT 33ms). Full revert: `fp8_utils.py.orig13`
  + `start_sglang_8b.sh`.
- Environment fix: symlink `libcuda.so` → CUDA stubs (flashinfer JIT
  sampling needs `-lcuda` on the first top_p request).
- 4B is available in the HF cache; just revert scripts if needed.
- All experiment files in `/root/*.py`, local copies in
  `D:\kerjaan\belajar\llm\`. No processes that need to be spared;
  Zed/Claude untouched.
