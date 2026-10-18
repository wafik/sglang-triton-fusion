# Custom Triton Kernel Experiments — RTX 5060 Ti (sm_120)

Summary of Triton kernel experiment results in WSL2, a continuation of PLAN.md.
All benchmarks: **warmup 20 + 100 iterations, `torch.cuda.Event`** (not
`time.time()` — the 345µs figure in PLAN.md turned out to be cold-start/lazy-init).

Standard shape: **B=16, V=151936, fp32** (≈ 9.7MB, fits in the 5060 Ti's L2).

---

## 1. Argmax standalone — NEGATIVE, torch wins

File: `triton_argmax.py`, `test_two_stage_argmax.py`

| Variant | Steady-state | vs torch |
|---|---|---|
| `torch.argmax` (CUB DeviceReduce) | **16.09 µs** | 1x |
| Triton two-stage (BLOCK=8192, partial buffer) | 30.65 µs | 0.52x |
| Triton single-launch (loop per row) | 19.28 µs | 0.83x |
| Two-stage à la other-agent (128 chunk) | 51.79 µs | 0.36x |

**Conclusion:** the 9.7MB tensor fits in L2, torch CUB is already near optimal.
Both lose because of launch overhead + the partial-buffer round trip.
The "two-stage argmax 345µs" suggestion from another agent was refuted with their
own harness (with warmup: the baseline drops to ~18µs, the kernel 51µs).

**Lesson:** don't optimize a lone reduction — look for **fused multi-op**.

---

## 2. Fused sampling — POSITIVE, wins 2–8x

File: `fused_sampling.py` (deploy: `/root/fused_sampling.py`)

### Design
Naive: full-vocab `softmax` (materializes 9.7MB) → sort/topk → top-p mask →
renormalize → `multinomial`. Fused:

1. `torch.topk(logits, 50)` straight on the raw logits — softmax is monotone,
   the order of topk(logits) ≡ topk(probs), **the full softmax pass is SKIPPED**.
2. One small Triton kernel (grid = rows): `exp((v−v0)/T)` → top-p cumsum →
   renormalize → **inverse-CDF sample** (`tl.rand` + `argmax` on cum ≥ u).

### Results (K=50, top_p=0.9, T=0.7)

| Pipeline | µs |
|---|---|
| Naive full-vocab (softmax + sort V) | 1385.29 |
| Naive competent (full softmax → topk) | 337.25 |
| **Fused** | **167.18** |

**8.29x** vs naive full · **2.02x** vs naive competent.
Breakdown: `torch.topk` ≈ 170µs (dominant), the Triton kernel itself a few µs.

### Verification
- `top_k=1` ≡ `torch.argmax` (deterministic) ✅
- 50 seeds × 16 rows: results always in the top-k **and** nucleus set (HF
  convention: keep_i iff cumsum-before-i ≤ top_p) ✅
- Statistics: 5000 samples, V=128, top_p=1 → `max|freq − p| = 0.0109`
  (tolerance 0.03) ✅

### Bug caught by the test
The early `keep = (cum − w) <= top_p` version compared the **unnormalized**
cumsum (total `w` can be ≫ 1) against raw `top_p` → wrong mask (distribution
error 0.29). Fix: `keep = (cum − w) <= top_p * total`. Example of why that
statistical check is mandatory — the deterministic/membership checks still pass
even when the mask is wrong.

### Next steps (if desired)
`torch.topk` is now the bottleneck; a custom radix-select is hard to anchor to
CUB (same as argmax). Diminishing returns — stop here.
The next fused candidates that still have room to win:
RMSNorm+residual, SwiGLU, quantization (all multi-op, not single reductions).

---

## 3. Fused add+RMSNorm — wins at decode

File: `fused_rmsnorm.py`. Qwen3-4B contract: hidden=2560, eps=1e-6, 36 layers.
Fused: 1 kernel → `out = rmsnorm(x+r)*w` + `residual_new = x+r`.
Naive: `x + residual` (materialize) → `F.rms_norm` (2 launches + intermediate).

| Shape | Naive | Fused | Speedup |
|---|---|---|---|
| decode (16, 2560) | 45.76 µs | 26.72 µs | **1.71x** |
| prefill (512, 2560) | 19.17 µs | 15.99 µs | 1.20x |

Accuracy: allclose vs `F.rms_norm` (max err ~1e-7) ✅.
Note: wins by eliminating the intermediate materialization — at decode
(small) launch-count wins, at prefill memory traffic wins.

---

## 4. Fused SwiGLU + FP8 E4M3 quant — BIG WIN

File: `fused_swiglu_fp8.py`. Contract verified from `config.json`:
`intermediate_size=9728` (not 2560/5120), hidden=2560, 36 layers.
Quantization groups of 128 elements → 76 groups/row. Quant grid `(B, 76)` —
absmax **local per group**, no cross-program reduce.

Naive chain (not a strawman): `silu*up` materialized FP32 → `amax` per group →
`scale=max/448` → `y/scale` clamp → cast FP8 (≈6 launches + 2 intermediates).
Fused: 1 kernel — `silu(g)*u` + absmax + scale + cast FP8, all in registers.

| Shape | Chain | Naive | Fused | Speedup |
|---|---|---|---|---|
| decode (16, 9728) | SwiGLU only | 14.55 µs | 16.14 µs | 0.90x (narrow loss) |
| decode (16, 9728) | SwiGLU+FP8 | 116.99 µs | 49.71 µs | **2.35x** |
| prefill (512, 9728) | SwiGLU only | 66.91 µs | 23.42 µs | **2.86x** |
| prefill (512, 9728) | SwiGLU+FP8 | 542.19 µs | 54.30 µs | **9.99x** |

Verification: scale allclose (rtol 1e-5) · dequant error ≤ half an E4M3 ulp ·
**FP8 output byte-identical 100%** vs baseline on all shapes ✅.

### Test lessons (2 test bugs; the kernel passed both)
1. `QMAX` global cannot be accessed by `@triton.jit` → must be an argument/constexpr.
2. The error bound assumed a `scale/2` grid (int8 grid) — E4M3 is a **float**
   grid (3-bit mantissa); the correct bound = `|y|/16` (half a relative ulp).
   The early test failed on 63k elements even though the fp8 output was 100%
   identical — the test itself was wrong.

### Strategic conclusion
- The fusions that win are the ones that **eliminate intermediate
  materialization** or **merge a chain of many launches** (the quant chain is the
  longest chain, biggest win 2.35–9.99x).
- Pure SwiGLU at small decode (16 rows) ≈ a tie — torch's 2 small kernels are
  already fast; the win shows up at prefill. The "MLP 90% of the time" claim
  needs end-to-end testing, and activation-Q8 in the current Qwen3-FP8 serving
  is *weight-only* — this harness applies to the W8A8 pattern, not as a direct
  plug-in for the server.

---

## 5. Stress battery (26 checks, all PASS)

File: `stress_tests.py`. Edge cases across all three kernels:

- **sampling (15)**: reproducible seeds · K=4096 support outside the top-50 ·
  top_p=0 ≡ top-1 · T=0.01+margin5 ≡ deterministic argmax · T=1000 still
  top-k · non-power-of-2 K (1,3,17,51,63,77) · input cast bf16 · duplicate rows ·
  logit scale 1e4 (anti exp overflow).
- **rmsnorm (4)**: zero rows · extremes 1e6 · non-contiguous · weight=1 ≡ manual.
- **swiglu_quant (7)**: byte determinism · all-zero (scale floor 1e-30) ·
  saturation touching exactly 448 · gate=−50 silu→0 finite · extremes ×10 fp8
  identical ≥99.9% · determinism.

### Findings from the battery (not kernel bugs — TEST bugs)
1. **K=V (151936) cannot be tested**: BLOCK=next_pow2(V)=262144 → Triton
   compilation >10 minutes of CPU. Skipped, used K=4096. Practical BLOCK limit: ≤16384.
2. **T=0.01 on random logits is indeed NOT argmax** — the top1/top2 gap ~0.01,
   e^(−gap/T)≈0.3 is still possible. The correct test: artificial margin (+5) → deterministic.
3. **bf16→fp32 round-trip ≠ original fp32** — equality is impossible; the correct
   check is membership in the cast result values.
4. Python `for seed in range(50)` calling the single kernel is still fine (<1s).

---

## 6. Fused RoPE — wins 3.3x / 8x

File: `fused_rope.py`. Contract: Qwen3-4B `head_dim=128`, 32 heads (config.json).
Naive = HF eager: `cat(cos,cos)` + `cat(sin,sin)` + neg + `cat` rotate_half +
mul + mul + add ≈7 launches + 3 intermediates. Fused: 1 launch/token —
`out = x*cos + sign(half) * mirror(x) * sin`, mirror via `offs ^ half`
(bit-flip of the second half's positions).

| Shape | Naive | Fused | Speedup |
|---|---|---|---|
| decode (16,32,1,128) | 131.89 µs | 39.74 µs | **3.32x** |
| prefill (4,32,1024,128) | 1718.84 µs | 214.63 µs | **8.01x** |

Accuracy: allclose bf16-scale · norm-invariance · pos-0 identity · bitwise
100% on the main run.

### Real bug caught by the test (the kernel was wrong at first)
First kernel version: `x1=where(offs<64,x,0); x2=where(offs>=64,x,0)` —
**the cross-halves term was missing** (`out_lo = x1*cos − 0*si`, should be
`− x[i+64]*si`). The S=1 case "passed" because sin(0)=0 → identity, so even the
decode test proved nothing. Fix: load the mirror value `x[offs ^ 64]`.
**Lesson: a test that passes due to a trivial case ≠ correct — prefill is mandatory.**

### FMA vs 3-rounding (not a bug)
Residual mismatch ≤1 bf16 ulp at boundary elements: Triton contracts `x*co+xm*si`
into an FMA (1 rounding) vs torch mul+mul+add (3 roundings). The fused version is
**more accurate**; the tolerance is at bf16 scale (rtol 8e-3), not fp32.

---

## 7. End-to-end server test (real A/B, port 30000)

File: `bench_e2e.py`, `bench_parallel.py`, `start_sglang_graph.sh`,
`start_sglang_fi.sh`. Method: client wall-clock — `max_new_tokens=1`
(≈prefill) vs `512`; `(t512 − t1)/511` = ms/token decode. Median of 3 runs.

| Variant | Decode tok/s | Conclusion |
|---|---|---|
| Baseline (triton, graph off) | **27.7** | reference |
| + CUDA graph active (capture bs 1–8 OK) | 27.3 | **no difference** |
| + flashinfer backend (warm) | 27.5 | **no difference** (first run 19.7 = JIT compile) |
| **4 parallel requests (aggregate)** | **116.7** | **4.2x linear!** |

### Findings
1. **Decode at bs=1 is purely weight-bandwidth-bound.** 4B FP8 ≈ 4.5GB / 36ms
   ≈ 125GB/s effective (5060 Ti peak ~448GB/s). Raising the batch → weights are
   shared: 4 parallel = 4.2x aggregate, **time per request identical** (9.4s vs 9.2s).
   Attention/KV is not a factor at short context.
2. **CUDA graph & flashinfer add no tok/s here** — because it isn't
   launch/attention-bound. The Triton backend is still used (canonical config).
   The server was reverted to `start_sglang.sh` after the A/B.
3. Version note: sglang 0.5.22 `--cuda-graph-backend-decode {full,...}`;
   the legacy arg key `cuda_graph: False` in the log is MISLEADING — what's
   active is `cuda_graph_config` + log evidence `Capture target decode CUDA graph end`.
4. `/metrics` is 404 in this build; `get_server_info` works.

### Implications
- Single-token latency (27.5) is a **bs=1 sweep** — it cannot be raised
  by attention/sampling fusion kernels; what rises is **throughput via batch**
  (SGLang's continuous batching exploits it under load).
- Our fusion kernels (RoPE/SwiGLU/RMSNorm/sampling) remain relevant for
  **reducing per-forward-pass time** — proportional to that 36ms —
  especially as the launch/traffic elements shrink in each layer.

---

## 8. Night-3 iteration: real-logits validation + autopsy of 125GB/s

### 8a. Fused sampling on the server's REAL logits — PASS

File: `real_logits_sampling.py`. Fetch real top-50 logprobs via
`/generate` + `return_logprob` + `top_logprobs_num: 50`, build the sparse row
(1, 151936), run `fused_sample` on it.

| Check | Method | Result |
|---|---|---|
| 1 support | 20 seeds, top_k=50, top_p=0.9 | all samples ∈ nucleus (22 tokens) ✅ |
| 2 distribution | 5000 samples vs theory + multinomial | TV(fused, theory)=**0.0211**, TV vs multinomial=**0.0440** (tol 0.05) ✅ |

**2 TEST bugs caught** (the kernel passed both):
1. Wrong API key: the response uses `output_top_logprobs` (a list per step,
   entries `[logprob, token_id, text]`), not the `meta_info["top_logprobs"]` dict.
2. TV=1.0 briefly appeared: `kept = order[keep]` is an **index into the top-50,
   not a token id** — the theory mass was placed at vocab positions 0..49 →
   totally disjoint support. Fix: `ref_p[toks[kept]] = probs[kept]`.
→ Fused sampling is officially validated on real logit distributions, not just synthetic ones.

### 8b. Autopsy of 125GB/s: naive dequant is NOT the server path

File: `bench_fp8_gemv.py` (weight-only simulation: dequant materialize → GEMV).

| Layer | dequant | GEMV bf16 | chain (GB/s) |
|---|---|---|---|
| q_proj 2560×4096 | 577.9 µs | 56.8 µs | 727.8 µs (14) |
| gate_up 2560×9728 | 1348.7 µs | 121.4 µs | 1819.4 µs (13) |
| down 9728×2560 | 1358.9 µs | 134.4 µs | 1816.4 µs (13) |
| lm_head 2560×151936 | 21458.8 µs | 1898.6 µs | 29066.9 µs (13) |

Chain at 13–14 GB/s ≪ 125GB/s server → **the server can't possibly be using
materialize-dequant per step** (decode steps would be ~300ms, not 36ms).
`torch._scaled_mm` raises a RuntimeError on sm_120 with 2D blockwise layout.

### 8c. The server's real path: W8A8 blockwise — here are the actual numbers

File: `bench_sglang_fp8.py`. Checkpoint verified: `weight_scale_inv`
**2D blockwise 128×128** (`[N/128, K/128]`, e.g. gate `[76, 20]`). Path:
`sglang_per_token_group_quant_fp8_row_padded` + `fp8_blockwise_scaled_mm`.

| Layer (N×K) | quant act | fp8 gemv | GB/s | bf16 gemv | fp8/bf16 |
|---|---|---|---|---|---|
| q_proj 4096×2560 | 46.5 µs | 43.4 µs | 232 | 56.9 µs | 1.31x |
| o_proj 2560×4096 | 41.5 µs | 17.4 µs | 579 | 40.5 µs | 2.33x |
| gate_up 9728×2560 | 45.7 µs | 28.7 µs | **833** | 117.9 µs | **4.11x** |
| down 2560×9728 | 39.8 µs | 33.9 µs | 705 | 111.4 µs | 3.29x |
| lm_head 151936×2560 | 40.6 µs | 951.8 µs | 392 | 1880.5 µs | 1.98x |

Honest note: GB/s figures >448 (peak GDDR7) on gate_up/o_proj/down =
the weights fit in the **32MB L2** during repeated microbench runs — the honest
steady-state DRAM figure is only lm_head (389MB): **392 GB/s fp8**.

### Major finding: where did the 36ms/step go?

Rough calculation per decode step (bs=1): total fp8 GEMV ~6–7ms (DRAM-honest),
lm_head ~1ms, the rest **~25ms is not GEMV**. Largest measured items:

- **Activation quantization per-linear: ~40µs/fan-call × 6 calls/layer × 36 layers
  ≈ 8.6ms ≈ 24% of the decode step** — almost always launch-bound (M=1, small
  work, fixed cost ~40µs).
- Attention, RMSNorm, RoPE, sampling, launch tail: the remainder.

→ **125GB/s effective ≠ the GPU is saturated on bandwidth.** DRAM headroom ~3.6x; the
bottleneck is launch/quant overhead. The strategic conclusion changes:
1. Our fusion kernels (RoPE 3-8x, RMSNorm, sampling) **are relevant** — they do
   strike at this overhead layer, provided they're wired into the graph.
2. Next largest single savings target: **quantize once per
   input hidden for {q,k,v}** (3 calls → 1) ≈ saves 4.3ms ≈ +12% tok/s —
   belongs to sglang integration, not a new kernel.
3. Server path confirmed as W8A8 (not weight-only); our SwiGLU+FP8
   harness is aligned with this quant+GEMM pattern.

#### Correction (after reading sglang's `qwen3.py`) — `bench_quant_fuse.py`

Already fused in sglang: `QKVParallelLinear` (1 quant for q+k+v) +
`gate_up_proj` (1 quant for gate+up) → **4 quant calls/layer, not 6**, and each
call's input differs (hidden / attn_out / ffn_in) → can't be merged again
at the Python level. Measured per call (bs=1): qkv 39.8 · o 42.3 · gate_up 39.2 ·
down 40.7 µs → **5.83 ms/step = 16.2% of decode** (corrected from 8.6ms/24%).
Overhead is nearly fixed at ~40µs (launch-bound), at bs=16 only 23.8µs.
→ The real target: **fuse quant into the GEMM kernel** (eliminate the separate
launch) — see experiment 10.

---
## 9. Fused QK-norm + RoPE + GQA expand — wins 4–11x

File: `fused_qk_rope.py`. Qwen3-4B contract: 32 q-heads, **8 kv-heads (GQA 4:1)**,
D=128, rms_norm_eps=1e-6. HF eager per layer: `rms_norm` q (materialize) →
rope q (≈7 launch) · `rms_norm` k → rope k → `repeat_interleave(4)` (materialize
4x). Fused: **1 launch q + 1 launch k** — grid in the output head space,
`kv_h = h // 4` read on-the-fly; mirror RoPE uses the **already-normalized**
values (ms same per head, w[i^half]).

| Shape | Naive | Fused | Speedup |
|---|---|---|---|
| decode (16, S=1) | 312.77 µs | 28.84 µs | **10.85x** |
| prefill (4, S=1024) | 2592.81 µs | 638.18 µs | **4.06x** |

Verification: allclose q & k vs eager · pos-0 ≡ pure rmsnorm (identity rope) ·
4 GQA group heads identical · post-rope norm-invariance ✅.

### Lessons
- **Another kernel bug (global eps)** — same as QMAX: a global `EPS` cannot be
  accessed by `@triton.jit` → pass `eps` as an argument. (Nth time —
  this pattern is now second nature: all constants must be arguments/constexpr.)
- Prefill briefly failed (0.03125): **not a bug** — naive rounds the norm to
  bf16 *before* rope (2 roundings), we are full fp32; cancellation elements
  (out≈0 from terms ~4) carry 1 bf16 ulp as absolute error. atol 5e-2.
- Decode run-to-run 6.3–10.9x (clock/thermal noise on WSL2) — direction holds:
  big wins, because materialization + GQA expand in intermediate is eliminated.
- GQA expand is not "free" in eager: `repeat_interleave` = copy 4x the size of K —
  in the fused kernel it disappears entirely (read kv head once, write output).

---

## 10. Fused quant+GEMV (1 launch) — wins 1.8–4.9x, PASS on all shapes

File: `fused_gemv_quant.py`, debug: `dbg_gemv_quant.py`, `dbg_cutlass_tol.py`,
`dbg_o_proj.py`. Attacking **5.83ms/step (16.2% of decode)** from 4 quant
launches/layer (see §8 correction). The sglang path = `row_padded quant` (launch 1)
+ `fp8_blockwise_scaled_mm` cutlass (launch 2); fused = 1 Triton kernel —
absmax per group-128, fp8 cast in registers, fp32 accumulation with scale
x × scale w (blockwise 128×128 from the checkpoint).

| Layer (N×K) | sglang (quant+gemm) | fused 1 kernel | Speedup | maxabs vs ref |
|---|---|---|---|---|
| q_proj 4096×2560 | 70.5 µs | **20.8 µs** (483 GB/s) | 3.38x | 0.50 ulp |
| o_proj 2560×4096 | 62.9 µs | **32.0 µs** (315 GB/s) | 1.97x | 1.50 ulp |
| gate_up 9728×2560 | 142.5 µs | **29.0 µs** (823 GB/s) | 4.91x | 1.00 ulp |
| down 2560×9728 | 117.1 µs | **64.6 µs** (370 GB/s) | 1.81x | 0.50 ulp |
| lm_head 151936×2560 | 969.6 µs | 952.1 µs (392 GB/s) | 1.02x | 0.00 ulp |

Honest caveat: q/gate_up/down are small → they fit in **L2** during microbench (the
833 GB/s figure exceeds peak DRAM) — the DRAM-honest one is lm_head (1.02x, pure bandwidth).
The real win = **eliminating the quant launch (~40µs separate)** + computing
in a single pass; in DRAM-bound decode, the projections save ≈ 4 calls × 36 layers
× 40µs ≈ 5.8ms/step once wired into the graph.

### Debug journey (rel 13.17 → PASS) — 3 bugs/insights

1. **Different quant formula → 0.04% byte flips.** I used `x / (amax/448)` (fpu
   divide, eps 1e-12 in a different spot); sglang uses `clamp(x * (1/sx))` with
   `sx = max(amax, 1e-10)/448` (fp8_kernel.py:170-173). The scale is bit-identical,
   but reciprocal+clamp at the E4M3 boundary flips ~1 in 2560 elements —
   enough to turn near-zero elements into rel 13. Fix: mirror their
   formula exactly.
2. **"The GEMM is wrong" ≠ wrong.** Debug compared torch-dequant vs
   cutlass: maxrel 13.18 also appears IN THE REFERENCE — not my kernel.
   Element-wise dissection: rel explodes at **near-zero cancellation** (ref −0.0028 vs
   0.034), whereas large elements differ by 0.5 = pure **half an ulp of bf16**.
   Lesson: `maxrel` with clamp_min(1e-3) is a BROKEN METRIC for GEMV
   (some outputs are ~0.03, ulp-relative explodes) — use maxabs in units of
   bf16 ulp at max magnitude.
3. **Tolerances computed, not guessed.** Separate the two errors:
   mine-vs-exact-torch = 0.61 ulp, cutlass-vs-exact = 0.50 ulp (measured by
   `dbg_cutlass_tol.py`, consistent across all shapes: 0.49–0.50 ulp). Two
   independent bf16 roundings → bound ~1.6 ulp → assert `atol = 2×ulp(max)`,
   rtol=0. o_proj briefly failed at 1.5 ulp with the old atol (4e-3 × max =
   0.95 < 1.5) — not a kernel bug, the tolerance was too tight.

### Proven debug patterns (for the next iteration)
`split-quant-vs-gemm`: the quant component is compared byte-level first
(scale vector + % bytes identical), then the GEMM is compared against **exact fp32
torch-dequant** AND against cutlass — all three, to separate "wrong
formula" from "reasonable rounding" from "small denominator".

---

## 11. Batch decode (bs=16): defeat → TM-blocking → mixed results

Three chained experiments attacking the question: does fusion also win in
**batched decode** (throughput mode), not just bs=1?

### 11a. Row-parallel kernel (from §10) = DECISIVE LOSS at bs=16

`fused_gemv_quant.py` extended M-aware (grid = M × n_blks, 1 program/row).
Accuracy PASS everywhere (≤2 ulp), but timings:

| Layer | bs=16 sglang | bs=16 fused row-parallel | vs |
|---|---|---|---|
| q_proj | 52.4 µs | 105.6 µs | 0.50x |
| gate_up | 42.2 µs | 242.1 µs | 0.17x |
| down | 53.6 µs | 270.3 µs | 0.20x |
| lm_head | 973.0 µs | **15321.2 µs** | **0.06x** |

**Root cause:** each program reads its own weight (N×K) → the weight is read M times.
lm_head 15321 ≈ 16 × 973µs — exact proof of 16x DRAM traffic without reuse.
Cutlass at M=16: weight read once, used by 16 rows (tensor-core tile).
**Lesson: GEMV-fusion bs=1 ≠ GEMM-fusion bs>1 — the program architecture must
change, not just add a grid dimension.**

### 11b. Fused SwiGLU+quant+down-GEMV (3 launches → 1) — wins at bs=1

File: `fused_swiglu_quant_gemv.py`. 1 kernel: load gate_up (M, 2I) →
`silu(g)*u` fp32 in registers → quant mirror-sglang → accumulate against down
weights blockwise. Reference chain = chunk+silu+materialize bf16 → quant → cutlass.

| M | chain | fused | Speedup |
|---|---|---|---|
| 1 | 132.5 µs | **68.1 µs** | **1.95x** |
| 16 | 122.7 µs | 343.0 µs | 0.36x (loses — same cause as 11a) |

Decode bs=1 projection: save ~64µs × 36 layers ≈ **2.3ms/step (6.4%)** if
wired in — the down-proj chain is the longest chain per MLP layer.

### 11c. TM-blocking (16 rows/program) — fixes lm_head, mixed in the middle

File: `fused_gemv_quant_tm.py`. Grid = ceil(M/16) × n_blks; per program:
xq (16,128) quantized per-row + `tl.dot` vs w (128,128) — weight read once
for 16 rows. The wrapper picks the row-kernel for M=1, the TM-kernel for M>1.

| Layer | M=1 | M=16 | M=32 |
|---|---|---|---|
| q_proj | 3.57x | **1.70x** | 1.01x |
| gate_up | 2.37x | 0.75x | 0.45x |
| down | 0.92x | 0.49x | 0.30x |
| lm_head | 0.99x | **1.01x** (from 0.06x!) | 0.50x |

- **Real fix:** lm_head M=16 15321→960µs (15.8x improvement) — weight traffic
  back to 1x. M=32 = 0.50x because tm=16 → weight read 2x (grid_m=2) —
  design ceiling, needs TM=32 (register pressure rises).
- **Remaining losses** (gate_up/down M=16): cutlass tile-scheduler for small
  M=16 GEMMs is already very efficient; our dot+quant overhead > launch
  savings. Conclusion: **fused quant+GEMV = a bs=1 decision (single decode);
  for M≥16 leave it to cutlass.**

### TM precision lessons (tf32 tail)
- `tl.dot` fp32 in Triton 3.8 defaults to **tf32**: our inputs are exact (mantissa
  ≤4 bits for fp8-vals) but accumulation is tailed — got-vs-exact max **1.9 ulp**
  (row-kernel: 0.61). Visible only in outliers; mean diff = pure
  bf16 rounding (0.149 ≈ ulp/√12).
- `input_precision="ieee"` tested: needs 114KB smem > sm_120 limit (101KB)
  → **does not fit**. Decision: measured 3-ulp tolerance (tail 2.5), the tf32
  tail is irrelevant to LLM quality (bf16 itself is 3 decimal digits).
- Tolerance computed from a **measured union bound**: mine-vs-exact (1.9) ∪
  bf16 double-rounding (1.0) ∪ cutlass-vs-exact (0.5).

### Latest integration strategy (§8 revision)
1. **bs=1 decode (latency):** wire in the row-kernel quant+GEMV (§10) +
   swiglu-q-gemm (11b) → potential -5.8ms (quant) -2.3ms (chain) ≈ **-8ms
   out of 36ms = +33% tok/s** (microbench projection figures, e2e not yet tested).
2. **Batched decode (throughput):** leave the sglang path (cutlass) —
   throughput wins come from continuous batching, not kernels.

---

## 12. Server upgrade 4B → Qwen3-8B-FP8 — SAFE, tok/s nearly identical

User request: try a 7/8B model, careful with RAM (free it first). (There is no
Qwen3-7B in the lineup — the choice: 8B. Qwen3-14B-FP8 ≈15GB weights + KV → tight
on 16GB, not tried.)

### Procedure
1. RAM freed: kill the 4B server (±4.5Gi), drop_caches, clear swap →
   14Gi available before download.
2. Download `Qwen/Qwen3-8B-FP8` (2 shards, 8.9GB) → `/root/models/Qwen3-8B-FP8`.
   934G disk free — safe.
3. Start (`start_sglang_8b.sh`, flags identical to 4B: triton, no graph) —
   **load OK on first run**.

### Numbers (identical harness §7: wall-clock client, median of 3)

| Metric | Qwen3-4B-FP8 | Qwen3-8B-FP8 |
|---|---|---|
| weights mem | 4.5 GB | 8.81 GB |
| total VRAM used | ~7.8 GB | **12.84 / 16.31 GB** |
| KV cache | ~30k tokens | 18815 tokens (K+V 2.58GB) |
| WSL RAM | ~4.5 Gi | 4.5 Gi |
| decode tok/s (bs=1) | 27.7 | **26.9** |
| aggregate 4-request | 116.7 | **120.9** |
| t(max=1) prefill+ovh | — | 42 ms |

### Finding: 8B is "free" on this hardware
- Decode bs=1: **-3% tok/s** for 2x the parameters. Step time rose from
  36.0 → 37.2 ms/token — even though weight bytes DOUBLED. This means step time is
  dominated by components that do NOT scale with weight size (launch/quant/attention-chain
  overhead ~25ms, see §8) — the 8B GEMV (~22ms honest
  at 392GB/s) still fits within that budget.
- 4 parallel: aggregate 120.9 tok/s ≈ 4.5x single — the linear batch pattern is
  exactly the same as the 4B (§7). 3.5GB VRAM headroom left, KV 18815 tokens
  is enough for short-to-medium contexts.
- Conclusion: **going 4B→8B = 2x quality, ~3% tok/s cost** on the RTX 5060 Ti.
  14B would be at the edge: weights ±15GB leaving <1GB for KV → practically
  unusable for any context.

### Process notes
- Swap-off briefly hung >2 minutes while buff/cache was fully used by the
  dying process → left it; returned to normal by itself after reclaim.
- Unauthenticated HF download: ~2.5 it/s for small files, large shard ±2.7
  minutes (total ~4 minutes for 8.9GB).

## 13. E2E integration: monkeypatch sglang — 26.8 → 37.2 tok/s (+39%)

Recommendation #1 this morning: wire the fused quant+GEMV (§10) into the real server.
Done, measured, no batch regression.

### Callsite & patch strategy

- `fp8.py:537` resolves the linear function via `dispatch_w8a8_block_fp8_linear()`
  at each layer's init → what gets stored is a function object. But the dispatch
  (in `fp8_utils.py`) looks up module-global **names at call time** →
  rebinding the name at the tail of the file suffices, as long as it is before model load.
- The auto backend on this box = **flashinfer** (0.7.1 installed + Blackwell),
  not the cutlass used by the microbench. Solution: wrap **both** entry points
  (`flashinfer_gemm_...` + `cutlass_w8a8_block_...`) with the same fast path:
  `input_scale is None ∧ M==1 ∧ bf16 ∧ multiple-of-128` → `fused_gemv_quant`
  (row kernel §10); anything else falls through to the original (M>1 prefill/batch →
  old path intact, per the §11 decision).
- Implementation: append a ~40-line tail to `fp8_utils.py` (backup
  `fp8_utils.py.orig13`). `fused_gemv_quant_tm` imported **lazily** on the
  first call (avoids circular imports; Triton compilation waits for the first decode
  step). `try/except` → whatever fails = fallback to original, the server cannot
  crash because of it.

### E2E numbers (Qwen3-8B-FP8, same §7 harness: wall-clock client, median of 3)

| Metric | baseline | patched | Δ |
|---|---|---|---|
| decode ms/token | 37.38 | **26.91** (run1: 26.47) | **-28%** |
| decode tok/s | 26.8 | **37.2** (run1: 37.8) | **+39%** |
| t(max=1) prefill+ovh | 73 ms | 53 ms | -27% |
| aggregate 4-request | 120.9 | 120.3 | ~0 (M>1 → original ✓) |

- Measured savings **10.5 ms/step** — above the ~8ms projection (from the 4B):
  the 8B weight = 2x, so quant+GEMV (launch-bound on the 4B) doubles in value too.
  The remaining 26.9ms/step is dominated by the attention-chain + RMSNorm +
  sampling that are not yet fused (§8).
- Quality: `temperature=0.7, top_p=0.95` generations are coherent (Indonesian,
  factual); microbench accuracy §10 ≤2 ulp bf16.

### Incidents (not caused by the patch)

1. First request with `top_p` → flashinfer JIT compiles the **sampling**
   module → link fails with `cannot find -lcuda` (stubs dir gone
   after `fix_nvcc_wrap.sh` symlinked `lib64` → pip nvidia/cu13) →
   scheduler SIGQUIT. Fix: `ln -s /usr/lib/wsl/lib/libcuda.so
   /usr/local/cuda/lib64/stubs/libcuda.so`. Note: `bench_e2e` (temp
   only, no top_p) never goes through this JIT path — it was never hit before
   because there was no top_p request since the CUDA layout changed.
2. `wsl bash -c "... nohup ..."` can kill the child when the wsl session exits
   (log 0 bytes, pid dead) → start with `setsid ... < /dev/null`.

### Patch status

- **Permanently active** in the venv_netra site-packages. To revert:
  `cp fp8_utils.py.orig13 fp8_utils.py` then restart the server.
- Local copy of the tail: `patch_fp8_fused_tail.py`.
- Not yet wired in: the SwiGLU+quant+GEMV chain (§11b, potential -2.3ms more)
  — needs a hook at the MLP intermediate, not the linear wrapper; in the backlog.

## 14. 8B prefill — 2314 tok/s input (baseline, patch does not touch M>1)

Method (`bench_prefill.py`): unique uuid-hex prompts (guaranteed not to hit the
prefix radix cache), `t(max_new=1)` single-run, slope
`(t_long − t_short)/(tok_long − tok_short)`; median of 3 pairs.

| | short | long |
|---|---|---|
| uncached tokens | 388 | 14028 |
| wall | 103 ms | 5998 ms |
| **slope** | — | **0.432 ms/token → 2314 tok/s** |

- Chunked prefill 2048 → 14028 tok ≈ 7 chunks ≈ 6s, consistent.
- Practical context: an 8k-token document ≈ **3.5s TTFT**; server context limit
  18809 tokens (KV 18815 − reserved).
- Far slower than the decode-side gain: prefill GEMMs with M≥2048 are already on
  the efficient flashinfer/cutlass path — the §13 patch deliberately only
  targets M==1 (decode). Prefill is not a GEMV fusion target.
- Method pitfalls (noted so they are not repeated): (a) using the same prompt
  → radix cache hit, bogus numbers; (b) uuid-hex 36k chars = 19k tokens >
  18809 limit → 400 error; (c) `max_new=1` overhead is still ~40-100ms →
  must use the long-short delta, not a single absolute.

## 15. Fusing the SwiGLU+quant+down chain (§11b) into the server — ZERO GAIN, reverted

Second integration attempt: patch `Qwen2MLP.forward` (qwen2.py — Qwen3
imports the same class) → M==1: `gate_up (via the §13 patch) →
fused_swiglu_q_gemm (silu+quant+down GEMV in 1 kernel)`.

### Results

| | §13 only | §13+§15 chain |
|---|---|---|
| decode ms/token | 26.91 | 27.14 (noise ±0.4) |
| tok/s | 37.2 | 36.8 |
| aggregate 4-req | 120.3 | 122.2 |
| chain fires? | — | **YES** (marker:1332 S,0 E) |

**Gain below noise → reverted** (`qwen2.py.orig15`).

### Why the -2.3ms projection missed

§11b measured the chain against the **cutlass baseline** (separate quant+GEMM).
After §13, both GEMMs (gate_up & down) are already fused quant+GEMV —
all that remains between them is just `silu_and_mul` (one small kernel
~15µs) + act materialization. That remainder is what the chain saves, and its
~0.5ms/step does not emerge from the noise. Lesson: **the projection baseline
must be the path AFTER the previous patch, not the vanilla path.**

### A/B quality test (3 greedy prompts + mean logprob)

Finding: the odd behavior ("Jelasan sejarah..."-style lead-ins, echo of the prompt)
appears in **all three** configurations: unpatched (U), §13 (B), §13+§15 (A) →
**not caused by the patch**; this is raw Qwen3 `/generate` behavior without a chat
template (completion mode, greedy). Factual answers correct in all:
391 ✓, blue/Rayleigh ✓, Paris appears at the end ✓. Mean logprob [1][2]
identical across configs (−0.26, −0.11).

### Process artifacts (reused next tick)

- Marker file `/root/swiglu15.log` (S/G/E) — a cheap "instrumentation via file
  append" pattern for paths inside a scheduler spawn.
- `pkill -f 'sglang[.]launch_server'` — **must** use the character-class
  trick; without it pkill matches itself (exit9, server not killed).
- `return_logprob`/`logprob_start_len` = TOP-LEVEL request fields of
  `/generate`, not inside `sampling_params` (400 if wrong).
## 16. CUDA graph + patch §13 — WIN, new production config

Hypothesis: capture graph (decode bs=[1,2,4,8]) on top of the §13-patched
server → eliminate python/kernel-launch overhead per decode step.

### Setup

- New script `start_sglang_8b_graph.sh` = `start_sglang_8b.sh` minus
  `--disable-cuda-graph` (old flag deprecated; sglang's current default
  already has graph on, prefill backend "breakable").
- Capture log: decode bs=[1,2,4,8] elapsed 2.20s / 0.06GB; prefill
  breakable 4..2048 token elapsed 14.2s / 0.48GB; remaining avail mem
  2.57GB. Triton kernels captured normally (python wrapper runs during
  capture, replay uses the recorded kernels).
- §13 patch remains active; §15 was already reverted before start.

### Results (Qwen3-8B-FP8)

| | no graph (§13) | + CUDA graph | Δ |
|---|---|---|---|
| decode ms/token | 26.91 | **23.78** | **-11.6%** |
| tok/s bs=1 | 37.2 | **42.1** | **+13%** |
| t(max=1) | ~73 ms | 76 ms | ≈ (prefill unchanged) |
| aggregate parallel-4 | 120.3 | **176.1** | **+46%** |
| per-req parallel | 5.8s @ wall | 5.8s [5.8×4] | balanced |

Parallel-4 reaches 176 tok/s — +46% over no-graph. Reason: at M=1 graph
replay eliminates Python/triton launch overhead per step (~3ms × 4
layer-call-sites), and at bs>1 the entire attention+MLP pipeline is
replayed without any launches at all.

### Verdict

- **New production config**: `start_sglang_8b_graph.sh` (graph ON +
  §13 patch). Coherent smoke (lead-in output matches raw Qwen3 greedy
  behavior — already normalized in §15, not a regression).
- Quality: logprob/quality not re-run in graph (same kernels, only a
  different launch path → zero numeric risk; sampling remains outside the
  graph).
- Total since the 4B baseline: 27.7 → 42.1 tok/s decode bs=1, and
  116.7 → 176.1 parallel-4 (all on 8B-FP8 + patch + graph).

### Incidents

None. Graph capture succeeded on the first try; the libcuda stub fix
from §13 remained sufficient.

### Addendum after graph bench

- t(max=1) warm = **33 ms** (cold 76 ms; no-graph 73 ms) —
  the breakable prefill-graph also benefits short-prompt TTFT.
- Parallel-4 confirmed at **172.7–176.1 tok/s** (from the server's own
  log: "gen throughput 176.00, cuda graph: True").

## 17. Attention-side fusion (QK-norm+RoPE): REJECTED — decode is already ROOFLINE

Hypothesis: integrate `fused_qk_rope` (§9, 10.85x vs HF eager) into the
server the §13 way. After investigation: **no headroom — integration
rejected without burning hours** (lesson from §15: project against the
RIGHT baseline first).

### Finding 1: QK-norm is already fused by sglang itself

`apply_qk_norm` (models/utils.py) uses JIT `fused_inplace_qknorm` when
CUDA ∧ head_dim∈{64,128,256,...} ∧ eps q==k. Qwen3-8B head_dim=128 →
**eligible & active** (zero warnings in the log). What remains: rope = 1
sgl_kernel launch, and under CUDA graph launch overhead is already zero.

### Finding 2: remaining traffic saved by fusion = KB vs GB

qknorm+rope fusion only removes the mid-layer materialization per layer:
(q 8KB + k 2KB) × 36 layers ≈ **<0.4MB/step**. Weight stream per step:
**~8.5GB**. Ratio 1:20,000. Contrast with §13 (there the Gb-scale was
real: quant kernels removed + second weight pass eliminated).

### Finding 3 (main result): measured DRAM ceiling

| Metric | Value |
|---|---|
| D2D copy 1.5GB (r+w) | **358 GB/s** (5060 Ti theory ≈ 448) |
| Decode step graph+§13 | 23.78–23.88 ms |
| Weight 8B-FP8 / step | ~8.5 GB |
| Effective if 100% = weight reads | **≈357 GB/s** |
| nvidia-smi during decode | SM 100%, mem-util 93%, 2827MHz, **113W** (not power-capped) |

SM 100% waiting on memory (power at 63% TDP). **Kernel optimization no
longer wins — the ceiling = weight_bytes / BW.** How to win: **reduce
weight reads** → speculative decoding (ngram/EAGLE). Continued in §18.

## 18. NGRAM speculative decoding — WIN, final production config

Following up on §17: reducing weight reads = speculative. Chose
**NGRAM** (built into sglang, `--speculative-algorithm NGRAM`) because
it **needs no draft model** — the draft = an n-gram tree from
already-generated text (CPU-side matching), verified as a batch of M=5
on GPU (cutlass path, safe with the §13 patch by-design).

### Setup

`start_sglang_8b_ngram.sh` = graph + §13 +:
`--speculative-algorithm NGRAM --speculative-num-steps 4
--speculative-num-draft-tokens 5`
Server note: max_running_requests auto-resets to 48; mixed
chunked-prefill is disabled by ngram.

### Results (Qwen3-8B-FP8)

| | graph+§13 (§16) | + NGRAM (§18) | Δ |
|---|---|---|---|
| decode ms/token | 23.78 | **19.92** | **-16%** |
| tok/s bs=1 | 42.1 | **50.2** | **+19%** |
| aggregate parallel-4 | 176.1 | **244.6** (warm 280.7) | **+39%** |
| server-side peak | 176 tok/s | **349 tok/s** (accept len 2.48) | — |
| t(max=1) | 33 ms (warm) | 78 ms | TTFT up (mixed prefill off) |

Accept len measured in the log: **1.5–2.5 tokens per verify-5** (accept
rate 0.13–0.37, rising as the n-gram corpus fills). Repetitive Indonesian
phrasing ("kota tersebut", list structures) yields decent matches.

### Total journey tonight

| | baseline 4B | final (8B+patch+graph+ngram) |
|---|---|---|
| decode bs=1 | 27.7 tok/s | **50.2 tok/s (+81%)** |
| aggregate parallel-4 | 116.7 | **244.6 (+109%, 2.1x)** |
| TTFT short prompt | ~73 ms | 78 ms (ngram) / 33 ms (graph only) |

### Verdict

- **Final production**: `start_sglang_8b_ngram.sh` (ngram ON). If the
  user needs minimal TTFT (short chats): fall back to
  `start_sglang_8b_graph.sh` (TTFT 33 ms).
- Quality: speculative preserves the distribution (verified by the
  target model); coherent smoke — the lead-in echo still matches raw
  Qwen3 greedy style (already normalized in §15, not a spec-dec
  effect).
- Slight loss on TTFT: ngram disables mixed chunked prefill —
  conscious trade-off: decode throughput ≫ TTFT for long use-cases.

### Incidents

None. Ngram started clean on the first try; no draft model download
needed; VRAM sufficient (verify M=5 still fits graph bs=8).

### Addendum

- **Verify graph captured**: "Capture target verify CUDA graph
  bs=[1..8], num_tokens_per_req=5" → verify runs on a full graph,
  not eager. This means **widening the draft window is nearly free** in
  the bandwidth-bound regime (weight reads are the same, M=5 vs M=8) —
  tried in §19.
- The `--enable-fused-qk-norm-rope` flag (found in server_args while
  drafting §17) turns out to be **used only by qwen3_moe.py & mellum.py**
  — Qwen3 dense is unaffected. Skipped without a restart (saved 2
  minutes).

## 19. Widened ngram window (steps 7, draft 8) — ANOTHER WIN

§18-addendum hypothesis: in the bandwidth-bound regime, verify M=8 is
almost as expensive as M=5 (same weight reads) → a wider window ≈
free for accepting more per verify.

### Setup

`start_sglang_8b_ngram_w8.sh`: identical to §18, only
`--speculative-num-steps 7 --speculative-num-draft-tokens 8`.
Verify graph capture automatically uses num_tokens_per_req=8.

### Results

| | §18 (s4/d5) | §19 (s7/d8) | Δ |
|---|---|---|---|
| decode ms/token (cold corpus) | 19.92 | **15.71** | -21% |
| tok/s single (cold) | 50.2 | **63.7** | **+27%** |
| tok/s single (warm corpus) | — | **81.7** (12.25 ms) | — |
| t(max=1) warm | 78 ms | 60 ms | ≈ |
| parallel-4 warm | 280.7 | 281.4 | flat |
| accept len (log) | 1.5–2.5 | 1.24 → **2.72** keeps rising | — |

Honest note: §19's parallel median (223.4) was dragged down by the first
cold run (165) — the second warm run matches §18 (≈281). The real
advantage is in **single-decode** (63.7–81.7 vs 50.2). Accept len grows
as the n-gram corpus fills → production gets faster with use.

### Verdict

**Final production = `start_sglang_8b_ngram_w8.sh` (steps7/draft8).**
Total from the 4B baseline (27.7 tok/s): **81.7 tok/s warm = +195%
(3x)**, parallel 116.7 → 281.4 (**2.4x**).

## 20. External ngram corpus (Wikipedia ID) — NEGATIVE, reverted

Hypothesis: an external corpus of 5.58M tokens (1500 Wikipedia ID
articles, per-document JSONL) → accept-rate rises across requests.

### Setup

- `prep_ngram_corpus.py`: `datasets.load_dataset("wikimedia/wikipedia",
  "20231101.id", streaming)` → 1500 articles → `/root/corpus_id.jsonl`
  (18MB). Note: the old `wikipedia` (script-based) is rejected by
  datasets 5.x — use the `wikimedia/wikipedia` repo.
- Flags: `--speculative-ngram-external-corpus-path` +
  **`--speculative-ngram-external-sam-budget 7`** — the budget MUST be
  ∈ (0, num_draft_tokens−1]; with draft8 → max 7. (Validation at
  speculative_hook.py:1244.)

### Results vs §19 (w8 without corpus)

| | §19 w8 | §20 +corpus |
|---|---|---|
| decode tok/s | 63.7 cold / **81.7** warm | **54.6** |
| parallel warm | 281.4 | **199.2** |
| accept len | 1.24→2.72 (rising) | **1.44–1.62** (flat and low) |

### Why it lost

The external SAM budget is capped at 7 nodes (validation ≤ draft−1) →
the wiki subtree barely plays a role, but its presence **shifts the SAM
distribution** so that self-ngram (which actually matches the model's
generation style) wins less. Net: harmful.

### Verdict

The external corpus is **removed from production**; config reverts to
`start_sglang_8b_ngram_w8.sh`. Do not retry unless the budget API
changes in a future sglang version. Remaining artifacts (harmless):
`/root/corpus_id.jsonl`, `prep_ngram_corpus.py`, the
`start_sglang_8b_ngram_corpus.sh` script (left as a note).




## 21. Ngram window draft>8 (steps15/draft16) — NEGATIVE, candidate 9b

Hypothesis: verify is bandwidth-bound (§17) → draft 16 nearly free,
accept len rises → tok/s rises.

### Setup

- `start_sglang_8b_ngram_w16.sh`: NGRAM steps15/draft16 (rest of
  config identical to w8). RAM cleaned first (drop_caches; 653Mi→14Gi
  free).
- Verify graph capture: , bs=[1..8], only +0.11GB, avail 2.62GB —
  capture SAFE (the "need to raise max_bs" hypothesis turned out
  unnecessary; the graph uses draft tokens as rows, not bs).

### Results vs §19 w8 (production)

| | §19 w8 | §21 w16 |
|---|---|---|
| decode single cold | 63.7 | 62.7 |
| decode single warm | **81.7** | 71.3 |
| parallel-4 median | **281.4** | 220.6 |
| accept len (parallel) | 1.24–2.72 | 1.86–3.14 |
| verify-graph mem | — | +0.11GB (not an issue) |

### Analysis

Accept len DOES rise (longer draft → more matching candidates), but the
increase does not cover the cost:

1. Verify 16 tokens vs 8 tokens at bandwidth-bound = double the weight
   width pass... no — verify is a single forward with M=16 rows, weight
   is still read once per layer (M is the token dimension). What
   increases: attention KV-read per token & logits decode of 16
   columns. Still ~bandwidth-bound for weights, but the ngram BFS
   (max_bfs_breadth=10, trie_depth=18) puts more and more "weak
   candidates" at the tips — accept rate/token (0.06–0.14) stays low.
2. Decision: **revert to w8.** Draft sweet-spot = 8 for this ngram-BFS
   config. Diminishing returns confirmed (candidate 9b complete,
   NEGATIVE).

### Final state

Server back to `start_sglang_8b_ngram_w8.sh`. The w16 script is left as
a note.
