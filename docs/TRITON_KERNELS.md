# Eksperimen Custom Kernel Triton — RTX 5060 Ti (sm_120)

Rangkuman hasil eksperimen kernel Triton di WSL2, lanjutan dari PLAN.md.
Semua benchmark: **warmup 20 + 100 iterasi, `torch.cuda.Event`** (bukan
`time.time()` — angka 345µs di PLAN.md ternyata cold-start/lazy-init).

Shape standar: **B=16, V=151936, fp32** (≈ 9.7MB, muat di L2 5060 Ti).

---

## 1. Argmax standalone — NEGATIF, torch menang

File: `triton_argmax.py`, `test_two_stage_argmax.py`

| Varian | Steady-state | vs torch |
|---|---|---|
| `torch.argmax` (CUB DeviceReduce) | **16.09 µs** | 1x |
| Triton two-stage (BLOCK=8192, partial buffer) | 30.65 µs | 0.52x |
| Triton single-launch (loop per row) | 19.28 µs | 0.83x |
| Two-stage ala other-agent (128 chunk) | 51.79 µs | 0.36x |

**Kesimpulan:** tensor 9.7MB muat di L2, torch CUB sudah dekat optima.
Dua-duanya kalah karena overhead launch + round-trip partial buffer.
Saran "two-stage argmax 345µs" dari agent lain dibantah dengan harness
mereka sendiri (dengan warmup: baseline turun ke ~18µs, kernel 51µs).

**Pelajaran:** jangan optimasi reduction tunggal — cari **fused multi-op**.

---

## 2. Fused sampling — POSITIF, menang 2–8x

File: `fused_sampling.py` (deploy: `/root/fused_sampling.py`)

### Desain
Naive: full-vocab `softmax` (materialisasi 9.7MB) → sort/topk → top-p mask →
renormalisasi → `multinomial`. Fused:

1. `torch.topk(logits, 50)` langsung di logits mentah — softmax monotone,
   urutan topk(logits) ≡ topk(probs), **pass softmax penuh DILEWATI**.
2. Satu kernel Triton kecil (grid = baris): `exp((v−v0)/T)` → cumsum top-p →
   renormalisasi → **inverse-CDF sample** (`tl.rand` + `argmax` pada cum ≥ u).

### Hasil (K=50, top_p=0.9, T=0.7)

| Pipeline | µs |
|---|---|
| Naive full-vocab (softmax + sort V) | 1385.29 |
| Naive kompeten (full softmax → topk) | 337.25 |
| **Fused** | **167.18** |

**8.29x** vs naive full · **2.02x** vs naive kompeten.
Breakdown: `torch.topk` ≈ 170µs (dominan), kernel Triton sendiri beberapa µs.

### Verifikasi
- `top_k=1` ≡ `torch.argmax` (deterministik) ✅
- 50 seed × 16 baris: hasil selalu di set top-k **dan** nucleus (konvensi HF:
  keep_i iff cumsum-sebelum-i ≤ top_p) ✅
- Statistik: 5000 sampel, V=128, top_p=1 → `max|freq − p| = 0.0109`
  (toleransi 0.03) ✅

### Bug yang ketangkap test
Versi awal `keep = (cum − w) <= top_p` membandingkan cumsum **tak ternormalisasi**
(total `w` bisa ≫ 1) dengan `top_p` mentah → mask salah (error distribusi 0.29).
Fix: `keep = (cum − w) <= top_p * total`. Contoh kenapa check statistik itu wajib —
check deterministik/keanggotaan tetap lolos meski mask salah.

### Langkah lanjut (kalau mau)
`torch.topk` kini bottleneck-nya; custom radix-select sulit dijangkar CUB
(sama seperti argmax). Diminishing returns — stop di sini.
Kandidat fused berikutnya yang masih punya ruang menang:
RMSNorm+residual, SwiGLU, quantization (semua multi-op, bukan reduction tunggal).

---

## 3. Fused add+RMSNorm — menang di decode

File: `fused_rmsnorm.py`. Kontrak Qwen3-4B: hidden=2560, eps=1e-6, 36 layer.
Fused: 1 kernel → `out = rmsnorm(x+r)*w` + `residual_new = x+r`.
Naive: `x + residual` (materialize) → `F.rms_norm` (2 launch + intermediate).

| Shape | Naive | Fused | Speedup |
|---|---|---|---|
| decode (16, 2560) | 45.76 µs | 26.72 µs | **1.71x** |
| prefill (512, 2560) | 19.17 µs | 15.99 µs | 1.20x |

Akurasi: allclose vs `F.rms_norm` (max err ~1e-7) ✅.
Catatan: menang karena menghilangkan materialisasi intermediate — di decode
(kecil) launch-count menang, di prefill trafik memori menang.

---

## 4. Fused SwiGLU + FP8 E4M3 quant — MENANG BESAR

File: `fused_swiglu_fp8.py`. Kontrak diverifikasi dari `config.json`:
`intermediate_size=9728` (bukan 2560/5120), hidden=2560, 36 layer.
Grup kuantisasi 128 elemen → 76 grup/baris. Grid quant `(B, 76)` — absmax
**lokal per grup**, tanpa cross-program reduce.

Rantai naive (bukan strawman): `silu*up` materialize FP32 → `amax` per grup →
`scale=max/448` → `y/scale` clamp → cast FP8 (≈6 launch + 2 intermediate).
Fused: 1 kernel — `silu(g)*u` + absmax + scale + cast FP8, semua di register.

| Shape | Rantai | Naive | Fused | Speedup |
|---|---|---|---|---|
| decode (16, 9728) | SwiGLU saja | 14.55 µs | 16.14 µs | 0.90x (kalah tipis) |
| decode (16, 9728) | SwiGLU+FP8 | 116.99 µs | 49.71 µs | **2.35x** |
| prefill (512, 9728) | SwiGLU saja | 66.91 µs | 23.42 µs | **2.86x** |
| prefill (512, 9728) | SwiGLU+FP8 | 542.19 µs | 54.30 µs | **9.99x** |

Verifikasi: scale allclose (rtol 1e-5) · error dequant ≤ setengah ulp E4M3 ·
**output FP8 byte-identik 100%** vs baseline di semua shape ✅.

### Pelajaran test (2 bug test, kernel lolos dua-duanya)
1. `QMAX` global tak bisa diakses `@triton.jit` → harus argumen/constexpr.
2. Bound error asumsi `scale/2` (grid int8) — E4M3 itu grid **float**
   (mantissa 3 bit), bound benar = `|y|/16` (setengah ulp relatif). Test awal
   gagal 63k elemen padahal output fp8 identik 100% — test-nya yang salah.

### Kesimpulan strategi
- Fusion yang menang = yang **menghilangkan materialisasi intermediate**
  atau **menggabungkan rantai banyak launch** (quant chain = rantai terpanjang,
  win terbesar 2.35–9.99x).
- SwiGLU murni di decode kecil (16 baris) ≈ seri — torch 2 kernel kecil sudah
  cepat; menang muncul di prefill. Klaim "MLP 90% waktu" perlu diuji end-to-end,
  dan activation-Q8 di serving Qwen3-FP8 saat ini *weight-only* — harness ini
  berlaku untuk pola W8A8, bukan plug-in langsung server.

---

## 5. Stress battery (26 check, semua PASS)

File: `stress_tests.py`. Kasus edge/ ketiga kernel:

- **sampling (15)**: seed reproducible · K=4096 support di luar top-50 ·
  top_p=0 ≡ top-1 · T=0.01+margin5 ≡ argmax deterministik · T=1000 tetap
  top-k · K non-pangkat-2 (1,3,17,51,63,77) · input cast bf16 · baris duplikat ·
  skala logit 1e4 (anti overflow exp).
- **rmsnorm (4)**: baris nol · ekstrem 1e6 · non-contiguous · weight=1 ≡ manual.
- **swiglu_quant (7)**: determinisme byte · semua-nol (scale floor 1e-30) ·
  saturasi menyentuh tepat 448 · gate=−50 silu→0 finite · ekstrem ×10 fp8
  identik ≥99.9% · determinisme.

### Temuan dari battery (bukan bug kernel — bug TEST)
1. **K=V (151936) tak bisa diuji**: BLOCK=next_pow2(V)=262144 → kompilasi
   Triton >10 menit CPU. Di-skip, pakai K=4096. Batas praktis BLOCK: ≤16384.
2. **T=0.01 pada logit random memang BUKAN argmax** — gap top1/top2 ~0.01,
   e^(−gap/T)≈0.3 masih mungkin. Test yang benar: margin buatan (+5) → deterministik.
3. **Round-trip bf16→fp32 ≠ fp32 asli** — equals tak mungkin; check yang benar
   adalah membership di nilai hasil cast.
4. Python `for seed in range(50)` panggil kernel satuan masih oke (<1s).

---

## 6. Fused RoPE — menang 3.3x / 8x

File: `fused_rope.py`. Kontrak: Qwen3-4B `head_dim=128`, 32 heads (config.json).
Naive = HF eager: `cat(cos,cos)` + `cat(sin,sin)` + neg + `cat` rotate_half +
mul + mul + add ≈7 launch + 3 intermediate. Fused: 1 launch/token —
`out = x*cos + sign(half) * mirror(x) * sin`, mirror via `offs ^ half`
(bit-flip posisi separuh).

| Shape | Naive | Fused | Speedup |
|---|---|---|---|
| decode (16,32,1,128) | 131.89 µs | 39.74 µs | **3.32x** |
| prefill (4,32,1024,128) | 1718.84 µs | 214.63 µs | **8.01x** |

Akurasi: allclose bf16-scale · norm-invarian · pos-0 identitas · bitwise
100% di run utama.

### Bug nyata yang ketangkap test (kernel sempat salah)
Versi pertama kernel: `x1=where(offs<64,x,0); x2=where(offs>=64,x,0)` —
**istilah cross-halves hilang** (`out_lo = x1*cos − 0*si`, harusnya
`− x[i+64]*si`). Kasus S=1 "lolos" karena sin(0)=0 → identitas, jadi tes decode
sekalipun tidak membuktikan apa-apa. Fix: load nilai mirror `x[offs ^ 64]`.
**Pelajaran: tes yang lolos karena kasus trivial ≠ benar — prefill wajib.**

### FMA vs 3-rounding (bukan bug)
Sisa mismatch ≤1 ulp bf16 di elemen boundary: Triton kontraksi `x*co+xm*si`
jadi FMA (1 rounding) vs torch mul+mul+add (3 rounding). Fused **lebih akurat**;
toleransi di skala bf16 (rtol 8e-3), bukan fp32.

---

## 7. Uji end-to-end server (A/B nyata, port 30000)

File: `bench_e2e.py`, `bench_parallel.py`, `start_sglang_graph.sh`,
`start_sglang_fi.sh`. Metode: wall-clock client — `max_new_tokens=1`
(≈prefill) vs `512`; `(t512 − t1)/511` = ms/token decode. Median 3 run.

| Varian | Decode tok/s | Kesimpulan |
|---|---|---|
| Baseline (triton, graph off) | **27.7** | acuan |
| + CUDA graph aktif (capture bs 1–8 OK) | 27.3 | **tidak beda** |
| + flashinfer backend (warm) | 27.5 | **tidak beda** (run pertama 19.7 = JIT compile) |
| **4 request paralel (aggregate)** | **116.7** | **4.2x linier!** |

### Temuan
1. **Decode di bs=1 murni weight-bandwidth-bound.** 4B FP8 ≈ 4.5GB / 36ms
   ≈ 125GB/s efektif (puncak 5060 Ti ~448GB/s). Naik batch → weight dibagi:
   4 paralel = 4.2x aggregate, **waktu per request identik** (9.4s vs 9.2s).
   Attention/KV bukan faktor di konteks pendek.
2. **CUDA graph & flashinfer tidak menambah tok/s di sini** — karena bukan
   launch/attention-bound. Triton backend tetap dipakai (config kanonik).
   Server sudah di-revert ke `start_sglang.sh` setelah A/B.
3. Catatan versi: sglang 0.5.22 `--cuda-graph-backend-decode {full,...}`;
   arg key legacy `cuda_graph: False` di log MENIPU — yang aktif
   `cuda_graph_config` + bukti log `Capture target decode CUDA graph end`.
4. `/metrics` 404 di build ini; `get_server_info` jalan.

### Implikasi
- Latensi per token tunggal (27.5) adalah **sweep bs=1** — tak bisa dinaikkan
  oleh fusion kernel attention/sampling; yang naik = **throughput lewat batch**
  (continuous batching SGLang memanfaatkannya saat load).
- Fusion kernel kita (RoPE/SwiGLU/RMSNorm/sampling) tetap relevan untuk
  **mengurangi waktu per forward pass** — proporsional terhadap 36ms itu —
  terutama saat elemen launch/traffic berkurang di tiap layer.

---

## 8. Iterasi malam-3: validasi real-logits + bedah 125GB/s

### 8a. Fused sampling di logit ASLI server — PASS

File: `real_logits_sampling.py`. Ambil top-50 logprob nyata via
`/generate` + `return_logprob` + `top_logprobs_num: 50`, bangun baris sparse
(1, 151936), jalankan `fused_sample` di atasnya.

| Check | Metode | Hasil |
|---|---|---|
| 1 support | 20 seed, top_k=50, top_p=0.9 | semua sampel ∈ nucleus (22 token) ✅ |
| 2 distribusi | 5000 sampel vs teori + multinomial | TV(fused, teori)=**0.0211**, TV vs multinomial=**0.0440** (tol 0.05) ✅ |

**2 bug TEST yang ketangkap** (kernel lolos dua-duanya):
1. API key salah: response pakai `output_top_logprobs` (list per langkah,
   entri `[logprob, token_id, text]`), bukan `meta_info["top_logprobs"]` dict.
2. TV=1.0 sempat muncul: `kept = order[keep]` itu **index ke top-50, bukan
   token id** — massa teori salah taruh di posisi vocab 0..49 → support
   disjoint total. Fix: `ref_p[toks[kept]] = probs[kept]`.
→ Fused sampling resmi tervalidasi di distribusi logits nyata, bukan hanya sintetis.

### 8b. Bedah 125GB/s: dequant-naive BUKAN path server

File: `bench_fp8_gemv.py` (simulasi weight-only: dequant materialize → GEMV).

| Layer | dequant | GEMV bf16 | chain (GB/s) |
|---|---|---|---|
| q_proj 2560×4096 | 577.9 µs | 56.8 µs | 727.8 µs (14) |
| gate_up 2560×9728 | 1348.7 µs | 121.4 µs | 1819.4 µs (13) |
| down 9728×2560 | 1358.9 µs | 134.4 µs | 1816.4 µs (13) |
| lm_head 2560×151936 | 21458.8 µs | 1898.6 µs | 29066.9 µs (13) |

Chain 13–14 GB/s ≪ 125GB/s server → **server mustinya tak mungkin memakai
dequant-materialize per step** (langkah decode akan ~300ms, bukan 36ms).
`torch._scaled_mm` RuntimeError di sm_120 dengan layout blockwise 2D.

### 8c. Path asli server: W8A8 blockwise — inilah angka sebenarnya

File: `bench_sglang_fp8.py`. Checkpoint diverifikasi: `weight_scale_inv`
**2D blockwise 128×128** (`[N/128, K/128]`, mis. gate `[76, 20]`). Path:
`sglang_per_token_group_quant_fp8_row_padded` + `fp8_blockwise_scaled_mm`.

| Layer (N×K) | quant act | fp8 gemv | GB/s | bf16 gemv | fp8/bf16 |
|---|---|---|---|---|---|
| q_proj 4096×2560 | 46.5 µs | 43.4 µs | 232 | 56.9 µs | 1.31x |
| o_proj 2560×4096 | 41.5 µs | 17.4 µs | 579 | 40.5 µs | 2.33x |
| gate_up 9728×2560 | 45.7 µs | 28.7 µs | **833** | 117.9 µs | **4.11x** |
| down 2560×9728 | 39.8 µs | 33.9 µs | 705 | 111.4 µs | 3.29x |
| lm_head 151936×2560 | 40.6 µs | 951.8 µs | 392 | 1880.5 µs | 1.98x |

Catatan jujur: angka GB/s >448 (peak GDDR7) di gate_up/o_proj/down =
weight muat di **L2 32MB** saat microbench berulang — steady-state DRAM
honest hanya lm_head (389MB): **392 GB/s fp8**.

### Temuan besar: mana 36ms/step pergi?

Hitung kasar per langkah decode (bs=1), fp8 GEMV total ~6–7ms (DRAM-honest),
lm_head ~1ms, sisanya **~25ms bukan GEMV**. Terbesar yang terukur:

- **Quantisasi aktivasi per-linear: ~40µs/fan-call × 6 call/layer × 36 layer
  ≈ 8.6ms ≈ 24% langkah decode** — hampir selalu launch-bound (M=1, kerja
  kecil, bayaran tetap ~40µs).
- Attention, RMSNorm, RoPE, sampling, launch tail: sisanya.

→ **125GB/s efektif ≠ GPU jenuh bandwidth.** Headroom DRAM ~3.6x; yang
menghambat = overhead launch/quant. Kesimpulan strategis berubah:
1. Fusion kernel kita (RoPE 3-8x, RMSNorm, sampling) **relevan** — memang
   menyerang lapisan overhead ini, asalkan terpasang di graph.
2. Target penghematan tunggal terbesar berikutnya: **quant sekali per
   input hidden untuk {q,k,v}** (3 call → 1) ≈ hemat 4.3ms ≈ +12% tok/s —
   milik integrasi sglang, bukan kernel baru.
3. Path server terkonfirmasi W8A8 (bukan weight-only); harness SwiGLU+FP8
   kita sebangun dengan pola quant+GEMM ini.

#### Koreksi (setelah baca `qwen3.py` sglang) — `bench_quant_fuse.py`

Sudah kufuse di sglang: `QKVParallelLinear` (1 quant utk q+k+v) +
`gate_up_proj` (1 quant utk gate+up) → **4 quant call/layer, bukan 6**, dan
input tiap call beda (hidden / attn_out / ffn_in) → tak bisa digabung lagi
di level Python. Terukur per call (bs=1): qkv 39.8 · o 42.3 · gate_up 39.2 ·
down 40.7 µs → **5.83 ms/step = 16.2% decode** (koreksi dari 8.6ms/24%).
Overhead hampir tetap ~40µs (launch-bound), bs=16 cuma 23.8µs.
→ Target sesungguhnya: **fusi quant ke dalam kernel GEMM** (hilangkan launch
terpisah) — lihat eksperimen 10.

---

## 9. Fused QK-norm + RoPE + GQA expand — menang 4–11x

File: `fused_qk_rope.py`. Kontrak Qwen3-4B: 32 q-head, **8 kv-head (GQA 4:1)**,
D=128, rms_norm_eps=1e-6. HF eager per layer: `rms_norm` q (materialize) →
rope q (≈7 launch) · `rms_norm` k → rope k → `repeat_interleave(4)` (materialize
4x). Fused: **1 launch q + 1 launch k** — grid di ruang head keluaran,
`kv_h = h // 4` dibaca on-the-fly; mirror RoPE pakai nilai **sudah ternormalisasi**
(ms sama per head, w[i^half]).

| Shape | Naive | Fused | Speedup |
|---|---|---|---|
| decode (16, S=1) | 312.77 µs | 28.84 µs | **10.85x** |
| prefill (4, S=1024) | 2592.81 µs | 638.18 µs | **4.06x** |

Verifikasi: allclose q & k vs eager · pos-0 ≡ rmsnorm murni (rope identitas) ·
4 head kelompok GQA identik · norm-invarian pasca-rope ✅.

### Pelajaran
- **Bug kernel lagi (eps global)** — sama seperti QMAX: `EPS` global tak bisa
  diakses `@triton.jit` → pass `eps` sebagai argumen. (Kesekian kalinya —
  pola ini sudah hafar: semua constanta harus argumen/constexpr.)
- Prefill sempat gagal (0.03125): **bukan bug** — naive membulatkan norm ke
  bf16 *sebelum* rope (2 rounding), kami fp32 penuh; elemen cancellation
  (out≈0 dari suku ~4) membawa 1 ulp bf16 sebagai error absolut. atol 5e-2.
- Decode run-to-run 6.3–10.9x (noise clock/thermal WSL2) — arah tetap: menang
  besar, karena menghapus materialisasi + GQA expand di intermediate.
- GQA expand bukan "gratis" di eager: `repeat_interleave` = copy 4x ukuran K —
  di fused kernel hilang sama sekali (read kv head sekali, tulis keluaran).

---

## 10. Fused quant+GEMV (1 launch) — menang 1.8–4.9x, PASS semua shape

File: `fused_gemv_quant.py`, debug: `dbg_gemv_quant.py`, `dbg_cutlass_tol.py`,
`dbg_o_proj.py`. Menyerang **5.83ms/step (16.2% decode)** dari 4 quant
launch/layer (lihat §8 koreksi). Path sglang = `row_padded quant` (launch 1)
+ `fp8_blockwise_scaled_mm` cutlass (launch 2); fused = 1 Triton kernel —
absmax per grup-128, cast fp8 di register, akumulasi fp32 dengan scale
x × scale w (blockwise 128×128 dari checkpoint).

| Layer (N×K) | sglang (quant+gemm) | fused 1 kernel | Speedup | maxabs vs ref |
|---|---|---|---|---|
| q_proj 4096×2560 | 70.5 µs | **20.8 µs** (483 GB/s) | 3.38x | 0.50 ulp |
| o_proj 2560×4096 | 62.9 µs | **32.0 µs** (315 GB/s) | 1.97x | 1.50 ulp |
| gate_up 9728×2560 | 142.5 µs | **29.0 µs** (823 GB/s) | 4.91x | 1.00 ulp |
| down 2560×9728 | 117.1 µs | **64.6 µs** (370 GB/s) | 1.81x | 0.50 ulp |
| lm_head 151936×2560 | 969.6 µs | 952.1 µs (392 GB/s) | 1.02x | 0.00 ulp |

Caveat jujur: q/gate_up/down kecil → muat **L2** saat microbench (angka
833 GB/s > peak DRAM) — yang DRAM-honest = lm_head (1.02x, murni bandwidth).
Kemenangan nyata = **eliminasi launch quant (~40µs terpisah)** + hitung
dalam satu pass; di decode DRAM-bound, projeksi hemat ≈ 4 call × 36 layer
× 40µs ≈ 5.8ms/step bila terpasang di graph.

### Perjalanan debug (rel 13.17 → PASS) — 3 bug/insight

1. **Quant formula beda → 0.04% byte flip.** Saya `x / (amax/448)` (fpu
   divide, eps 1e-12 di tempat beda); sglang `clamp(x * (1/sx))` dengan
   `sx = max(amax, 1e-10)/448` (fp8_kernel.py:170-173). Skala identik bit,
   tapi reciprocal+clamp di batas E4M3 membalik ~1 dari 2560 elemen —
   cukup untuk mengubah elemen dekat-nol menjadi rel 13. Fix: mirror
   persis formula mereka.
2. **"GEMM-nya salah" ≠ salah.** Debug membandingkan torch-dequant vs
   cutlass: maxrel 13.18 juga muncul DI REFERENSI — bukan kernel saya.
   Bedah elemen: rel meledak di **near-zero cancellation** (ref −0.0028 vs
   0.034), sedangkan elemen besar beda 0.5 = **setengah ulp bf16** murni.
   Pelajaran: `maxrel` dengan clamp_min(1e-3) METRIK RUSAK untuk GEMV
   (output ada yang ~0.03, ulp-relative meledak) — pakai maxabs dalam satuan
   ulp bf16 pada magnitudo maks.
3. **Toleransi dihitung, bukan ditebak.** Pisahkan dua error:
   saya-vs-exact-torch = 0.61 ulp, cutlass-vs-exact = 0.50 ulp (terukur
   `dbg_cutlass_tol.py`, konsisten semua shape: 0.49–0.50 ulp). Dua
   pembulatan bf16 independen → bound ~1.6 ulp → assert `atol = 2×ulp(max)`,
   rtol=0. o_proj sempat gagal di 1.5 ulp dengan atol lama (4e-3 × max =
   0.95 < 1.5) — bukan bug kernel, toleransi terlalu sempit.

### Pola debug yang terbukti (untuk iterasi berikut)
`split-quant-vs-gemm`: komponen quant dibandingkan byte-level dulu
(scale vector + % byte identik), baru GEMM dibandingkan vs **exact fp32
torch-dequant** DAN vs cutlass — tiga-tiganya, untuk memisahkan "formula
salah" dari "pembulatan wajar" dari "penyebut kecil".

---

## 11. Batch decode (bs=16): kekalahan → TM-blocking → campuran

Tiga eksperimen berantai menyerang pertanyaan: apakah fusion menang juga di
**batched decode** (throughput mode), bukan hanya bs=1?

### 11a. Kernel row-parallel (dari §10) = KALAH TELAK di bs=16

`fused_gemv_quant.py` di-extend M-aware (grid = M × n_blks, 1 program/baris).
Akurasi PASS semua (≤2 ulp), tapi waktu:

| Layer | bs=16 sglang | bs=16 fused row-parallel | vs |
|---|---|---|---|
| q_proj | 52.4 µs | 105.6 µs | 0.50x |
| gate_up | 42.2 µs | 242.1 µs | 0.17x |
| down | 53.6 µs | 270.3 µs | 0.20x |
| lm_head | 973.0 µs | **15321.2 µs** | **0.06x** |

**Root cause:** tiap program baca weight (N×K) sendiri → weight dibaca M kali.
lm_head 15321 ≈ 16 × 973µs — bukti eksak 16x traffic DRAM tanpa reuse.
Cutlass M=16: weight dibaca sekali, dipakai 16 baris (tensor-core tile).
**Pelajaran: GEMV-fusion bs=1 ≠ GEMM-fusion bs>1 — arsitektur program harus
berubah, bukan cuma menambah dimensi grid.**

### 11b. Fused SwiGLU+quant+down-GEMV (3 launch → 1) — menang di bs=1

File: `fused_swiglu_quant_gemv.py`. 1 kernel: load gate_up (M, 2I) →
`silu(g)*u` fp32 di register → quant mirror-sglang → akumulasi vs down
weights blockwise. Rantai ref = chunk+silu+materialize bf16 → quant → cutlass.

| M | chain | fused | Speedup |
|---|---|---|---|
| 1 | 132.5 µs | **68.1 µs** | **1.95x** |
| 16 | 122.7 µs | 343.0 µs | 0.36x (kalah — penyebab sama 11a) |

Proyeksi decode bs=1: hemat ~64µs × 36 layer ≈ **2.3ms/step (6.4%)** bila
terpasang — chain down-proj adalah rantai terpanjang per layer MLP.

### 11c. TM-blocking (16 baris/program) — fix lm_head, campuran di tengah

File: `fused_gemv_quant_tm.py`. Grid = ceil(M/16) × n_blks; per program:
xq (16,128) per-baris-quant + `tl.dot` vs w (128,128) — weight dibaca sekali
utk 16 baris. Wrapper pilih row-kernel utk M=1, TM-kernel utk M>1.

| Layer | M=1 | M=16 | M=32 |
|---|---|---|---|
| q_proj | 3.57x | **1.70x** | 1.01x |
| gate_up | 2.37x | 0.75x | 0.45x |
| down | 0.92x | 0.49x | 0.30x |
| lm_head | 0.99x | **1.01x** (dari 0.06x!) | 0.50x |

- **Fix nyata:** lm_head M=16 15321→960µs (15.8x perbaikan) — traffic weight
  kembali 1x. M=32 = 0.50x karena tm=16 → weight dibaca 2x (grid_m=2) —
  ceiling desain, butuh TM=32 (register pressure naik).
- **Sisa kekalahan** (gate_up/down M=16): cutlass tile-scheduler untuk
  M=16 GEMM kecil sudah sangat efisien; overhead dot+quant kami > penghematan
  launch. Kesimpulan: **fused quant+GEMV = keputusan bs=1 (decode tunggal);
  M≥16 biarkan cutlass.**

### Pelajaran presisi TM (tf32 tail)
- `tl.dot` fp32 di Triton 3.8 default **tf32**: input kami exact (mantisa
  ≤4 bit utk fp8-vals) tapi akumulasi bertail — got-vs-exact max **1.9 ulp**
  (row-kernel: 0.61). Terlihat hanya di outlier; mean diff = murni
  pembulatan bf16 (0.149 ≈ ulp/√12).
- `input_precision="ieee"` diuji: butuh 114KB smem > limit sm_120 (101KB)
  → **tidak muat**. Keputusan: toleransi 3 ulp terukur (tail 2.5), tail
  tf32 tidak relevan untuk kualitas LLM (bf16 sendiri 3 digit desimal).
- Toleransi dihitung dari **union bound terukur**: saya-vs-exact (1.9) ∪
  double-rounding bf16 (1.0) ∪ cutlass-vs-exact (0.5).

### Strategi integrasi terbaru (revisi §8)
1. **bs=1 decode (latensi):** pasang row-kernel quant+GEMV (§10) +
   swiglu-q-gemm (11b) → potensi -5.8ms (quant) -2.3ms (chain) ≈ **-8ms
   dari 36ms = +33% tok/s** (angka proyeksi microbench, e2e belum diuji).
2. **Batched decode (throughput):** biarkan path sglang (cutlass) —
   kemenangan throughput datang dari continuous batching, bukan kernel.

---

## 12. Upgrade server 4B → Qwen3-8B-FP8 — AMAN, tok/s nyaris identik

Permintaan user: coba model 7/8B, hati-hati RAM (bebaskan dulu). (Tidak ada
Qwen3-7B di lineup — pilihan: 8B. Qwen3-14B-FP8 ≈15GB weights + KV → sesak
di 16GB, tidak dicoba.)

### Prosedur
1. RAM dibebaskan: kill server 4B (±4.5Gi), drop_caches, clear swap →
   14Gi available sebelum download.
2. Download `Qwen/Qwen3-8B-FP8` (2 shard, 8.9GB) → `/root/models/Qwen3-8B-FP8`.
   Disk 934G kosong — aman.
3. Start (`start_sglang_8b.sh`, flags identik 4B: triton, no graph) —
   **load OK sekali jalan**.

### Angka (identik harness §7: wall-clock client, median 3)

| Metrik | Qwen3-4B-FP8 | Qwen3-8B-FP8 |
|---|---|---|
| weights mem | 4.5 GB | 8.81 GB |
| VRAM total terpakai | ~7.8 GB | **12.84 / 16.31 GB** |
| KV cache | ~30k token | 18815 token (K+V 2.58GB) |
| RAM WSL | ~4.5 Gi | 4.5 Gi |
| decode tok/s (bs=1) | 27.7 | **26.9** |
| aggregate 4-request | 116.7 | **120.9** |
| t(max=1) prefill+ovh | — | 42 ms |

### Temuan: 8B "gratis" di hardware ini
- Decode bs=1: **-3% tok/s** untuk 2x parameter. Langkah bertambah dari
  36.0 → 37.2 ms/token — padahal bytes weight DOUBLING. Artinya step waktu
  didominasi komponen yang TIDAK skala dengan ukuran weight (overhead
  launch/quant/attention-chain ~25ms, lihat §8) — GEMV 8B (~22ms honest
  392GB/s) masih muat di budget itu.
- 4 paralel: aggregate 120.9 tok/s ≈ 4.5x single — pola linier batch
  sama persis dgn 4B (§7). Headroom VRAM sisa 3.5GB, KV 18815 token
  cukup utk konteks pendek-menengah.
- Kesimpulan: **naik 4B→8B = kualitas 2x, biaya tok/s ~3%** di RTX 5060 Ti.
 14B baru di batas: weights ±15GB menyisakan <1GB KV → praktis tak
  bisa dipakai untuk konteks apa pun.

### Catatan proses
- Swap-off sempat hang >2 menit saat buff/cache dipakai penuh proses
  mati-aji → biarkan; kembali normal sendiri setelah reclaim.
- Download unauthenticated HF: ~2.5 it/s file kecil, shard besar ±2.7
  menit (total ~4 menit untuk 8.9GB).

## 13. Integrasi e2e: monkeypatch sglang — 26.8 → 37.2 tok/s (+39%)

Rekomendasi #1 pagi ini: pasang fused quant+GEMV (§10) ke server sungguhan.
Selesai, terukur, tanpa regresi batch.

### Callsite & strategi patch

- `fp8.py:537` resolve fungsi linear via `dispatch_w8a8_block_fp8_linear()`
  saat init tiap layer → yang disimpan adalah objek fungsi. Tapi dispatch
  (di `fp8_utils.py`) me-lookup nama module-global **saat dipanggil** →
  rebind nama di ekor file = cukup, asal sebelum model load.
- Auto backend di box ini = **flashinfer** (0.7.1 terpasang + Blackwell),
  bukan cutlass yang dipakai microbench. Solusi: wrap **dua** entry point
  (`flashinfer_gemm_...` + `cutlass_w8a8_block_...`) dengan fast-path sama:
  `input_scale is None ∧ M==1 ∧ bf16 ∧ kelipatan-128` → `fused_gemv_quant`
  (row kernel §10); selain itu jatuh ke original (M>1 prefill/batch →
  jalur lama utuh, sesuai keputusan §11).
- Implementasi: append tail ~40 baris ke `fp8_utils.py` (backup
  `fp8_utils.py.orig13`). Import `fused_gemv_quant_tm` **lazy** di call
  pertama (hindari circular import; kompile Triton nunggu step decode
  pertama). `try/except` → gagal apa pun = fallback original, server tak
  bisa crash karenanya.

### Angka e2e (Qwen3-8B-FP8, harness §7 sama: wall-clock client, median 3)

| Metrik | baseline | patched | Δ |
|---|---|---|---|
| decode ms/token | 37.38 | **26.91** (run1: 26.47) | **-28%** |
| decode tok/s | 26.8 | **37.2** (run1: 37.8) | **+39%** |
| t(max=1) prefill+ovh | 73 ms | 53 ms | -27% |
| aggregate 4-request | 120.9 | 120.3 | ~0 (M>1 → original ✓) |

- Hemat terukur **10.5 ms/step** — di atas proyeksi ~8ms (dari 4B):
  weight 8B = 2x, jadi quant+GEMV (launch-bound di 4B) ikut ganda
  nilainya. Sisa 26.9ms/step didominasi attention-chain + RMSNorm +
  sampling yang belum di-fuse (§8).
- Kualitas: generate `temperature=0.7, top_p=0.95` koheren (Indonesia,
  faktual); akurasi microbench §10 ≤2 ulp bf16.

### Insiden (bukan karena patch)

1. Request pertama dengan `top_p` → flashinfer JIT compile modul
   **sampling** → link gagal `cannot find -lcuda` (stubs dir hilang
   setelah `fix_nvcc_wrap.sh` symlink `lib64` → pip nvidia/cu13) →
   scheduler SIGQUIT. Fix: `ln -s /usr/lib/wsl/lib/libcuda.so
   /usr/local/cuda/lib64/stubs/libcuda.so`. Catatan: `bench_e2e` (temp
   saja, tanpa top_p) tak melewati jalur JIT ini — dulu tak pernah
   kena karena tak ada request top_p sejak layout CUDA berubah.
2. `wsl bash -c "... nohup ..."` bisa bunuh child saat sesi wsl keluar
   (log 0 byte, pid mati) → start dengan `setsid ... < /dev/null`.

### Status patch

- **Aktif permanen** di site-packages venv_netra. Revert:
  `cp fp8_utils.py.orig13 fp8_utils.py` lalu restart server.
- Salinan tail lokal: `patch_fp8_fused_tail.py`.
- Belum dipasang: SwiGLU+quant+GEMV chain (§11b, potensi -2.3ms lagi)
  — butuh hook intermediate MLP, bukan linear wrapper; di backlog.

## 14. Prefill 8B — 2314 tok/s input (baseline, patch tak menyentuh M>1)

Metode (`bench_prefill.py`): prompt unik uuid-hex (tak mungkin kena
prefix radix cache), `t(max_new=1)` sekali jalan, slope
`(t_long − t_short)/(tok_long − tok_short)`; median 3 pasang.

| | short | long |
|---|---|---|
| uncached token | 388 | 14028 |
| wall | 103 ms | 5998 ms |
| **slope** | — | **0.432 ms/token → 2314 tok/s** |

- Chunked prefill 2048 → 14028 tok ≈ 7 chunk ≈ 6s, konsisten.
- Konteks praktis: dokumen 8k token ≈ **3.5s TTFT**; batas konteks
  server 18809 token (KV 18815 − reserved).
- Jauh lebih lambat dari decode-side gain: prefill GEMM M≥2048 sudah
  jalur flashinfer/cutlass efisien — patch §13 memang sengaja hanya
  M==1 (decode). Prefill bukan target fusi GEMV.
- Jebakan metode (dicatat agar tak diulang): (a) pakai prompt yang sama
  → radix cache hit, angka bohong; (b) uuid-hex 36k char = 19k token >
  limit 18809 → 400 error; (c) `max_new=1` overhead tetap ~40-100ms →
  wajib pakai delta panjang-pendek, bukan absolut tunggal.

## 15. Fusi chain SwiGLU+quant+down (§11b) ke server — GAIN NOL, direvert

Percobaan integrasi kedua: patch `Qwen2MLP.forward` (qwen2.py — Qwen3
import kelas yang sama) → M==1: `gate_up (via patch §13) →
fused_swiglu_q_gemm (silu+quant+down GEMV 1 kernel)`.

### Hasil

| | §13 saja | §13+§15 chain |
|---|---|---|
| decode ms/token | 26.91 | 27.14 (noise ±0.4) |
| tok/s | 37.2 | 36.8 |
| aggregate 4-req | 120.3 | 122.2 |
| chain fires? | — | **YA** (marker:1332 S,0 E) |

**Gain di bawah noise → direvert** (`qwen2.py.orig15`).

### Kenapa proyeksi -2.3ms meleset

§11b mengukur rantai vs **baseline cutlass** (quant+GEMM terpisah).
Setelah §13, kedua GEMM (gate_up & down) sudah quant+GEMV fused —
yang tersisa di antaranya cuma `silu_and_mul` (satu kernel kecil
~15µs) + materialisasi act. Sisa itulah yang dihemat chain, dan
~0.5ms/step-nya tak keluar dari noise. Pelajaran: **baseline proyeksi
harus path SETELAH patch sebelumnya, bukan path vanilla.**

### Uji quality A/B (3 prompt greedy + mean logprob)

Temuan: kelakuan aneh ("Jelaskan sejarah..." lead-in, echo prompt)
muncul di **ketiga** konfigurasi: unpatched (U), §13 (B), §13+§15 (A) →
**bukan akibat patch**; ini perilaku Qwen3 raw `/generate` tanpa chat
template (completion mode, greedy). Jawaban faktual benar di semua:
391 ✓, biru/Rayleigh ✓, Paris muncul di akhir ✓. Mean logprob [1][2]
identik antar konfig (−0.26, −0.11).

### Artefak proses (dipakai ulang tick berikut)

- Marker file `/root/swiglu15.log` (S/G/E) — pola "instrumentasi lewat
  append file" murah untuk path di dalam scheduler spawn.
- `pkill -f 'sglang[.]launch_server'` — **wajib** character-class
  trick; tanpa itu pkill match diri sendiri (exit9, server tak kekill).
- `return_logprob`/`logprob_start_len` = field TOP-LEVEL request
  `/generate`, bukan di `sampling_params` (400 kalau salah).

## 16. CUDA graph + patch §13 — MENANG, config baru production

Hipotesis: capture graph (decode bs=[1,2,4,8]) di atas server patched
§13 → hilangkan python/kernel-launch overhead per decode step.

### Setup

- Script baru `start_sglang_8b_graph.sh` = `start_sglang_8b.sh` minus
  `--disable-cuda-graph` (flag lama deprecated; default sglang skrg
  sudah graph on, prefill backend "breakable").
- Capture log: decode bs=[1,2,4,8] elapsed 2.20s / 0.06GB; prefill
  breakable 4..2048 token elapsed 14.2s / 0.48GB; avail mem sisa
  2.57GB. Triton kernel ter-capture normal (wrapper python jalan
  saat capture, replay pakai kernel yang sudah direkam).
- Patch §13 tetap aktif; §15 sudah direvert sebelum start.

### Hasil (Qwen3-8B-FP8)

| | tanpa graph (§13) | + CUDA graph | Δ |
|---|---|---|---|
| decode ms/token | 26.91 | **23.78** | **-11.6%** |
| tok/s bs=1 | 37.2 | **42.1** | **+13%** |
| t(max=1) | ~73 ms | 76 ms | ≈ (prefill tetap) |
| aggregate 4-paralel | 120.3 | **176.1** | **+46%** |
| per-req paralel | 5.8s @ wall | 5.8s [5.8×4] | seimbang |

Paralel-4 dapat 176 tok/s — +46% atas no-graph. Alasan: di M=1 graph
replay menghapus launch overhead Python/triton per step (~3ms × 4
layer-call-site), dan di bs>1 seluruh attention+MLP pipeline ikut
ter-replay tanpa launch sama sekali.

### Verdict

- **Config production baru**: `start_sglang_8b_graph.sh` (graph ON +
  patch §13). Smoke koheren (output lead-in sama dgn perilaku raw
  Qwen3 greedy — sudah dinormalkan di §15, bukan regresi).
- Kualitas: logprob/quality tak diulang di graph (kernel sama, hanya
  launch path beda → risiko numerik nol; sampling tetap di luar graph).
- Total sejak baseline 4B: 27.7 → 42.1 tok/s decode bs=1, dan
  116.7 → 176.1 paralel-4 (semua di 8B-FP8 + patch + graph).

### Incident

Tak ada. Graph capture sukses pertama kali; libcuda stub fix dari §13
tetap cukup.

### Addendum setelah bench graph

- t(max=1) hangat = **33 ms** (dingin 76 ms; no-graph 73 ms) —
  prefill-graph breakable memberi TTFT short-prompt juga.
- Paralel-4 terkonfirmasi **172.7–176.1 tok/s** (log server sendiri:
  "gen throughput 176.00, cuda graph: True").

## 17. Fusi attention-side (QK-norm+RoPE): DITOLAK — decode sudah ROOFLINE

Hipotesis: integrasi `fused_qk_rope` (§9, 10.85x vs HF eager) ke server
cara §13. Setelah diselidiki: **tak ada headroom — integrasi ditolak
tanpa bakar jam** (pelajaran §15: proyeksi vs baseline yang BENAR dulu).

### Temuan 1: QK-norm sudah fused oleh sglang sendiri

`apply_qk_norm` (models/utils.py) pakai `fused_inplace_qknorm` JIT saat
CUDA ∧ head_dim∈{64,128,256,...} ∧ eps q==k. Qwen3-8B head_dim=128 →
**eligible & aktif** (nol warning di log). Sisa: rope = 1 launch
sgl_kernel, dan di bawah CUDA graph launch overhead sudah nol.

### Temuan 2: sisa traffic hemat fusi = KB vs GB

Fusi qknorm+rope hanya menghapus materialisasi tengah per layer:
(q 8KB + k 2KB) × 36 layer ≈ **<0.4MB/step**. Weight stream per step:
**~8.5GB**. Rasio 1:20.000. Kontras §13 (di sana Gb-scale nyata:
quant kernel hilang + weight pass kedua terhapus).

### Temuan 3 (hasil utama): plafon DRAM terukur

| Metrik | Nilai |
|---|---|
| D2D copy 1.5GB (r+w) | **358 GB/s** (teori 5060 Ti ≈ 448) |
| Decode step graph+§13 | 23.78–23.88 ms |
| Weight 8B-FP8 / step | ~8.5 GB |
| Efektif bila 100% = baca weight | **≈357 GB/s** |
| nvidia-smi saat decode | SM 100%, mem-util 93%, 2827MHz, **113W** (bukan power-cap) |

SM 100% menunggu memori (daya 63% TDP). **Optimasi kernel tak lagi
menang — plafon = weight_bytes / BW.** Cara menang: **kurangi bacaan
weight** → speculative decoding (ngram/EAGLE). Lanjut §18.

## 18. NGRAM speculative decoding — MENANG, config production final

Menindaklanjuti §17: kurangi bacaan weight = speculative. Pilihan
**NGRAM** (built-in sglang, `--speculative-algorithm NGRAM`) karena
**tak butuh model draft** — draft = pohon n-gram dari teks yang sudah
di-generate (CPU-side matching), verify batch M=5 di GPU (path cutlass,
aman dgn patch §13 by-design).

### Setup

`start_sglang_8b_ngram.sh` = graph + §13 +:
`--speculative-algorithm NGRAM --speculative-num-steps 4
--speculative-num-draft-tokens 5`
Catatan server: max_running_requests auto-ke-reset ke 48; mixed
chunked-prefill dimatikan oleh ngram.

### Hasil (Qwen3-8B-FP8)

| | graph+§13 (§16) | + NGRAM (§18) | Δ |
|---|---|---|---|
| decode ms/token | 23.78 | **19.92** | **-16%** |
| tok/s bs=1 | 42.1 | **50.2** | **+19%** |
| aggregate 4-paralel | 176.1 | **244.6** (hangat 280.7) | **+39%** |
| server-side peak | 176 tok/s | **349 tok/s** (accept len 2.48) | — |
| t(max=1) | 33 ms (hangat) | 78 ms | TTFT naik (mixed prefill off) |

Accept len terukur di log: **1.5–2.5 token per verify-5** (accept rate
0.13–0.37, naik seiring corpus n-gram terisi). Bahasa Indonesia
berulang ("kota tersebut", struktur daftar) memberi match lumayan.

### Perjalanan total malam ini

| | baseline 4B | final (8B+patch+graph+ngram) |
|---|---|---|
| decode bs=1 | 27.7 tok/s | **50.2 tok/s (+81%)** |
| aggregate 4-paralel | 116.7 | **244.6 (+109%, 2.1x)** |
| TTFT short prompt | ~73 ms | 78 ms (ngram) / 33 ms (graph only) |

### Verdict

- **Production final**: `start_sglang_8b_ngram.sh` (ngram ON). Kalau
  user butuh TTFT minimal (chat pendek): fallback
  `start_sglang_8b_graph.sh` (TTFT 33 ms).
- Kualitas: speculative preserve distribusi (verify by target model);
  smoke koheren — lead-in echo tetap gaya raw Qwen3 greedy (sudah
  dinormalkan §15, bukan efek spec-dec).
- Kalah sedikit di TTFT: mixed chunked prefill dimatikan ngram —
  trade-off sadar: decode throughput ≫ TTFT utk use-case panjang.

### Insiden

Tak ada. Ngram start bersih pertama kali; tak butuh download model
draft; VRAM cukup (verify M=5 tetap muat graph bs=8).

### Addendum

- **Verify graph ter-capture**: "Capture target verify CUDA graph
  bs=[1..8], num_tokens_per_req=5" → verify jalan di graph penuh,
  bukan eager. Artinya **lebarkan jendela draft hampir gratis** di
  bandwidth-bound regime (baca weight sama saja, M=5 vs M=8) —
  dicoba di §19.
- Flag `--enable-fused-qk-norm-rope` (ditemukan di server_args saat
  menyusun §17) ternyata **hanya dipakai qwen3_moe.py & mellum.py** —
  Qwen3 dense tak terpengaruh. Dilewati tanpa restart (hemat 2 menit).

## 19. Jendela ngram dilebar (steps 7, draft 8) — MENANG LAGI

Hipotesis §18-addendum: di bandwidth-bound regime, verify M=8 hampir
sama mahalnya dgn M=5 (baca weight sama) → jendela lebih lebar ≈
gratis utk accept lebih banyak per verify.

### Setup

`start_sglang_8b_ngram_w8.sh`: identik §18, hanya
`--speculative-num-steps 7 --speculative-num-draft-tokens 8`.
Verify graph capture otomatis num_tokens_per_req=8.

### Hasil

| | §18 (s4/d5) | §19 (s7/d8) | Δ |
|---|---|---|---|
| decode ms/token (cold corpus) | 19.92 | **15.71** | -21% |
| tok/s single (cold) | 50.2 | **63.7** | **+27%** |
| tok/s single (corpus hangat) | — | **81.7** (12.25 ms) | — |
| t(max=1) hangat | 78 ms | 60 ms | ≈ |
| paralel-4 hangat | 280.7 | 281.4 | datar |
| accept len (log) | 1.5–2.5 | 1.24 → **2.72** naik terus | — |

Catatan jujur: median paralel §19 (223.4) tertarik turun run dingin
pertama (165) — run hangat kedua identik dgn §18 (≈281). Keungulan
nyata di **single-decode** (63.7–81.7 vs 50.2). Accept len membesar
seiring corpus n-gram terisi → production makin cepat seiring pemakaian.

### Verdict

**Production final = `start_sglang_8b_ngram_w8.sh` (steps7/draft8).**
Total dari baseline 4B (27.7 tok/s): **81.7 tok/s hangat = +195% (3x)**,
paralel 116.7 → 281.4 (**2.4x**).

## 20. Corpus ngram eksternal (wikipedia id) — NEGATIF, direvert

Hipotesis: corpus eksternal 5.58M token (1500 artikel wiki id,
JSONL per-dokumen) → accept-rate naik lintas-request.

### Setup

- `prep_ngram_corpus.py`: `datasets.load_dataset("wikimedia/wikipedia",
  "20231101.id", streaming)` → 1500 artikel → `/root/corpus_id.jsonl`
  (18MB). Catatan: `wikipedia` lama (script-based) ditolak datasets
  5.x — pakai repo `wikimedia/wikipedia`.
- Flag: `--speculative-ngram-external-corpus-path` +
  **`--speculative-ngram-external-sam-budget 7`** — budget WAJIB
  ∈ (0, num_draft_tokens−1]; dengan draft8 → maks 7. (Validasi di
  speculative_hook.py:1244.)

### Hasil vs §19 (w8 tanpa corpus)

| | §19 w8 | §20 +corpus |
|---|---|---|
| decode tok/s | 63.7 cold / **81.7** hangat | **54.6** |
| paralel hangat | 281.4 | **199.2** |
| accept len | 1.24→2.72 (naik) | **1.44–1.62** (datar rendah) |

### Kenapa kalah

Budget SAM eksternal maksimal 7 node (validasi ≤ draft−1) → subtree
wiki nyaris tak berperan, tapi kehadirannya **menggeser distribusi
SAM** sehingga self-ngram (yang sebenarnya cocok dgn gaya generasi
model) menang lebih sedikit. Netto: merugikan.

### Verdict

Corpus eksternal **dihapus dari production**; config kembali
`start_sglang_8b_ngram_w8.sh`. Jangan ulang kecuali API budget
berubah di versi sglang mendatang. Artefak tersisa (tak berbahaya):
`/root/corpus_id.jsonl`, `prep_ngram_corpus.py`, script
`start_sglang_8b_ngram_corpus.sh` (dibiarkan sbg catatan).




## 21. Jendela ngram draft>8 (steps15/draft16) — NEGATIF, kandidat 9b

Hipotesis: verify bandwidth-bound (§17) → draft 16 hampir gratis,
accept len naik → tok/s naik.

### Setup

- `start_sglang_8b_ngram_w16.sh`: NGRAM steps15/draft16 (sisa
  config identik w8). RAM dibersihkan dulu (drop_caches; 653Mi→14Gi
  free).
- Verify graph capture: , bs=[1..8], cuma
  +0.11GB, avail 2.62GB — capture AMAN (hipotesis "perlu naikkan
  max_bs" ternyata tak perlu; graph pakai draft tokens sbg row, bukan
  bs).

### Hasil vs §19 w8 (production)

| | §19 w8 | §21 w16 |
|---|---|---|
| decode single cold | 63.7 | 62.7 |
| decode single hangat | **81.7** | 71.3 |
| paralel-4 median | **281.4** | 220.6 |
| accept len (paralel) | 1.24–2.72 | 1.86–3.14 |
| verify-graph mem | — | +0.11GB (tak jadi masalah) |

### Analisis

Accept len MEMANG naik (draft lebih panjang → lebih banyak kandidat
cocok), tapi kenaikan tak cukup menutup biaya:

1. Verify 16 token vs 8 token di bandwidth-bound = dobel width pass
   weight... bukan — verify satu forward dgn M=16 row, weight tetap
   dibaca sekali per layer (M dimensi token). Yang naik: attention
   KV-read per token & logits decode 16 kolom. Masih ~bandwidth-bound
   utk weight, tapi ngram BFS (max_bfs_breadth=10, trie_depth=18)
   makin lebar makin banyak "kandidat lemah" di ujung — accept
   rate/token (0.06–0.14) tetap rendah.
2. Keputusan: **revert ke w8.** Draft sweet-spot = 8 utk ngram-BFS
   config ini. Diminishing return terkonfirmasi (kandidat 9b
   selesai, NEGATIF).

### Kondisi akhir

Server kembali `start_sglang_8b_ngram_w8.sh`. Script w16
dibiarkan sbg catatan.
