# Laporan Pagi — Sesi Malam 17–18 Okt 2026

Rangkuman semua eksperimen malam ini. Detail teknis lengkap:
[`TRITON_KERNELS.md`](TRITON_KERNELS.md) §8–§21.

## TL;DR

1. **Fused quant+GEMV PASS** (§10) — menang 1.8–4.9x di bs=1; debug saga
   rel-13.17 berakhir di 3 bug/insight (formula quant beda, metrik maxrel
   rusak, toleransi harus dihitung).
2. **Kalah di batch bs=16** (§11a) → root cause: row-parallel tanpa reuse
   weight (lm_head 15.3ms = 16×973µs) → **TM-blocking fix** (§11c):
   lm_head M=16 kembali 960µs. Sisa layer kecil: cutlass masih menang —
   keputusan: fusion = untuk bs=1 saja.
3. **Fused SwiGLU+quant+down-GEMV** (§11b): 1.95x di bs=1 → proyeksi
   hemat ~2.3ms/step.
4. **Server naik 4B → Qwen3-8B-FP8** (§12, permintaan user): **AMAN** —
   VRAM 12.8/16.3GB, RAM 4.5Gi, dan tok/s nyaris identik
   (26.9 vs 27.7 single; 120.9 vs 116.7 paralel-4). **8B = upgrade gratis.**
5. **★ INTEGRASI e2e BERHASIL** (§13): quant+GEMV dipasang ke server
   sungguhan via monkeypatch → **26.8 → 37.2 tok/s (+39%), decode
   -28% ms/token (37.4→26.9)**, batch paralel tak regres (120.3≈120.9),
   output koheren. Tonggak awal — dikalahkan §16/§18/§19 di bawah.
6. **SwiGLU chain (§15) NEGATIF** — proyeksi -2.3ms dari baseline
   vanilla; di atas §13 sisa saving cuma silu (~0.5ms, noise). Chain
   fires 100% tapi 27.14 vs 26.91 ms/token. **Direvert.** A/B quality
   3-way buktikan echo = perilaku Qwen3 raw greedy, bukan patch.
7. **★ CUDA graph + §13 = CONFIG BARU** (§16): decode **23.78 ms/token
   / 42.1 tok/s** (graph ON), paralel-4 **176.1 tok/s (+46%)**.
   Total dari baseline 4B: **27.7 → 42.1 tok/s decode**, **116.7 →
   176.1 paralel**.
8. **Roofline ditemukan** (§17): decode 8B = **≈357 GB/s efektif vs
   358 GB/s terukur D2D** → sudah di plafon DRAM. QK-norm ternyata
   sudah difuse sglang; fusi attention-side ditolak (hemat KB vs
   weight 8.5GB). **Plafon = baca weight; cara menang = speculative
   decoding** → dicoba (§18 NGRAM).
9. **★ NGRAM spec-dec MENANG** (§18): **50.2 tok/s decode (+19%)**,
   paralel **244.6 (hangat 280.7, +39%)**, accept len 1.5–2.5/verify-5.
   Tak butuh model draft.
10. **★★ Jendela dilebar (§19, steps7/draft8)**: single **63.7 cold →
    81.7 tok/s hangat**, paralel hangat 281. **Total dari baseline 4B:
    27.7 → 81.7 tok/s (+195%, 3x); paralel 116.7 → 281.4 (2.4x).**
    Verify lebih lebar hampir gratis di bandwidth-bound.
11. **Corpus wiki id eksternal (§20) NEGATIF** — budget SAM dibatasi
    ≤ draft−1 (7 node) → wiki nyaris tak dipakai malah menggeser
    self-ngram: accept 1.5 vs 2.7, 54.6 vs 81.7 tok/s. Direvert.
12. **Draft>8 (§21, kandidat 9b) NEGATIF** — steps15/draft16: accept
    len naik (1.9–3.1) tapi tok/s turun (71.3 vs 81.7 single, 220.6
    vs 281.4 paralel). Verify graph 16 aman (+0.11GB). Sweet-spot
    draft tetap 8. Direvert.
13. Fused sampling tervalidasi ulang di logits asli 8B: TV 0.0012 PASS.

## Bukti kernel — 4B & 8B (ringkas)

**1. Micro-benchmark (bs=1, µs per panggilan) — logika kernel terbukti benar:**

| Kernel | Baseline | Fused | Speedup | Akurasi |
|---|---:|---:|---:|---|
| quant+GEMV `q` | 138 µs | 41 µs | **3.38x** | PASS 4B & 8B |
| quant+GEMV `gate_up` | 271 µs | 55 µs | **4.91x** | PASS 4B & 8B |
| quant+GEMV `lm_head` | 953 µs | 937 µs | 1.02x | PASS |
| SwiGLU+quant+GEMV (§11b) | 132 µs | 68 µs | **1.95x** | PASS |
| QK-norm+RoPE+GQA (§9) | 112 µs | 10 µs | **10.85x** | PASS 4B & 8B |
| Sampling real-logits | 8–16 µs | 1–2 µs | **2–8x** | TV 0.0211/0.0012 |

Semua fused kernel diuji dengan **real-logits kedua checkpoint** (bukan tensor random).

**2. E2e server nyata — dampak ke user (tok/s decode bs=1):**

| Tahap | 4B | 8B | Bukti |
|---|---:|---:|---|
| Polos | 27.7 | 26.8 | baseline |
| + fused quant+GEMV (§13) | — | **37.2** | **+39%** = kernel menang di server |
| + graph (§16) | — | 42.1 | kernel tetap jalan di graph capture |
| + NGRAM w8 (§19) | — | **81.7** | production final |

Micro 5x → e2e +39% wajar: di server, GEMV cuma ~40% waktu decode
(sisanya attention/sampling/graph). Yang penting **arahnya naik di A/B
server yang sama** = terbukti, bukan bench kertas. QK-norm 11x tak
dihitung e2e karena sudah difuse sglang lain (§17).

## scoreboard kernel (bs=1 decode, kecuali dinyatakan lain)

| Eksperimen | Hasil | Status |
|---|---|---|
| Fused QK-norm+RoPE+GQA (§9) | decode **10.85x**, prefill 4.06x | menang besar |
| Fused sampling real-logits 4B/8B | TV 0.0211 / 0.0012 | tervalidasi |
| Fused quant+GEMV (§10) | q 3.38x · gate_up **4.91x** · lm_head 1.02x | menang bs=1 |
| Fused SwiGLU+quant+GEMV (§11b) | **1.95x** (132→68µs) | menang bs=1 |
| quant+GEMV M=16 row-parallel (§11a) | 0.06–0.5x | **NEGATIF** |
| quant+GEMV TM-16 blocked (§11c) | lm_head M=16 15.8x fix; tengah campuran | fix parsial |
| **Integrasi e2e server 8B (§13)** | decode **37.4→26.9 ms/token (+39% tok/s)** | **menang besar** |
| SwiGLU+quant+chain ke server (§15) | 27.14 vs 26.91 ms/token | **NEGATIF** (direvert) |
| **CUDA graph + §13 (§16)** | **42.1 tok/s · paralel 176** | **config baru** |
| Fusi QK-norm+RoPE (§17) | ditolak: sudah fused sglang + KB vs 8.5GB | analisis roofline |
| Roofline decode 8B (§17) | 357 GB/s efektif ≈ 358 GB/s terukur | **plafon DRAM** |
| **NGRAM spec-dec (§18)** | **50.2 tok/s · paralel 244.6** | **menang besar** |
| **NGRAM jendela lebar s7/d8 (§19)** | **63.7–81.7 tok/s single** | **menang besar** |
| Corpus wiki id eksternal (§20) | 54.6 tok/s · accept 1.5 · paralel 199 | **NEGATIF** (direvert) |
| Draft>8 w16 (§21) | 71.3 tok/s · accept 1.9–3.1 · paralel 220.6 | **NEGATIF** (direvert) |

## Angka server

| | 4B-FP8 | 8B-FP8 |
|---|---|---|
| VRAM | 7.8 GB | ~13.4 GB + graph/ngram (sisa ~2.6) |
| KV token | ~30k | 18.8k |
| decode bs=1 | 27.7 tok/s | **81.7 tok/s hangat** (w8 ngram+graph+§13; polos 26.8) |
| aggregate 4-paralel | 116.7 | **281.4** (hangat) |

## Pelajaran malam ini (paling berharga)

1. **Semua konstanta harus argumen/constexpr** — QMAX/EPS kena 4× total.
2. **maxrel itu metrik rusak** untuk GEMV (cancellation → penyebut kecil);
   pakai maxabs dalam satuan ulp bf16 pada magnitudo maks.
3. **GEMV-fusion ≠ GEMM-fusion** — menang di M=1 bisa jadi kalah 16x di
   M=16 kalau program tak reuse weight; TM-blocking wajib.
4. **Toleransi dihitung dari komponen terukur** (saya-vs-exact, cutlass-vs-
   exact, double-rounding), bukan ditebak.
5. **8B gratis di 5060 Ti** — step time didominasi overhead yang tak
   skala dengan ukuran weight.

## Langkah berikutnya (rekomendasi)

1. ~~**Integrasi e2e**~~ ✅ **SELESAI (§13)**: +39% tok/s 8B, batch aman.
2. ~~**SwiGLU chain**~~ ✅ **DICOBA (§15)**: gain 0 di atas §13, direvert.
3. ~~Prefill 8B~~ ✅ **TERUKUR (§14)**: **2314 tok/s input** — 8k-token
   doc ≈ 3.5s TTFT; patch tak menyentuh prefill (M>1 by design).
4. ~~**CUDA graph**~~ ✅ **MENANG (§16)**: 42.1 tok/s + paralel 176.
5. ~~Fusi attention-side~~ ✅ **DITOLAK (§17)**: sudah fused sglang;
   decode di roofline DRAM (357 ≈ 358 GB/s).
6. ~~**NGRAM spec-dec**~~ ✅ **MENANG (§18)**: 50.2 tok/s + paralel
   244.6.
7. ~~**Jendela ngram lebar**~~ ✅ **MENANG (§19)**: steps7/draft8 →
   **63.7–81.7 tok/s**. Production final = `start_sglang_8b_ngram_w8.sh`.
8. ~~**Corpus eksternal**~~ ✅ **NEGATIF (§20)**: budget SAM ≤7 node
   menggeser self-ngram → merugikan. Direvert.
9. Kandidat tick berikutnya (urut nilai):
   a. EAGLE draft (download ckpt ~0.6B) — accept-rate jauh lebih
      pintar dari ngram; tapi cek dulu: sglang tak punya jalur
      resmi qwen3+eagle (grep kosong), tak ada ckpt di
      /root/models → riset dulu sebelum commit.
   b. ~~Jendela draft >8~~ ✅ **NEGATIF (§21)**: w16 → 71.3 tok/s
      (vs w8 81.7). Sweet-spot draft = 8.
   c. ~~Rapikan dokumentasi~~ ✅ **SELESAI**: Bukti-kernel 4B/8B di
      MORNING_REPORT; §1–§21 utuh di TRITON_KERNELS.md.

## Kondisi sistem

- Server 8B **patched + graph + NGRAM w8** running di :30000
  (`start_sglang_8b_ngram_w8.sh`, pid di `/root/sglang.pid`). Fused
  quant+GEMV (§13) + CUDA graph + ngram steps7/draft8 aktif.
  Fallback: `start_sglang_8b_ngram.sh` (s4/d5), `start_sglang_8b_graph.sh`
  (tanpa spec, TTFT minimal 33ms). Revert penuh: `fp8_utils.py.orig13`
  + `start_sglang_8b.sh`.
- Fix lingkungan: symlink `libcuda.so` → stubs CUDA (flashinfer JIT
  sampling butuh `-lcuda` waktu request pertama pakai top_p).
- 4B tersedia di HF cache, tinggal revert script bila perlu.
- Semua file eksperimen di `/root/*.py`, salinan lokal di
  `D:\kerjaan\belajar\llm\`. Tidak ada proses yang perlu dikecualikan;
  Zed/Claude tak tersentuh.
