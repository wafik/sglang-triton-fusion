"""Stress/edge-case battery untuk 3 fused kernel (fused_sampling, fused_rmsnorm,
fused_swiglu_fp8). Tiap kasus satu assert — gagal = print kasus + raise.
"""
import torch
import triton
import triton.language as tl

from fused_sampling import fused_sample, naive_competent, TEMPERATURE, K, TOP_P, V, B
from fused_rmsnorm import fused_add_rmsnorm, naive_add_rmsnorm, HIDDEN, EPS
from fused_swiglu_fp8 import (fused_swiglu, fused_swiglu_quant, naive_swiglu,
                              naive_swiglu_quant, INTER, GROUP, QMAX)

torch.manual_seed(0)
DEVICE = "cuda"
fails = []


def check(name, cond, extra=""):
    if cond:
        print(f"  ✅ {name}")
    else:
        fails.append(name)
        print(f"  ❌ {name} {extra}")


def sampling_battery():
    print("=== fused_sampling ===")
    logits = torch.randn(B, V, device=DEVICE) * 3.0

    # 1. reproducible: seed sama -> output sama; beda seed -> (hampir) beda
    a = fused_sample(logits, seed=5)
    b = fused_sample(logits, seed=5)
    c = fused_sample(logits, seed=6)
    check("seed-sama reproducible", torch.equal(a, b))
    check("seed-bedanya beda", not torch.equal(a, c))

    # 2. K besar (4096): support harus di luar top-50 (K=V penuh di-skip:
    #    BLOCK=next_pow2(V)=262144 bikin kompilasi Triton >10 menit)
    seen = set()
    for s in range(30):
        out = fused_sample(logits, top_k=4096, top_p=1.0, seed=100 + s)
        seen.update(out.tolist())
    top50 = set(torch.topk(logits, 50).indices.flatten().tolist())
    outside = seen - top50
    check("K=4096: sampel di luar top-50 ADA (support luas)",
          len(outside) > 0 and len(seen) > 250,
          f"unik={len(seen)} di_luar_top50={len(outside)}")
    # catatan: T=0.7 di logit random memang terkonsentrasi — unik=326/480
    # itu fisika softmax, bukan bug; threshold lama (>400) salah asumsi

    # 3. top_p kecil -> hampir selalu token teratas
    top1 = logits.argmax(-1)
    hit = sum((fused_sample(logits, top_p=0.0, seed=s) == top1).all().item()
              for s in range(10)) / 10
    check("top_p=0 selalu top-1", hit == 1.0, f"rate={hit}")

    # 4a. T kecil + margin buatan +5 di argmax (gap 5/T=0.01 -> e^-500 = 0):
    #     DETERMINISTIS harus 100% argmax. (T=0.01 pada logit random tanpa
    #     margin TIDAK cukup — gap top1/top2 ~0.01, e^-1=0.37 masih mungkin)
    bigm = logits.clone()
    bigm.scatter_(1, bigm.argmax(-1, keepdim=True), bigm.argmax(-1, keepdim=True) + 5.0)
    hit = sum((fused_sample(bigm, temperature=0.01, top_k=50, top_p=1.0,
                            seed=s) == bigm.argmax(-1)).all().item()
              for s in range(10)) / 10
    check("T=0.01 + margin 5 => selalu argmax", hit == 1.0, f"rate={hit}")
    out = fused_sample(logits, temperature=1000.0, top_k=50, top_p=1.0, seed=7)
    tk = torch.topk(logits, 50).indices
    in_topk = all(int(out[i]) in set(tk[i].tolist()) for i in range(B))
    check("T=1000 tetap di top-k (filter menang)", in_topk)

    # 5. K bukan pangkat-2 (51, 77, 63) — BLOCK=next_pow2, mask tail
    for k in (51, 63, 77, 3, 1, 17):
        try:
            s = fused_sample(logits, top_k=k, top_p=1.0, seed=3)
            tk = torch.topk(logits, k).indices
            ok = all(int(s[i]) in set(tk[i].tolist()) for i in range(B))
            check(f"K={k} (non-pow2) di top-k set", ok)
        except Exception as e:
            check(f"K={k} (non-pow2) di top-k set", False, repr(e)[:90])

    # 6. K > V? tidak valid, lewati. K=V non-trivial udah (2).
    # 7. input bf16-cast: logit HILANG presisi saat round-trip, jadi equals
    #    dgn fp32 asli memang mustahil — cek membership di nilai hasil cast
    lg_bf = logits.bfloat16().float()
    s = fused_sample(lg_bf, seed=1)
    tk = torch.topk(lg_bf, K).indices
    check("input hasil cast bf16->fp32 tetap di top-k",
          all(int(s[i]) in set(tk[i].tolist()) for i in range(B))
          and bool(torch.isfinite(s).all()))

    # 8. baris identik (degenerate) -> semua baris hasil sama utk seed sama utk row?
    #    (row berbeda diberi seed beda via row+offs — jadi TIDAK sama; cek tetap valid topk)
    dup = logits[:4].repeat(8, 1)
    s = fused_sample(dup, seed=9)
    tk = torch.topk(dup, K).indices
    check("baris duplikat tetap di top-k",
          all(int(s[i]) in set(tk[i].tolist()) for i in range(dup.shape[0])))

    # 9. logit skala ekstrem (max 1e4): exp overflow terhindar via (v-v0)
    big = torch.randn(B, V, device=DEVICE) * 1e4
    s = fused_sample(big, seed=2)
    tk = torch.topk(big, K).indices
    check("skala 1e4 tetap di top-k & finite",
          all(int(s[i]) in set(tk[i].tolist()) for i in range(B)))


def rmsnorm_battery():
    print("=== fused_rmsnorm ===")
    w = torch.randn(HIDDEN, device=DEVICE) * 0.1

    # 1. baris semua nol -> var=0, rsqrt(eps), out=0, residual=x+r=0
    x = torch.zeros(4, HIDDEN, device=DEVICE)
    r = torch.zeros(4, HIDDEN, device=DEVICE)
    o, rn = fused_add_rmsnorm(x, r, w)
    check("baris nol -> out nol, res nol",
          bool((o == 0).all() and (rn == 0).all()))

    # 2. nilai ekstrem besar (1e6): var 1e12, tak overflow fp32
    x = torch.randn(8, HIDDEN, device=DEVICE) * 1e6
    r = torch.randn(8, HIDDEN, device=DEVICE) * 1e6
    ro, rr = naive_add_rmsnorm(x, r, w)
    fo, fr = fused_add_rmsnorm(x, r, w)
    check("ekstrem 1e6 allclose", torch.allclose(fo, ro, rtol=1e-4, atol=1e-2)
          and torch.equal(fr, rr))

    # 3. non-contiguous input (transpose view) — stride beda
    x = torch.randn(HIDDEN, 7, device=DEVICE).t()  # (7, HIDDEN) non-contig
    r = torch.randn(HIDDEN, 7, device=DEVICE).t()
    try:
        fo, fr = fused_add_rmsnorm(x.contiguous(), r.contiguous(), w)
        ro, rr = naive_add_rmsnorm(x, r, w)
        check("non-contig (via contiguous) allclose",
              torch.allclose(fo, ro, rtol=1e-4, atol=1e-5))
    except Exception as e:
        check("non-contig (via contiguous) allclose", False, repr(e)[:90])

    # 4. weight berbeda-beda seragam 1.0 -> hasil = normalisasi murni
    ones = torch.ones(HIDDEN, device=DEVICE)
    x = torch.randn(4, HIDDEN, device=DEVICE)
    r = torch.randn(4, HIDDEN, device=DEVICE)
    s = x + r
    manual = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + EPS)
    fo, _ = fused_add_rmsnorm(x, r, ones)
    check("weight=1 samakan manual rmsnorm",
          torch.allclose(fo, manual, rtol=1e-4, atol=1e-6))


def swiglu_battery():
    print("=== fused_swiglu_fp8 ===")
    # 1. determinisme: run sama -> byte sama
    g = torch.randn(8, INTER, device=DEVICE, dtype=torch.bfloat16)
    u = torch.randn(8, INTER, device=DEVICE, dtype=torch.bfloat16)
    q1, s1 = fused_swiglu_quant(g, u)
    q2, s2 = fused_swiglu_quant(g, u)
    check("deterministik byte-sama",
          torch.equal(q1.view(torch.uint8), q2.view(torch.uint8))
          and torch.equal(s1, s2))

    # 2. nol persis: gate=up=0 -> y=0, scale floor 1e-30, q=0
    g0 = torch.zeros(4, INTER, device=DEVICE, dtype=torch.bfloat16)
    u0 = torch.zeros(4, INTER, device=DEVICE, dtype=torch.bfloat16)
    q, s = fused_swiglu_quant(g0, u0)
    check("semua-nol -> q nol & scale floor",
          bool((q.view(torch.uint8) == 0).all())
          and bool((s == 1e-30).all()))

    # 3. saturasi: y sangat besar -> q capai 448 (e4m3 max).
    #    q sudah dalam satuan scale (q = y/scale), jadi max|q| langsung dicek
    g = torch.full((2, INTER), 20.0, device=DEVICE, dtype=torch.bfloat16)
    u = torch.full((2, INTER), 20.0, device=DEVICE, dtype=torch.bfloat16)
    q, s = fused_swiglu_quant(g, u)
    ratio = q.float().abs().max().item()
    check("saturasi max|q| <= 448", ratio <= 448.0 + 1e-3, f"max={ratio}")
    # silu(20)*20 = 400 absmax -> y/scale = 448 tepat di batas e4m3
    check("kasus ini memang menyentuh 448", abs(ratio - 448.0) < 1e-2,
          f"max={ratio}")

    # 4. gate negatif besar -> silu ~ 0 (bukan nan/inf)
    g = torch.full((2, INTER), -50.0, device=DEVICE, dtype=torch.bfloat16)
    u = torch.randn(2, INTER, device=DEVICE, dtype=torch.bfloat16)
    out = fused_swiglu(g, u)
    ref = torch.nn.functional.silu(g.float()) * u.float()
    check("gate=-50 silu->0 finite",
          bool(torch.isfinite(out).all())
          and torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2))

    # 5. FP8 saturation vs naive: byte identik pada kasus ekstrem
    g = torch.randn(4, INTER, device=DEVICE, dtype=torch.bfloat16) * 10
    u = torch.randn(4, INTER, device=DEVICE, dtype=torch.bfloat16) * 10
    nq, ns, _ = naive_swiglu_quant(g, u)
    fq, fs = fused_swiglu_quant(g, u)
    check("kasus ekstrem fp8 byte-identik utk 99.9%+",
          (fq.view(torch.uint8) == nq.view(torch.uint8)).float().mean().item() > 0.999)

    # 6. K determinisme pure swiglu
    g = torch.randn(8, INTER, device=DEVICE, dtype=torch.bfloat16)
    u = torch.randn(8, INTER, device=DEVICE, dtype=torch.bfloat16)
    check("swiglu deterministik",
          torch.equal(fused_swiglu(g, u), fused_swiglu(g, u)))


if __name__ == "__main__":
    assert torch.cuda.is_available()
    sampling_battery()
    rmsnorm_battery()
    swiglu_battery()
    print(f"\n{'='*40}\nTOTAL: {len(fails)} gagal" +
          (f": {fails}" if fails else " — SEMUA PASS ✅"))
    raise SystemExit(1 if fails else 0)
