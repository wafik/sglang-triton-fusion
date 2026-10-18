"""Fused QK-norm + RoPE (+ GQA expand utk K) vs HF eager.

Kontrak Qwen3-4B (config.json): 32 q-head, 8 kv-head (GQA 4:1), D=128,
rms_norm_eps=1e-6. HF eager per langkah:
  q: rms_norm (materialize) → rope (≈7 launch)
  k: rms_norm → rope → repeat_interleave(4) (materialize 4x)
Fused: 1 launch untuk q (norm+rope), 1 launch utk k (norm+rope+expand) —
grid dihitung di ruang HEAD KELUARAN, kv_h = h // ratio dibaca on-the-fly.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def qk_norm_rope_kernel(x_ptr, w_ptr, cos_ptr, sin_ptr, out_ptr,
                        S, Hkv, ratio, D: tl.constexpr, eps):
    tok = tl.program_id(0)              # over B * H_out * S
    s = tok % S
    h_out = (tok // S) % (Hkv * ratio)
    b = tok // (S * Hkv * ratio)
    kv_h = h_out // ratio               # ratio=1 (q) → kv_h = h_out
    x_off = ((b * Hkv + kv_h) * S + s) * D
    half = D // 2
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + x_off + offs).to(tl.float32)
    xm = tl.load(x_ptr + x_off + (offs ^ half)).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)
    wm = tl.load(w_ptr + (offs ^ half)).to(tl.float32)
    # rmsnorm per-head (ms diakumulasi fp32, mirror pakai ms sama)
    ms = tl.sum(x * x, axis=0) / D
    inv = tl.rsqrt(ms + eps)
    xn = x * inv * w
    xmn = xm * inv * wm
    # RoPE NeoX — mirror sudah ternormalisasi (ms sama, w[i^half])
    co = tl.load(cos_ptr + s * half + (offs % half)).to(tl.float32)
    si = tl.load(sin_ptr + s * half + (offs % half)).to(tl.float32)
    sgn = tl.where(offs < half, -1.0, 1.0)
    out = xn * co + sgn * xmn * si
    tl.store(out_ptr + tok * D + offs, out.to(tl.bfloat16))


def fused_qk_rope(q, k, qn_w, kn_w, cos, sin, ratio=4):
    """q: (B,Hq,S,D); k: (B,Hkv,S,D); cos/sin: (S,D/2). Out k: (B,Hq,S,D)."""
    bq, hq, sq, d = q.shape
    _, hkv, _, _ = k.shape
    nq = bq * hq * sq
    nk = bq * hq * sq  # grid ruang output head untuk k juga
    out_q = torch.empty(nq, d, device=q.device, dtype=q.dtype)
    out_k = torch.empty(nk, d, device=k.device, dtype=k.dtype)
    qk_norm_rope_kernel[(nq,)](q, qn_w, cos, sin, out_q, sq, hq, 1, D=d, eps=1e-6)
    qk_norm_rope_kernel[(nk,)](k, kn_w, cos, sin, out_k, sq, hkv, ratio, D=d, eps=1e-6)
    return out_q.view(bq, hq, sq, d), out_k.view(bq, hq, sq, d)


def naive_rope(x, cos, sin):
    cos_f = torch.cat([cos, cos], dim=-1)
    sin_f = torch.cat([sin, sin], dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    return (x * cos_f + rot * sin_f).to(x.dtype)


def naive_qk_rope(q, k, qn_w, kn_w, cos, sin, ratio=4):
    d = q.shape[-1]
    q2 = F.rms_norm(q, (d,), qn_w)
    k2 = F.rms_norm(k, (d,), kn_w)
    qr = naive_rope(q2, cos, sin)
    kr = naive_rope(k2, cos, sin)
    ke = kr.repeat_interleave(ratio, dim=1)
    return qr, ke


def make_cache(seq_len, head_dim, theta=1e6, device="cuda"):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                                             device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    return freqs.cos(), freqs.sin()


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1e3  # µs


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    D, HQ, HKV, RATIO = 128, 32, 8, 4  # Qwen3-4B
    qn_w = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.1 + 1
    kn_w = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.1 + 1

    for name, (b, s) in [("decode  (16,1)", (16, 1)),
                         ("prefill (4,1024)", (4, 1024))]:
        q = torch.randn(b, HQ, s, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
        cos, sin = make_cache(s, D)

        ref_q, ref_k = naive_qk_rope(q, k, qn_w, kn_w, cos, sin, RATIO)
        got_q, got_k = fused_qk_rope(q, k, qn_w, kn_w, cos, sin, RATIO)

        # atol longgar: naive membulatkan norm ke bf16 SEBELUM rope (2 rounding),
        # kami fp32 penuh — elemen cancellation (out≈0 dari suku ~4) membawa
        # 1 ulp bf16 di 4.0 = 0.03125 sebagai error absolut. Sifat benign.
        assert torch.allclose(got_q.float(), ref_q.float(), rtol=8e-3, atol=5e-2), \
            f"{name} q: mismatch {(got_q.float() - ref_q.float()).abs().max()}"
        assert torch.allclose(got_k.float(), ref_k.float(), rtol=8e-3, atol=5e-2), \
            f"{name} k: mismatch {(got_k.float() - ref_k.float()).abs().max()}"

        # pos-0: rope identitas → hasil = rmsnorm murni
        ref0 = F.rms_norm(q[:, :, 0], (D,), qn_w)
        assert torch.allclose(got_q[:, :, 0].float(), ref0.float(),
                              rtol=8e-3, atol=8e-3), f"{name}: pos-0 q bukan rmsnorm"

        # GQA: 4 q-head kelompok = salinan identik (setelah FMA tolerance)
        for j in range(RATIO):
            assert torch.allclose(got_k[:, 0].float(), got_k[:, j].float(),
                                  rtol=1e-6, atol=1e-6), f"{name}: GQA head {j} beda"
            assert torch.allclose(ref_k[:, 0].float(), ref_k[:, j].float(),
                                  rtol=1e-6, atol=1e-6)

        # norm-invarian sebelum rope: ||out|| ≈ ||rmsnorm(x)|| per head
        n_ref = F.rms_norm(q, (D,), qn_w).float().norm(dim=-1)
        n_got = got_q.float().norm(dim=-1)
        assert torch.allclose(n_got, n_ref, rtol=2e-2), f"{name}: norm tak invarian"

        t_na = bench(lambda: naive_qk_rope(q, k, qn_w, kn_w, cos, sin, RATIO))
        t_fu = bench(lambda: fused_qk_rope(q, k, qn_w, kn_w, cos, sin, RATIO))
        print(f"{name}: PASS ✅ | naive {t_na:.2f} µs | fused {t_fu:.2f} µs | "
              f"{t_na / t_fu:.2f}x (q+k termasuk GQA expand)")
