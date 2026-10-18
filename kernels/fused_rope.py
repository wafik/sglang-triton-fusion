"""Fused RoPE (NeoX rotate_half) vs HF-style eager multi-kernel.

Naive HF eager: cat(cos,cos) + cat(sin,sin) + neg + cat rotate_half + mul + mul
+ add ≈ 7 launch + 3 intermediate.
Fused: 1 launch per token — load x[D], cos/sin[D/2]:
  o1 = x1*cos - x2*sin  (posisi di separuh pertama)
  o2 = x2*cos + x1*sin  (posisi di separuh kedua)
  out = where(offs < half, o1, o2)
Check: (a) allclose vs naive, (b) norm-invariance (sifat RoPE), (c) pos-0 identitas.
Qwen3-4B: head_dim=128, 32 heads (diverifikasi dari config.json).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, seq_len, stride_x, stride_c,
                D: tl.constexpr):
    tok = tl.program_id(0)
    s = tok % seq_len
    half = D // 2
    offs = tl.arange(0, D)
    x = tl.load(x_ptr + tok * stride_x + offs).to(tl.float32)
    # mirror = nilai separuh berlawanan di koordinat ini: i^half (bit-flip 64)
    #   i<64 -> i+64, i>=64 -> i-64 (half constexpr, power-of-2)
    xm = tl.load(x_ptr + tok * stride_x + (offs ^ half)).to(tl.float32)
    co = tl.load(cos_ptr + s * stride_c + (offs % half)).to(tl.float32)
    si = tl.load(sin_ptr + s * stride_c + (offs % half)).to(tl.float32)
    # NeoX: i<64  -> x[i]*cos - x[i+64]*sin
    #        i>=64 -> x[i]*cos + x[i-64]*sin
    sgn = tl.where(offs < half, -1.0, 1.0)
    out = x * co + sgn * xm * si
    tl.store(out_ptr + tok * stride_x + offs, out.to(tl.bfloat16))


def fused_rope(x, cos, sin):
    """x: (B, H, S, D) bf16; cos/sin: (S, D/2) fp32 — keduanya kontigu."""
    b, h, s, d = x.shape
    n = b * h * s
    xf = x.reshape(n, d)  # view kontigu
    out = torch.empty_like(xf)
    rope_kernel[(n,)](xf, cos, sin, out, s, xf.stride(0), cos.stride(0), D=d)
    return out.view(b, h, s, d)


def naive_rope(x, cos, sin):
    """HF apply_rotary_pos_emb (NeoX) eager — banyak launch, banyak intermediate."""
    cos_f = torch.cat([cos, cos], dim=-1)          # (S, D)
    sin_f = torch.cat([sin, sin], dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)             # rotate_half
    return (x * cos_f + rot * sin_f).to(x.dtype)


def make_cache(seq_len, head_dim, device="cuda"):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2,
                                             device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    return freqs.cos(), freqs.sin()  # (S, D/2)


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
    D = 128  # Qwen3-4B head_dim

    for name, shape in [("decode  (16,32,1,128)", (16, 32, 1, D)),
                        ("prefill (4,32,1024,128)", (4, 32, 1024, D))]:
        b, h, s, d = shape
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        cos, sin = make_cache(s, D)

        ref = naive_rope(x, cos, sin)
        got = fused_rope(x, cos, sin)
        # Triton emit FMA (1 rounding) utk x*cos+sgn*xm*sin; torch mul+mul+add
        # (3 rounding) -> parent fp32 beda ~1e-9 di elemen boundary, lolos jadi
        # flip 1 ulp bf16. Skala toleransi = bf16 (eps 2^-8 ~ 4e-3), bukan fp32.
        exact = (got.view(torch.uint8) == ref.view(torch.uint8)).float().mean().item()
        assert torch.allclose(got.float(), ref.float(), rtol=8e-3, atol=8e-3), \
            f"{name}: mismatch {(got.float() - ref.float()).abs().max()}"

        # sifat RoPE: norm per token invarian
        nx = x.float().norm(dim=-1)
        ng = got.float().norm(dim=-1)
        assert torch.allclose(ng, nx, rtol=2e-2), \
            f"{name}: norm tak invarian {(ng - nx).abs().max()}"

        # pos-0: cos=1 sin=0 -> identitas
        assert torch.allclose(got[:, :, 0].float(), x[:, :, 0].float(),
                              rtol=1e-6, atol=1e-6), f"{name}: pos-0 bukan identitas"

        t_na = bench(lambda: naive_rope(x, cos, sin))
        t_fu = bench(lambda: fused_rope(x, cos, sin))
        print(f"{name}: akurasi PASS ✅ (bitwise {exact * 100:.2f}%) | "
              f"naive {t_na:.2f} µs | fused {t_fu:.2f} µs | {t_na / t_fu:.2f}x")
