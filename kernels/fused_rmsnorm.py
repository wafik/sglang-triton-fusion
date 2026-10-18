"""Fused add+RMSNorm vs naive (add materialize -> F.rms_norm).

Pola decode Qwen3: hidden, residual = fused_add_rmsnorm(x, residual, w)
  -> out = rmsnorm(x + residual) * w ; residual_new = x + residual
Naive = 2 kernel + 1 intermediate; fused = 1 kernel, intermediate hilang.
"""
import torch
import triton
import triton.language as tl

HIDDEN, EPS = 2560, 1e-6  # Qwen3-4B hidden_size, rms_eps


@triton.jit
def add_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr, n_cols,
                       eps, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0)
    r = tl.load(res_ptr + row * stride + offs, mask=mask, other=0.0)
    s = x + r
    tl.store(res_out_ptr + row * stride + offs, s, mask=mask)  # residual baru
    var = tl.sum(s * s, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0)
    tl.store(out_ptr + row * stride + offs, s * rstd * w, mask=mask)


def fused_add_rmsnorm(x, residual, weight, eps=EPS):
    n, h = x.shape
    out = torch.empty_like(x)
    res_out = torch.empty_like(x)
    add_rmsnorm_kernel[(n,)](x, residual, weight, out, res_out, h, eps,
                             x.stride(0), BLOCK=triton.next_power_of_2(h))
    return out, res_out


def naive_add_rmsnorm(x, residual, weight, eps=EPS):
    s = x + residual
    return torch.nn.functional.rms_norm(s, (s.shape[-1],), weight, eps), s


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # µs


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    w = torch.randn(HIDDEN, device="cuda") * 0.1

    for name, shape in [("decode  (16, 2560)", (16, HIDDEN)),
                        ("prefill (512, 2560)", (512, HIDDEN))]:
        x = torch.randn(shape, device="cuda")
        r = torch.randn(shape, device="cuda")
        ref_out, ref_res = naive_add_rmsnorm(x, r, w)
        got_out, got_res = fused_add_rmsnorm(x, r, w)
        assert torch.allclose(got_out, ref_out, rtol=1e-4, atol=1e-5), \
            f"{name}: out mismatch {(got_out - ref_out).abs().max()}"
        assert torch.allclose(got_res, ref_res, rtol=0, atol=0), f"{name}: res mismatch"
        print(f"{name}: akurasi PASS ✅ (max err "
              f"{(got_out - ref_out).abs().max():.2e})")

        t_naive = bench(lambda: naive_add_rmsnorm(x, r, w))
        t_fused = bench(lambda: fused_add_rmsnorm(x, r, w))
        print(f"{name}: naive {t_naive:.2f} µs | fused {t_fused:.2f} µs "
              f"| {t_naive / t_fused:.2f}x\n")
