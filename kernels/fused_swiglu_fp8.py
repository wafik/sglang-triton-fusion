"""Fused SwiGLU (+ opsi FP8 E4M3 per-group-128 quant) vs naive PyTorch.

Kontrak MLP Qwen3-4B: intermediate_size=9728, 36 layer, dipanggil tiap layer.
Varian:
  A) fused_swiglu        : silu(gate)*up di register, tulis BF16 — baseline
                           naive = silu lalu mul (2 launch + intermediate).
  B) fused_swiglu_quant  : + absmax/448 per grup-128, tulis FP8 + scale —
                           baseline naive = materialize y FP32 lalu quant.
Grid quant: (B, 9728/128=76), tiap program 1 grup — absmax lokal, tanpa
cross-program reduce.
"""
import torch
import triton
import triton.language as tl

INTER, GROUP, QMAX = 9728, 128, 448.0  # Qwen3-4B intermediate_size; e4m3fn max


@triton.jit
def swiglu_kernel(gate_ptr, up_ptr, out_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    g = tl.load(gate_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
    y = g * tl.sigmoid(g) * u  # silu(g)*u
    tl.store(out_ptr + row * stride + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def swiglu_quant_kernel(gate_ptr, up_ptr, q_ptr, s_ptr, n_cols, n_groups,
                        stride, qmax, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    grp = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    cols = grp * BLOCK + offs
    mask = cols < n_cols
    g = tl.load(gate_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    y = g * tl.sigmoid(g) * u
    am = tl.max(tl.abs(y), axis=0)
    scale = tl.maximum(am / qmax, 1e-30)
    q = tl.clamp(y / scale, -qmax, qmax)
    tl.store(q_ptr + row * stride + cols, q.to(tl.float8e4nv), mask=mask)
    tl.store(s_ptr + row * n_groups + grp, scale)


def fused_swiglu(gate, up):
    n, h = gate.shape
    out = torch.empty_like(gate)
    swiglu_kernel[(n,)](gate, up, out, h, gate.stride(0),
                        BLOCK=triton.next_power_of_2(h))
    return out


def fused_swiglu_quant(gate, up):
    n, h = gate.shape
    ng = h // GROUP
    q = torch.empty_like(gate, dtype=torch.float8_e4m3fn)
    scale = torch.empty((n, ng), dtype=torch.float32, device=gate.device)
    swiglu_quant_kernel[(n, ng)](gate, up, q, scale, h, ng,
                                 gate.stride(0), QMAX, BLOCK=GROUP)
    return q, scale


def naive_swiglu(gate, up):
    return torch.nn.functional.silu(gate) * up


def naive_swiglu_quant(gate, up):
    y = torch.nn.functional.silu(gate.float()) * up.float()  # materialize FP32
    am = y.abs().view(y.shape[0], -1, GROUP).amax(-1)
    scale = (am / QMAX).clamp_min(1e-30)
    q = (y.view(y.shape[0], -1, GROUP) / scale.unsqueeze(-1)).clamp(-QMAX, QMAX)
    q = q.view(y.shape[0], -1).to(torch.float8_e4m3fn)  # bentuk sama dgn fused
    return q, scale, y


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

    for name, shape in [("decode  (16, 9728)", (16, INTER)),
                        ("prefill (512, 9728)", (512, INTER))]:
        gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        up = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

        # A: SwiGLU murni
        ref = torch.nn.functional.silu(gate.float()) * up.float()
        got = fused_swiglu(gate, up)
        assert torch.allclose(got.float(), ref, rtol=2e-2, atol=2e-2), \
            f"{name}: swiglu mismatch {(got.float() - ref).abs().max()}"
        t_na = bench(lambda: naive_swiglu(gate, up))
        t_fu = bench(lambda: fused_swiglu(gate, up))
        print(f"{name} SwiGLU     : naive {t_na:.2f} µs | fused {t_fu:.2f} µs "
              f"| {t_na / t_fu:.2f}x ✅")

        # B: SwiGLU + FP8 quant per grup 128
        nq, ns, y32 = naive_swiglu_quant(gate, up)
        fq, fs = fused_swiglu_quant(gate, up)
        assert torch.allclose(fs, ns, rtol=1e-5, atol=1e-8), \
            f"{name}: scale mismatch {(fs - ns).abs().max():.2e}"
        deq = fq.view(y32.shape[0], -1, GROUP).float() * fs.unsqueeze(-1)
        # E4M3 = grid float (mantissa 3 bit) -> setengah ulp <= |y|/16, BUKAN
        # scale/2 (bound int8). + slack 1e-3*scale buat noise silu fp32.
        yv = y32.view(y32.shape[0], -1, GROUP)
        err = (deq - yv).abs()
        limit = yv.abs() * 0.0625 + fs.unsqueeze(-1) * 1e-3
        assert (err <= limit).all(), \
            f"{name}: quant error lebih dari setengah ulp e4m3 ({(err > limit).sum()} elemen)"
        match = (fq.view(torch.uint8) == nq.view(torch.uint8)).float().mean().item()
        t_na = bench(lambda: naive_swiglu_quant(gate, up))
        t_fu = bench(lambda: fused_swiglu_quant(gate, up))
        print(f"{name} SwiGLU+FP8 : naive {t_na:.2f} µs | fused {t_fu:.2f} µs "
              f"| {t_na / t_fu:.2f}x | fp8 identik {match * 100:.2f}% ✅\n")
