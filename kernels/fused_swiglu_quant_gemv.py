"""Fused SwiGLU + per-group quant + down-GEMV dalam 1 kernel Triton.

Rantai naive (path sglang per layer MLP):
  silu(gate)*up  → materialize act bf16 (launch + tulis 2*M*9728)
  quant act      → launch terpisah ~40µs
  down GEMV      → launch terpisah (cutlass)
= 3 launch + 2 intermediate. Fused: 1 launch — silu di register, quant di
register, akumulasi GEMV langsung ke weight blockwise 128x128.

Layout gate_up output: [gate(I) | up(I)] concat — sama persis dgn
sglang silu_and_mul (d = len//2, silu(x[:d]) * x[d:]).
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.fp8_blockwise_gemm import fp8_blockwise_scaled_mm
from sglang.srt.layers.quantization.fp8_utils import (
    sglang_per_token_group_quant_fp8_row_padded,
)

GB = 1024 ** 3


@triton.jit
def swiglu_q_gemm_kernel(gu_ptr, w_ptr, ws_ptr, out_ptr, I, N, n_blks,
                         stride_wn, stride_ws,
                         BLOCK_N: tl.constexpr, BLOCK_I: tl.constexpr,
                         qmax: tl.constexpr, eps):
    # grid = (M * n_blks,); I, N kelipatan 128
    pid = tl.program_id(0)
    n0 = (pid % n_blks) * BLOCK_N
    row = pid // n_blks
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_i = tl.arange(0, BLOCK_I)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, I, BLOCK_I):
        # gate & up: dua segmen berdampingan di dim 1 (concat [gate|up])
        g = tl.load(gu_ptr + row * 2 * I + k0 + offs_i).to(tl.float32)
        u = tl.load(gu_ptr + row * 2 * I + I + k0 + offs_i).to(tl.float32)
        act = g * tl.sigmoid(g) * u          # silu(gate) * up, fp32 penuh
        # quant mirror sglang: max(amax,eps)/qmax → kali inverse → clamp
        absmax = tl.maximum(tl.max(tl.abs(act), axis=0), eps)
        sx = absmax / qmax
        xq = tl.clamp(act * (1.0 / sx), -qmax, qmax).to(tl.float8e4nv).to(tl.float32)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn
                    + (k0 + offs_i)[None, :]).to(tl.float32)
        sw = tl.load(ws_ptr + (n0 // 128) * stride_ws + (k0 // 128))
        acc += tl.sum(xq[None, :] * w, axis=1) * (sx * sw)
    tl.store(out_ptr + row * N + offs_n, acc.to(out_ptr.dtype.element_ty))


def fused_swiglu_q_gemm(gu, Wd, Ws):
    """gu: (M, 2I) bf16 gate_up output; Wd: (N, I) fp8; Ws: (N/128, I/128)
    → (M, N) bf16."""
    m, two_i = gu.shape
    i = two_i // 2
    n = Wd.shape[0]
    assert two_i % 2 == 0 and i % 128 == 0 and n % 128 == 0
    n_blks = n // 128
    out = torch.empty(m, n, device=gu.device, dtype=torch.bfloat16)
    swiglu_q_gemm_kernel[(m * n_blks,)](gu, Wd, Ws, out, i, n, n_blks,
                                        Wd.stride(0), Ws.stride(0),
                                        BLOCK_N=128, BLOCK_I=128,
                                        qmax=448.0, eps=1e-10)
    return out


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1e3


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    I, N = 9728, 2560  # Qwen3-4B down proj
    print("fused SwiGLU+quant+down-GEMV 1 launch vs rantai 3 langkah:")
    for m in (1, 16):
        Wd = torch.randn(N, I, device="cuda").to(torch.float8_e4m3fn)
        Ws = (torch.rand(N // 128, I // 128, device="cuda") + 0.5).float()
        gu = torch.randn(m, 2 * I, device="cuda", dtype=torch.bfloat16)

        # referensi rantai: silu_and_mul torch → quant sglang → cutlass
        g, u = gu.chunk(2, dim=-1)
        act = (F.silu(g.float()) * u.float()).to(torch.bfloat16)
        qi, xs = sglang_per_token_group_quant_fp8_row_padded(act, 128)
        ref = fp8_blockwise_scaled_mm(qi, Wd.T, xs, Ws.T,
                                      out_dtype=torch.bfloat16)[:m]
        got = fused_swiglu_q_gemm(gu, Wd, Ws)

        rf = ref.float()
        rmax = rf.abs().max().item()
        ulp = 2.0 ** (torch.floor(torch.log2(torch.tensor(rmax))).item() - 7)
        atol = 2.0 * ulp + 1e-3
        diff = (got.float() - rf).abs()
        assert torch.allclose(got.float(), rf, rtol=0, atol=atol), \
            f"m={m}: maxabs {diff.max().item():.4f} (atol {atol:.4f})"

        def chain():
            g_, u_ = gu.chunk(2, dim=-1)
            a_ = (F.silu(g_.float()) * u_.float()).to(torch.bfloat16)
            qi_, xs_ = sglang_per_token_group_quant_fp8_row_padded(a_, 128)
            return fp8_blockwise_scaled_mm(qi_, Wd.T, xs_, Ws.T,
                                           out_dtype=torch.bfloat16)
        t_ch = bench(chain)
        t_fu = bench(lambda: fused_swiglu_q_gemm(gu, Wd, Ws))
        bw = m * (2 * I + I + N) * 1.0 / (t_fu * 1e-6) / GB
        print(f"m={m:<3} chain {t_ch:7.1f} µs | fused {t_fu:7.1f} µs | "
              f"{t_ch / t_fu:.2f}x | maxabs {diff.max().item():.3f} "
              f"({diff.max().item() / ulp:.2f} ulp) PASS ✅")
