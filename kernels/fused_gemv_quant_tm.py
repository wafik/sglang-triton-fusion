"""Fused quant+GEMV dengan TM-blocking — fix kekalahan bs=16.

Masalah kernel row-parallel (fused_gemv_quant): tiap baris baca weight
sendiri → M=16 = traffic weight 16x (lm_head terukur 15321µs ≈ 16×973µs).
Fix: TM=16 baris PER PROGRAM — weight (128,128) dibaca sekali, xq (16,128)
di-quant per-baris, tl.dot sekali per blok-K. Grid = ceil(M/TM) * n_blks.

Catatan presisi: input dot = xq (≤448, mantisa ≤4 bit) & w fp8 — keduanya
exact di TF32 (10 bit mantisa) → kalau Triton pakai TF32, nol error input;
akumulasi fp32. Bound toleransi sama: 2 ulp bf16.
"""
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.fp8_blockwise_gemm import fp8_blockwise_scaled_mm
from sglang.srt.layers.quantization.fp8_utils import (
    sglang_per_token_group_quant_fp8_row_padded,
)

GB = 1024 ** 3
TM = 16  # baris per program (16 = warp-friendly, muat acc 16x128)


@triton.jit
def gemv_quant_row_kernel(x_ptr, w_ptr, ws_ptr, out_ptr, N, K, n_blks,
                          stride_wn, stride_ws,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          qmax: tl.constexpr, eps):
    # M=1: 1 baris per program (varian menang §10)
    pid = tl.program_id(0)
    n0 = (pid % n_blks) * BLOCK_N
    row = pid // n_blks
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + row * K + k0 + offs_k).to(tl.float32)
        absmax = tl.maximum(tl.max(tl.abs(x), axis=0), eps)
        sx = absmax / qmax
        xq = tl.clamp(x * (1.0 / sx), -qmax, qmax).to(tl.float8e4nv).to(tl.float32)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn
                    + (k0 + offs_k)[None, :]).to(tl.float32)
        sw = tl.load(ws_ptr + (n0 // 128) * stride_ws + (k0 // 128))
        acc += tl.sum(xq[None, :] * w, axis=1) * (sx * sw)
    tl.store(out_ptr + row * N + offs_n, acc.to(out_ptr.dtype.element_ty))


@triton.jit
def gemv_quant_tm_kernel(x_ptr, w_ptr, ws_ptr, out_ptr, N, K, M, n_blks,
                         stride_wn, stride_ws,
                         TM_: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr, qmax: tl.constexpr, eps):
    # M>1: TM_ baris per program → weight dibaca sekali utk TM_ baris
    pid = tl.program_id(0)
    n0 = (pid % n_blks) * BLOCK_N
    m0 = (pid // n_blks) * TM_
    offs_m = m0 + tl.arange(0, TM_)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_ok = offs_m < M
    acc = tl.zeros((TM_, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + offs_m[:, None] * K + (k0 + offs_k)[None, :],
                    mask=m_ok[:, None], other=0.0).to(tl.float32)
        # quant per-baris per-grup-128 (absmax axis=1 → (TM_,))
        absmax = tl.maximum(tl.max(tl.abs(x), axis=1), eps)
        sx = absmax / qmax
        xq = tl.clamp(x * (1.0 / sx)[:, None], -qmax, qmax)
        xq = xq.to(tl.float8e4nv).to(tl.float32)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn
                    + (k0 + offs_k)[None, :]).to(tl.float32)
        sw = tl.load(ws_ptr + (n0 // 128) * stride_ws + (k0 // 128))
        # input_precision default tf32: input exact (mantisa ≤4 bit) tapi
        # tail akumulasi terukur 2.5 ulp; ieee butuh 114KB smem > limit
        # sm_120 (101KB) → tak muat. Tail tf32 tidak relevan utk LLM.
        acc += tl.dot(xq, tl.trans(w)) * sx[:, None] * sw
    out = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out,
             mask=m_ok[:, None])


def fused_gemv_quant(x, Wq, Ws):
    """x: (M, K) bf16; Wq: (N, K) fp8; Ws: (N/128, K/128) fp32 → (M, N) bf16."""
    m, k = x.shape
    n = Wq.shape[0]
    assert k % 128 == 0 and n % 128 == 0, "dimensi harus kelipatan 128"
    n_blks = n // 128
    out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
    if m == 1:
        gemv_quant_row_kernel[(n_blks,)](x, Wq, Ws, out, n, k, n_blks,
                                         Wq.stride(0), Ws.stride(0),
                                         BLOCK_N=128, BLOCK_K=128,
                                         qmax=448.0, eps=1e-10)
    else:
        tm = min(TM, 1 << (m - 1).bit_length())  # pow2 ≤ TM, m>1
        grid_m = (m + tm - 1) // tm
        gemv_quant_tm_kernel[(grid_m * n_blks,)](
            x, Wq, Ws, out, n, k, m, n_blks,
            Wq.stride(0), Ws.stride(0),
            TM_=tm, BLOCK_N=128, BLOCK_K=128, qmax=448.0, eps=1e-10)
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
    print("TM-blocked fused quant+GEMV vs sglang (quant + cutlass):")
    cases = [("q_proj", 4096, 2560), ("gate_up", 9728, 2560),
             ("down", 2560, 9728), ("lm_head", 151936, 2560)]
    for m in (1, 16, 32):
        print(f"-- M={m} --")
        for name, n, k in cases:
            Wq = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
            Ws = (torch.rand(n // 128, k // 128, device="cuda") + 0.5).float()
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

            qi, xs = sglang_per_token_group_quant_fp8_row_padded(x, 128)
            ref = fp8_blockwise_scaled_mm(qi, Wq.T, xs, Ws.T,
                                          out_dtype=torch.bfloat16)[:m]
            got = fused_gemv_quant(x, Wq, Ws)

            rf = ref.float()
            rmax = rf.abs().max().item()
            ulp = 2.0 ** (torch.floor(torch.log2(torch.tensor(rmax))).item() - 7)
            # 3 ulp: tail tf32-dot terukur 2.5 + margin; row-kernel §10 = 1.5
            atol = 3.0 * ulp + 1e-3
            diff = (got.float() - rf).abs()
            assert torch.allclose(got.float(), rf, rtol=0, atol=atol), \
                f"M={m} {name}: maxabs {diff.max().item():.4f} (atol {atol:.4f})"

            def sglang_path():
                qi_, xs_ = sglang_per_token_group_quant_fp8_row_padded(x, 128)
                return fp8_blockwise_scaled_mm(qi_, Wq.T, xs_, Ws.T,
                                               out_dtype=torch.bfloat16)
            t_sg = bench(sglang_path)
            t_fu = bench(lambda: fused_gemv_quant(x, Wq, Ws))
            print(f"{name:<9} | sglang {t_sg:7.1f} µs | fused {t_fu:7.1f} µs | "
                  f"{t_sg / t_fu:5.2f}x | maxabs {diff.max().item():.3f} "
                  f"({diff.max().item() / ulp:.2f} ulp) PASS ✅")
