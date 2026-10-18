"""Fused per-token-group-128 quant + fp8 GEMV dalam 1 kernel Triton.

Menyerang 5.83ms/step (16.2% decode) dari quant launch terpisah (~40µs/call).
Path sglang: row_padded quant (launch #1) → fp8_blockwise_scaled_mm (launch #2).
Fused: 1 launch — absmax per group-128 x, cast fp8 di register, akumulasi
fp32 dengan scale x * scale w (blockwise 128x128 dari checkpoint).
Referensi akurasi = output path sglang asli (quant sama + cutlass mm).
"""
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.fp8_blockwise_gemm import fp8_blockwise_scaled_mm
from sglang.srt.layers.quantization.fp8_utils import (
    sglang_per_token_group_quant_fp8_row_padded,
)

GB = 1024 ** 3


@triton.jit
def gemv_quant_kernel(x_ptr, w_ptr, ws_ptr, out_ptr, N, K, n_blks,
                      stride_wn, stride_ws,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      qmax: tl.constexpr, eps):
    # grid = (M * n_blks,); K & N kelipatan 128 (div asserted di wrapper)
    pid = tl.program_id(0)
    n0 = (pid % n_blks) * BLOCK_N
    row = pid // n_blks
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + row * K + k0 + offs_k).to(tl.float32)
        # quant per-token-group: MIRROR sglang fp8_kernel.py:170-173
        # byte-identik: max(amax,eps) dulu → scale → kali INVERSE (bukan bagi)
        absmax = tl.maximum(tl.max(tl.abs(x), axis=0), eps)
        sx = absmax / qmax
        sinv = 1.0 / sx
        xq = tl.clamp(x * sinv, -qmax, qmax).to(tl.float8e4nv).to(tl.float32)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn
                    + (k0 + offs_k)[None, :]).to(tl.float32)
        # scale w: satu skalar per tile 128x128 → (n0//128, k0//128)
        sw = tl.load(ws_ptr + (n0 // 128) * stride_ws + (k0 // 128))
        acc += tl.sum(xq[None, :] * w, axis=1) * (sx * sw)
    tl.store(out_ptr + row * N + offs_n, acc.to(out_ptr.dtype.element_ty))


def fused_gemv_quant(x, Wq, Ws):
    """x: (M, K) bf16; Wq: (N, K) fp8; Ws: (N/128, K/128) fp32 → (M, N) bf16."""
    m, k = x.shape
    n = Wq.shape[0]
    assert k % 128 == 0 and n % 128 == 0, "dimensi harus kelipatan 128"
    n_blks = n // 128
    out = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
    gemv_quant_kernel[(m * n_blks,)](x, Wq, Ws, out, n, k, n_blks,
                                     Wq.stride(0), Ws.stride(0),
                                     BLOCK_N=128, BLOCK_K=128, qmax=448.0,
                                     eps=1e-10)  # sama dgn sglang default
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
    print("fused quant+GEMV 1 kernel vs sglang (quant + cutlass terpisah):")
    cases = [("q_proj", 4096, 2560), ("o_proj", 2560, 4096),
             ("gate_up", 9728, 2560), ("down", 2560, 9728),
             ("lm_head", 151936, 2560)]
    # bs=16: decode batch — quant makin launch-bound-free (23.8µs), cek menang
    cases16 = [("q_proj", 4096, 2560), ("gate_up", 9728, 2560),
               ("down", 2560, 9728), ("lm_head", 151936, 2560)]
    for tag, batch, subset in [("bs=1", 1, cases), ("bs=16", 16, cases16)]:
        print(f"-- {tag} --")
        for name, n, k in subset:
            Wq = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
            Ws = (torch.rand(n // 128, k // 128, device="cuda") + 0.5).float()
            x = torch.randn(batch, k, device="cuda", dtype=torch.bfloat16)

            # referensi: path sglang (row_padded menambah baris → potong [:m])
            q_in, x_scale = sglang_per_token_group_quant_fp8_row_padded(x, 128)
            ref = fp8_blockwise_scaled_mm(q_in, Wq.T, x_scale, Ws.T,
                                          out_dtype=torch.bfloat16)[:batch]
            got = fused_gemv_quant(x, Wq, Ws)

            # toleransi jujur: saya vs exact 0.61 ulp, cutlass vs exact 0.50 ulp,
            # dua pembulatan bf16 independen → bound ~1.6 ulp. Pakai 2× ulp bf16
            # pada magnitudo maks (near-zero cancellation ikut tercakup atol).
            rf = ref.float()
            rmax = rf.abs().max().item()
            ulp = 2.0 ** (torch.floor(torch.log2(torch.tensor(rmax))).item() - 7)
            atol = 2.0 * ulp + 1e-3
            diff = (got.float() - rf).abs()
            rel = (diff / rf.abs().clamp_min(1e-3)).max().item()
            assert torch.allclose(got.float(), rf, rtol=0, atol=atol), \
                f"{name}: maxabs {diff.max().item():.4f} (atol {atol:.4f}) rel {rel:.4f}"

            # ukur terpisah agar adil (quant + mm dua launch)
            def sglang_path():
                qi, xs = sglang_per_token_group_quant_fp8_row_padded(x, 128)
                return fp8_blockwise_scaled_mm(qi, Wq.T, xs, Ws.T,
                                               out_dtype=torch.bfloat16)
            t_sg = bench(sglang_path)
            t_fu = bench(lambda: fused_gemv_quant(x, Wq, Ws))
            bw = batch * n * k * 1.03 / (t_fu * 1e-6) / GB
            print(f"{name:<10} {n:>6}x{k:<6} | sglang {t_sg:7.1f} µs | "
                  f"fused {t_fu:7.1f} µs ({bw:5.0f} GB/s) | {t_sg / t_fu:.2f}x | "
                  f"maxabs {diff.max().item():.3f} ({diff.max().item()/ulp:.2f} ulp) PASS ✅")
