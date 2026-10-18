"""Jalan nyata server: fp8_blockwise_scaled_mm (W8A8), bukan dequant-naive.

Bandwidth efektif decode = weight fp8 dibaca per step. Microbench naive chain
(13 GB/s) membuktikan dequant-materialize TIDAK dipakai server (jelas lebih
lambat dari 125GB/s). Ini: benchmark path asli sglang + quant aktivasinya.
Layout dari checkpoint: weight [N, K], weight_scale_inv [N/128, K/128] (2D
blockwise), scales_b kernel = [K/128, N/128] = Ws.T.
"""
import torch

from sglang.kernels.ops.gemm.fp8_blockwise_gemm import fp8_blockwise_scaled_mm
from sglang.srt.layers.quantization.fp8_utils import (
    sglang_per_token_group_quant_fp8_row_padded,
)

GB = 1024 ** 3
G = 128


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
    return a.elapsed_time(b) / iters * 1e3  # µs


def run(name, n_out, k):
    dev = "cuda"
    Wq = torch.randn(n_out, k, device=dev).to(torch.float8_e4m3fn)
    Ws = (torch.rand(n_out // G, k // G, device=dev) + 0.5).float()
    x = torch.randn(1, k, device=dev)

    try:
        q_in, x_scale = sglang_per_token_group_quant_fp8_row_padded(x, G)

        def gemv_fp8():
            return fp8_blockwise_scaled_mm(q_in, Wq.T, x_scale, Ws.T,
                                           out_dtype=torch.bfloat16)

        t_q = bench(lambda: sglang_per_token_group_quant_fp8_row_padded(x, G))
        t_fp8 = bench(gemv_fp8)
    except Exception as e:
        print(f"{name:<10} {n_out:>6}x{k:<6} | fp8 path gagal: "
              f"{type(e).__name__}: {str(e)[:160]}")
        return

    # bf16 referensi: dequant sekali (di luar timing), GEMV murni
    Wb = (Wq.float() * Ws.repeat_interleave(G, 0).repeat_interleave(G, 1)
          ).bfloat16().T.contiguous().T  # [N, K] bf16
    t_bf16 = bench(lambda: x.bfloat16() @ Wb.T)

    fp8_bytes = n_out * k * 1.03
    bf16_bytes = n_out * k * 2
    print(f"{name:<10} {n_out:>6}x{k:<6} | quant {t_q:6.1f} | "
          f"fp8 gemv {t_fp8:7.1f} µs ({fp8_bytes / (t_fp8 * 1e-6) / GB:5.0f} GB/s) | "
          f"bf16 gemv {t_bf16:7.1f} µs ({bf16_bytes / (t_bf16 * 1e-6) / GB:5.0f} GB/s) | "
          f"fp8/bf16 {t_bf16 / t_fp8:.2f}x")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print("path asli server: per-token-group-128 quant + fp8_blockwise_scaled_mm")
    for nm, n, k in [("q_proj", 4096, 2560), ("o_proj", 2560, 4096),
                     ("gate_up", 9728, 2560), ("down", 2560, 9728),
                     ("lm_head", 151936, 2560)]:
        try:
            run(nm, n, k)
        except torch.OutOfMemoryError:
            print(f"{nm:<10} {n:>6}x{k:<6} | OOM-skip")
