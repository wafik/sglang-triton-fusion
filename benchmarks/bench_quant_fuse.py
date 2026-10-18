"""Koreksi budget quant: sglang SUDAH mem-fuse qkv & gate_up.

qwen3.py: QKVParallelLinear (1 quant utk q+k+v), gate_up_proj (1 quant utk
gate+up). Jadi 4 quant call/layer (qkv, o, gate_up, down), bukan 6.
Microbench: overhead quant per call di shape nyata layer → total ms/step.
"""
import torch

from sglang.srt.layers.quantization.fp8_utils import (
    sglang_per_token_group_quant_fp8_row_padded,
)

G = 128


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
    return a.elapsed_time(b) / iters * 1e3


if __name__ == "__main__":
    torch.manual_seed(0)
    print("quant act per call (shape nyata layer Qwen3-4B):")
    total_us = 0.0
    for name, (b, s, h) in [("qkv   (hidden=2560)", (1, 1, 2560)),
                            ("o     (attn_out=4096)", (1, 1, 4096)),
                            ("gate_up (hidden=2560)", (1, 1, 2560)),
                            ("down  (ffn_in=9728)", (1, 1, 9728)),
                            ("qkv   bs=16", (16, 1, 2560))]:
        x = torch.randn(b * s, h, device="cuda", dtype=torch.bfloat16)
        t = bench(lambda: sglang_per_token_group_quant_fp8_row_padded(x, G))
        if "bs=16" not in name:
            total_us += t
        print(f"  {name:<24} {t:6.1f} µs")
    step = total_us * 36
    print(f"4 call/layer × 36 layer = {step / 1000:.2f} ms/step "
          f"({step / 36000 * 100:.1f}% dari 36ms decode)")
    print("catatan: sglang sudah fused qkv & gate_up — sisa 4 call/layer, "
          "masing-masing input TIDAK sama (hidden vs attn_out vs ffn_in) "
          "→ tak bisa digabung lebih jauh tanpa ubah kernel GEMM.")
