"""Bedah 125GB/s efektif decode: apakah fp8 weight-only dequant atau GEMV-nya?

Simulasi path weight-only: W fp8 + block-scale 128 -> dequant bf16 -> GEMV.
Bandingkan: (a) dequant saja, (b) GEMV bf16 saja, (c) rantai dequant+GEMV,
(d) torch._scaled_mm fp8 langsung (kalau tersedia di sm_120).
Shape khas layer Qwen3-4B: gate 2560x9728, down 9728x2560, lm_head 2560x151936.
"""
import torch

GB = 1024 ** 3


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


def run(name, n, m, group=128):
    device = "cuda"
    Wq = torch.randn(n, m, device=device).to(torch.float8_e4m3fn)
    # scale blockwise per 128 kolom
    ns = m // group
    scale = (torch.rand(n, ns, device=device) + 0.5).float()
    x = torch.randn(1, n, device=device)

    t_deq = bench(lambda: Wq.float() * scale.repeat_interleave(group, -1))
    Wb = (Wq.float() * scale.repeat_interleave(group, -1)).bfloat16()
    t_gemv = bench(lambda: x.bfloat16() @ Wb)  # (1,n)@(n,m)
    t_chain = bench(lambda: (x.bfloat16() @ (Wq.float()
                    * scale.repeat_interleave(group, -1)).bfloat16()))
    w_bytes = n * m * (1 + 4 / group)  # fp8 + scale
    gbps_chain = w_bytes / (t_chain * 1e-6) / GB

    t_mm = None
    try:
        xs = x.float()  # scaled_mm: fp8 both, per-row scale
        sx = torch.ones(1, device=device)
        t_mm = bench(lambda: torch._scaled_mm(xs.t().contiguous().to(torch.float8_e4m3fn),
                                              Wq.t().contiguous(),
                                              scale_a=sx, scale_b=scale.t().contiguous(),
                                              out_dtype=torch.bfloat16))
    except Exception as e:
        t_mm = f"n/a ({type(e).__name__})"

    print(f"{name:<16} {n:>5}x{m:<6} | dequant {t_deq:6.1f} µs | "
          f"gemv bf16 {t_gemv:7.1f} µs | chain {t_chain:7.1f} µs "
          f"({gbps_chain:5.0f} GB/s) | scaled_mm {t_mm}")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print("fp8 weight-only per-layer (1 token, batch=1):")
    for name, n, m in [("q_proj", 2560, 4096), ("gate_up", 2560, 9728),
                       ("down", 9728, 2560), ("lm_head", 2560, 151936)]:
        try:
            run(name, n, m)
        except torch.OutOfMemoryError:
            # ponytail: VRAM sisa ~3GB (server jalan); lm_head fp32 temp >4GB
            print(f"{name:<16} {n:>5}x{m:<6} | OOM-skip (temp fp32 > sisa VRAM)")
