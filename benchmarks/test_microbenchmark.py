"""Tahap 1: Synthetic microbenchmark for baseline PyTorch argmax (PLAN.md)."""
import time

import torch

print("--- MEMULAI SINTETIS MICROBENCHMARK (RTX 5060 Ti) ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat: {device} ({torch.cuda.get_device_name(0)})")

# ponytail: shape hardcoded (batch 16, vocab Qwen 151936) — tanpa argparse/CLI karena ini skrip sekali-jalan.
logits = torch.randn(16, 151936, dtype=torch.float32, device=device)

# 1. Baseline: Standard PyTorch Argmax
torch.cuda.synchronize()
start_time = time.time()
for _ in range(100):
    baseline_result = torch.argmax(logits, dim=-1)
torch.cuda.synchronize()
baseline_time = (time.time() - start_time) / 100 * 1e6  # dalam mikrodetik (µs)

print(f"[Baseline PyTorch Argmax] Waktu rata-rata: {baseline_time:.2f} µs")
print("Status Tahap 1 (Synthetic Test): BERHASIL ✅")
