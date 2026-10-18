Here is the **detailed work plan (Action Plan)** for starting kernel optimization experiments using **WSL2 (Ubuntu on Windows)** with the lightest model first (`Qwen3-4B-FP8`), so SSD space stays safe and you don't have to bother with *dual-boot* right away.

---

## 🛠️ Environment Preparation Stage in WSL2 (Ubuntu)

Open your **WSL2 (Ubuntu)** terminal, then run the following steps in order:

### 1. Update the System & Make Sure the NVIDIA Driver Is Detected

WSL2 automatically *passes through* the NVIDIA driver from the main Windows installation. Verify that your GPU is read correctly:

```bash
sudo apt update && sudo apt upgrade -y
nvidia-smi

```

*Make sure `RTX 5060 Ti` appears in the table below along with its CUDA version (minimum CUDA 12.8).*

### 2. Create a Dedicated Virtual Environment for Experiments

Use a clean Python 3.10/3.11 so the PyTorch and SGLang installations don't conflict with anything else on the system:

```bash
sudo apt install -y python3-pip python3-venv git curl
python3 -m venv venv_netra
source venv_netra/bin/activate
pip install --upgrade pip

```

### 3. Install PyTorch (Latest CUDA Version) & SGLang

Install PyTorch with support for the Blackwell architecture (`sm_120`), then install the SGLang framework:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "sglang[all]" triton

```

---

## 🚀 Stage 1: Synthetic Microbenchmark (0 GB Model)

Before downloading a physical model, we first test the *baseline* vs *custom kernel* speed using random tensors in VRAM (for example, simulating *Two-Stage Argmax* or *Fused SiLU* operations).

Create a new file named `test_microbenchmark.py`:

```bash
nano test_microbenchmark.py

```

Fill it with the following simple test script (copy and paste):

```python
import torch
import time

print("--- MEMULAI SINTETIS MICROBENCHMARK (RTX 5060 Ti) ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat: {device} ({torch.cuda.get_device_name(0)})")

# Simulasi bentuk tensor logika Qwen (Batch 16, Vocabulary 151936)
logits = torch.randn(16, 151936, dtype=torch.float32, device=device)

# 1. Baseline: Standard PyTorch Argmax
torch.cuda.synchronize()
start_time = time.time()
for _ in range(100):
    baseline_result = torch.argmax(logits, dim=-1)
torch.cuda.synchronize()
baseline_time = (time.time() - start_time) / 100 * 1e6 # dalam mikrodetik (µs)

print(f"[Baseline PyTorch Argmax] Waktu rata-rata: {baseline_time:.2f} µs")
print("Status Tahap 1 (Synthetic Test): BERHASIL ✅")

```

Save the file (`Ctrl+O`, `Enter`, then `Ctrl+X`), then run it:

```bash
python3 test_microbenchmark.py

```

---

## 📥 Stage 2: Preparing the Light Model (`Qwen/Qwen3-4B-FP8`)

Once the framework is tested, download the first model whose size is SSD-friendly (~4.2 GB):

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-4B-FP8

```

---

## 🏃 Stage 3: Running the SGLang Server with the 4B Model

Run the SGLang server in WSL2 to verify that the server responds correctly:

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B-FP8 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --port 30000

```

Open a new WSL2 terminal tab (leave the server running in the first tab), then test sending a text *request*:

```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen3-4B-FP8",
    "prompt": "Halo, jelaskan apa itu optimasi kernel GPU dalam satu kalimat.",
    "max_tokens": 64
  }'

```

---

### Next Steps

Please copy or run the **Stage 1** steps above first in your WSL2. Let me know if everything runs smoothly or if you hit any dependency *error* issues, and then we can immediately design the first Triton *custom kernel* to cut down that argmax/fused-op execution time!
