Berikut adalah **rencana kerja detail (Action Plan)** untuk memulai eksperimen optimasi kernel menggunakan **WSL2 (Ubuntu di Windows)** dengan model yang paling ringan terlebih dahulu (`Qwen3-4B-FP8`), sehingga ruang SSD aman dan Anda tidak perlu langsung repot melakukan *dual-boot*.

---

## 🛠️ Tahap Persiapan Lingkungan di WSL2 (Ubuntu)

Buka terminal **WSL2 (Ubuntu)** Anda, lalu jalankan langkah-langkah berikut secara berurutan:

### 1. Update Sistem & Pastikan Driver NVIDIA Terdeteksi

WSL2 secara otomatis melakukan *passthrough* driver NVIDIA dari Windows utama. Verifikasi bahwa GPU Anda terbaca dengan benar:

```bash
sudo apt update && sudo apt upgrade -y
nvidia-smi

```

*Pastikan `RTX 5060 Ti` muncul di tabel bawah beserta versi CUDA-nya (minimal CUDA 12.8).*

### 2. Buat Virtual Environment Khusus Eksperimen

Gunakan Python 3.10/3.11 bersih agar instalasi PyTorch dan SGLang tidak berbenturan dengan sistem lain:

```bash
sudo apt install -y python3-pip python3-venv git curl
python3 -m venv venv_netra
source venv_netra/bin/activate
pip install --upgrade pip

```

### 3. Instal PyTorch (Versi CUDA Terbaru) & SGLang

Instal PyTorch yang mendukung arsitektur Blackwell (`sm_120`), lalu instal kerangka SGLang:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "sglang[all]" triton

```

---

## 🚀 Tahap 1: Microbenchmark Sintetis (0 GB Model)

Sebelum mengunduh model fisik, kita uji dulu kecepatan *baseline* vs *custom kernel* menggunakan tensor acak di VRAM (misalnya mensimulasikan operasi *Two-Stage Argmax* atau *Fused SiLU*).

Buat file baru bernama `test_microbenchmark.py`:

```bash
nano test_microbenchmark.py

```

Isi dengan skrip pengujian sederhana berikut (salin dan tempel):

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

Simpan file (`Ctrl+O`, `Enter`, lalu `Ctrl+X`), kemudian jalankan:

```bash
python3 test_microbenchmark.py

```

---

## 📥 Tahap 2: Menyiapkan Model Ringan (`Qwen/Qwen3-4B-FP8`)

Setelah kerangka kerja teruji, unduh model pertama yang ukurannya ramah SSD (~4.2 GB):

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-4B-FP8

```

---

## 🏃 Tahap 3: Menjalankan Server SGLang dengan Model 4B

Jalankan server SGLang di WSL2 untuk memverifikasi apakah server merespons dengan benar:

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B-FP8 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --port 30000

```

Buka tab terminal WSL2 baru (biarkan server tetap berjalan di tab pertama), lalu tes kirim *request* teks:

```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen3-4B-FP8",
    "prompt": "Halo, jelaskan apa itu optimasi kernel GPU dalam satu kalimat.",
    "max_tokens": 64
  }'

```

---

### Langkah Selanjutnya

Silakan salin atau jalankan langkah **Tahap 1** di atas terlebih dahulu di WSL2 Anda. Beritahu saya jika sudah berjalan mulus atau ada kendala *error* dependensi, lalu kita bisa langsung merancang *custom kernel* Triton pertamanya untuk memangkas waktu eksekusi argmax/fused-op tersebut!
