"""E2E decode benchmark server SGLang: tok/s terpisah prefill vs decode.

Metode: jalan dengan max_new_tokens=1 (≈prefill+overhead) dan 512;
(t512 - t1)/511 ≈ waktu per token decode. Ponytail: wall-clock client-side,
bukan server metric — cukup utk A/B kualitatif.
"""
import statistics
import time

import requests

URL = "http://localhost:30000/generate"
PROMPT = ("Jelaskan dengan bahasa Indonesia yang baik cara kerja model bahasa "
          "large language model dalam menghasilkan teks, mulai dari tokenisasi, "
          "embedding, attention, hingga decoding. Berikan contoh konkret di "
          "setiap tahapnya dan panjang minimal dua ratus kata agar detail.")
PARAMS = {"skip_special_tokens": True}


def run(max_new_tokens, n=3):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = requests.post(URL, json={
            "text": PROMPT,
            "sampling_params": {**PARAMS, "max_new_tokens": max_new_tokens,
                                "temperature": 0.7},
        }, timeout=300)
        r.raise_for_status()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


if __name__ == "__main__":
    t1 = run(1)
    t512 = run(512)
    dtok = (t512 - t1) / 511
    print(f"t(max=1)   : {t1 * 1000:.0f} ms  (prefill+overhead)")
    print(f"t(max=512) : {t512 * 1000:.0f} ms")
    print(f"decode     : {dtok * 1000:.2f} ms/token -> {1 / dtok:.1f} tok/s")
