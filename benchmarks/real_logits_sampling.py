"""Validasi fused sampling di logit ASLI server (bukan randn sintetis).

Ambil top-50 logprob nyata dari /generate (return_logprob + top_logprobs_num),
bangun baris sparse (sisanya -inf), lalu:
  1) support: semua sampel ∈ nucleus set dari 50 token itu
  2) distribusi: TV distance vs torch.multinomial pada bobot yang sama
"""
import requests
import torch

from fused_sampling import fused_sample

URL = "http://localhost:30000/generate"


def fetch_top_logprobs():
    r = requests.post(URL, json={
        "text": "Sebutkan ibu kota Indonesia dan jelaskan singkat.",
        "sampling_params": {"max_new_tokens": 8, "temperature": 1.0},
        "return_logprob": True,
        "top_logprobs_num": 50,
        "logprob_start_len": 0,
    }, timeout=60)
    r.raise_for_status()
    meta = r.json()["meta_info"]
    # output_top_logprobs: list per langkah, tiap entri [logprob, token_id, text]
    step = meta["output_top_logprobs"][0]
    return [(float(t[0]), int(t[1])) for t in step]


def main():
    assert torch.cuda.is_available()
    tops = fetch_top_logprobs()
    print(f"top-50 nyata dari server: {len(tops)} entri, "
          f"logprob [{tops[-1][0]:.3f}, {tops[0][0]:.3f}]")

    V = 151936
    row = torch.full((1, V), float("-inf"), device="cuda")
    toks = [t for _, t in tops]
    lps = torch.tensor([lp for lp, _ in tops], device="cuda")
    row[0, torch.tensor(toks, device="cuda")] = lps  # log p (monoton dgn logit)

    # 1) support: K=50, top_p=0.9 — semua sampel di nucleus
    probs50 = torch.softmax(lps, -1)
    order = torch.argsort(lps, descending=True)
    cum = torch.cumsum(probs50[order], -1)
    keep = (cum - probs50[order]) <= 0.9
    nucleus = set(toks[i] for i in order[keep].tolist())
    for s in range(20):
        out = fused_sample(row, temperature=1.0, top_k=50, top_p=0.9, seed=s)
        assert int(out[0]) in nucleus, f"sampel {out[0]} di luar nucleus"
    print(f"Check 1: support di nucleus ({len(nucleus)} token) PASS ✅")

    # 2) distribusi: TV distance vs multinomial reference pada bobot sama
    N = 5000
    counts = torch.zeros(V, dtype=torch.int64)
    for b in range(10):
        # per-batch (500 baris), bukan N sekaligus — 5000×151936 fp32 = 3GB
        rowN = row.repeat(500, 1)
        s = fused_sample(rowN, temperature=1.0, top_k=50, top_p=0.9, seed=1000 + b)
        counts += torch.bincount(s.cpu(), minlength=V)
        del rowN
    freq = counts.double() / N

    # reference: multinomial dari distribusi nucleus-renorm yang sama
    ref_p = torch.zeros(V, device="cuda")
    kept = order[keep]            # index ke-50, BUKAN id token
    toks_t = torch.tensor(toks, device="cuda")
    ref_p[toks_t[kept]] = probs50[kept]
    ref_p /= ref_p.sum()
    gen = torch.Generator(device="cuda").manual_seed(0)
    ref_counts = torch.bincount(
        torch.multinomial(ref_p, N, replacement=True, generator=gen).cpu(),
        minlength=V).double() / N
    tv = (freq - ref_p.cpu()).abs().sum().item() / 2
    tv_stat = (freq - ref_counts).abs().sum().item() / 2
    print(f"Check 2: TV(fused, teori) = {tv:.4f} | TV(fused, multinomial) = "
          f"{tv_stat:.4f} (toleransi 0.05)")
    assert tv < 0.05, "distribusi fused meleset dari teori!"
    print("Distribusi PASS ✅")


if __name__ == "__main__":
    main()
