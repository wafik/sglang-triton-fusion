"""Fused sampling: topk(logits) + 1 kernel kecil (exp+cumsum+topp+renorm+sample).

vs baseline PyTorch yang materialisasi full-vocab softmax (16 x 151936 x 4B = 9.7MB).
Key: softmax monotone -> urutan topk(logits) == topk(probs), jadi pass softmax
penuh bisa DILEWATI; bobot filtered dihitung langsung dari k logits teratas.
"""
import torch
import triton
import triton.language as tl

B, V = 16, 151936
K, TOP_P, TEMPERATURE = 50, 0.9, 0.7


@triton.jit
def sample_kernel(topk_logits_ptr, topk_idx_ptr, out_ptr, K, temperature, top_p,
                  seed, stride, BLOCK: tl.constexpr):
    """1 program per baris: inverse-CDF sample dari distribusi top-k/top-p."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    vals = tl.load(topk_logits_ptr + row * stride + offs, mask=mask, other=float("-inf"))
    v0 = tl.load(topk_logits_ptr + row * stride)  # max (topk sorted desc)

    # bobot filtered = softmax unnormalized di subset (konstanta e^-max cancel saat renorm)
    w = tl.where(mask, tl.exp((vals - v0) / temperature), 0.0)

    # top-p (konvensi HF): keep_i iff cumsum SEBELUM i <= top_p; selalu keep pertama.
    # w tak ternormalisasi -> bandingkan dgn top_p * total, bukan top_p mentah.
    cum = tl.cumsum(w)
    total = tl.sum(w)
    keep = (cum - w) <= top_p * total
    w2 = tl.where(keep, w, 0.0)
    total = tl.sum(w2)

    # u ~ U(0, total); elemen pertama dari rand dipakai per baris
    r = tl.rand(seed, row + offs)
    u = tl.sum(tl.where(offs == 0, r, 0.0)) * total
    cum2 = tl.cumsum(w2)
    # first index where cum2 >= u (argmax pada 0/1 -> posisi True pertama)
    pos = tl.argmax((cum2 >= u).to(tl.int32), axis=0)
    tok = tl.load(topk_idx_ptr + row * stride + pos)
    tl.store(out_ptr + row, tok)


def fused_sample(logits, temperature=TEMPERATURE, top_k=K, top_p=TOP_P, seed=0):
    n = logits.shape[0]
    tk_v, tk_i = torch.topk(logits, top_k)  # sorted desc — urutan sama dgn topk(softmax(x))
    out = torch.empty(n, dtype=torch.int64, device=logits.device)
    sample_kernel[(n,)](tk_v, tk_i, out, top_k, temperature, top_p, seed,
                        tk_v.stride(0), BLOCK=triton.next_power_of_2(top_k))
    return out


# ---------------- Baseline PyTorch (bukan strawman) ----------------
def naive_full(logits, generator):
    """Unfused: full softmax + sort full-vocab + mask + renorm + multinomial."""
    probs = torch.softmax(logits / TEMPERATURE, dim=-1)
    sp, si = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sp, dim=-1)
    keep = ((cum - sp) <= TOP_P) & (torch.arange(V, device=logits.device) < K)
    f = torch.where(keep, sp, 0.0)
    f = f / f.sum(-1, keepdim=True).clamp_min(1e-20)
    return torch.gather(si, -1, torch.multinomial(f, 1, generator=generator)).squeeze(-1)


def naive_competent(logits, generator):
    """Fair baseline: full softmax -> topk -> topp pada subset -> renorm -> multinomial."""
    probs = torch.softmax(logits / TEMPERATURE, dim=-1)
    tp, ti = torch.topk(probs, K)
    cum = torch.cumsum(tp, dim=-1)
    keep = (cum - tp) <= TOP_P
    f = torch.where(keep, tp, 0.0)
    f = f / f.sum(-1, keepdim=True).clamp_min(1e-20)
    return torch.gather(ti, -1, torch.multinomial(f, 1, generator=generator)).squeeze(-1)


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # µs


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    logits = torch.randn(B, V, device="cuda") * 3.0

    # 1) Deterministik: top_k=1 -> selalu argmax (dua sample pertama tak acak)
    assert torch.equal(fused_sample(logits, top_k=1, top_p=1.0, seed=1),
                       torch.argmax(logits, dim=-1)), "top_k=1 bukan argmax"
    print("Check 1: top_k=1 == argmax PASS ✅")

    # 2) Keanggotaan: top_k=50 -> hasil selalu di set topk; top_p=0.9 -> di nucleus set
    tk = torch.topk(logits, K).indices
    for seed in range(50):
        s = fused_sample(logits, seed=seed)
        for i in range(B):
            assert s[i] in tk[i], f"token {s[i]} di luar top-{K}"
    # nucleus set (HF convention)
    probs = torch.softmax(logits / TEMPERATURE, -1)
    tp, ti = torch.topk(probs, K)
    cum = torch.cumsum(tp, -1)
    keep = (cum - tp) <= TOP_P
    nucleus = [set(ti[i][keep[i]].tolist()) for i in range(B)]
    for seed in range(50):
        s = fused_sample(logits, seed=seed)
        for i in range(B):
            assert int(s[i]) in nucleus[i], f"token {s[i]} di luar nucleus"
    print("Check 2: keanggotaan top-k & nucleus (50 seed x 16 baris) PASS ✅")

    # 3) Distribusi statistik: V=128, top_p=1, 10000 sampel -> freq ≈ softmax
    v2 = 128
    lg = torch.randn(500, v2, device="cuda") * 2.0  # 500 baris, seed+row memberi seed berbeda
    exact = torch.softmax(lg / TEMPERATURE, -1).mean(0).cpu()  # baris identik distribusi? TIDAK — beda logits
    # pakai 1 baris diulang: distribusi target = softmax baris itu
    row0 = lg[:1].repeat(500, 1)
    target = torch.softmax(row0[0] / TEMPERATURE, -1).cpu()
    counts = torch.zeros(v2, dtype=torch.int64)
    n_samples = 0
    for batch in range(10):  # 10 x 500 = 5000 sampel
        s = fused_sample(row0, top_k=v2, top_p=1.0, seed=1000 + batch)
        counts += torch.bincount(s.cpu(), minlength=v2)
        n_samples += 500
    freq = counts.double() / n_samples
    max_err = (freq - target).abs().max().item()
    print(f"Check 3: distribusi V=128, max|freq-p| = {max_err:.4f} (toleransi 0.03)")
    assert max_err < 0.03, "distribusi meleset!"
    print("Distribusi PASS ✅\n")

    gen = torch.Generator(device="cuda").manual_seed(7)
    g2 = torch.Generator(device="cuda").manual_seed(7)
    base_full = bench(lambda: naive_full(logits, gen))
    base_comp = bench(lambda: naive_competent(logits, g2))
    fused_t = bench(lambda: fused_sample(logits, seed=1))
    print(f"[Baseline naive full-vocab]  {base_full:.2f} µs")
    print(f"[Baseline naive kompeten]    {base_comp:.2f} µs")
    print(f"[Fused sampling]             {fused_t:.2f} µs")
    print(f"Speedup vs naive full: {base_full / fused_t:.2f}x | vs kompeten: {base_comp / fused_t:.2f}x")
