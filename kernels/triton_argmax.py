"""Tahap 4: Triton argmax — 2 varian vs torch.argmax (PLAN.md).

Catatan: baseline steady-state (dengan warmup + cuda event) ~18µs, bukan
345µs — angka PLAN.md itu cold-start (lazy cub init + time.time tanpa warmup).
Tensor 9.7MB muat di L2, jadi steady-state memang jauh lebih cepat.
"""
import torch
import triton
import triton.language as tl

SHAPE = (16, 151936)  # ponytail: hardcoded, skrip sekali-jalan


# ---------- Varian A: two-stage (optimized) ----------
@triton.jit
def stage1(x_ptr, pval_ptr, pidx_ptr, n_cols, stride_x, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    bid = tl.program_id(1)
    start = bid * BLOCK
    cols = start + tl.arange(0, BLOCK)
    mask = cols < n_cols
    vals = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=float("-inf"))
    m = tl.max(vals, axis=0)
    i = start + tl.argmax(vals, axis=0)
    pid = row * tl.num_programs(1) + bid
    tl.store(pval_ptr + pid, m)
    tl.store(pidx_ptr + pid, i)


@triton.jit
def stage2(pval_ptr, pidx_ptr, out_ptr, n_blocks, BLOCK: tl.constexpr):
    # load hanya partial values untuk cari pos, idx di-load 1x di pos itu
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_blocks
    vals = tl.load(pval_ptr + row * n_blocks + offs, mask=mask, other=float("-inf"))
    pos = tl.argmax(vals, axis=0)
    final = tl.load(pidx_ptr + row * n_blocks + pos)
    tl.store(out_ptr + row, final)


def two_stage(x):
    n_rows, n_cols = x.shape
    BLOCK = 8192
    n_blocks = triton.cdiv(n_cols, BLOCK)
    pval = torch.empty(n_rows * n_blocks, dtype=torch.float32, device=x.device)
    pidx = torch.empty(n_rows * n_blocks, dtype=torch.int64, device=x.device)
    out = torch.empty(n_rows, dtype=torch.int64, device=x.device)
    stage1[(n_rows, n_blocks)](x, pval, pidx, n_cols, x.stride(0), BLOCK=BLOCK)
    stage2[(n_rows,)](pval.view(n_rows, n_blocks), pidx.view(n_rows, n_blocks),
                      out, n_blocks, BLOCK=triton.next_power_of_2(n_blocks))
    return out


# ---------- Varian B: single-launch, loop per row ----------
@triton.jit
def argmax_row(x_ptr, out_ptr, n_cols, stride_x, CHUNK: tl.constexpr):
    """1 program per row, loop chunk -> 1 launch. Tie: first-wins (== torch)."""
    row = tl.program_id(0)
    acc_v = float("-inf")
    acc_i = 0
    for start in range(0, n_cols, CHUNK):
        cols = start + tl.arange(0, CHUNK)
        mask = cols < n_cols
        v = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=float("-inf"))
        m = tl.max(v, axis=0)
        i = start + tl.argmax(v, axis=0)
        better = m > acc_v  # strict >: tie pertama menang, sama seperti torch
        acc_v = tl.where(better, m, acc_v)
        acc_i = tl.where(better, i, acc_i)
    tl.store(out_ptr + row, acc_i)


def single_launch(x):
    n_rows, n_cols = x.shape
    out = torch.empty(n_rows, dtype=torch.int64, device=x.device)
    argmax_row[(n_rows,)](x, out, n_cols, x.stride(0), CHUNK=8192)
    return out


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # µs


if __name__ == "__main__":
    assert torch.cuda.is_available(), "butuh GPU"
    torch.manual_seed(42)
    logits = torch.randn(SHAPE, dtype=torch.float32, device="cuda")

    ref = torch.argmax(logits, dim=-1)
    for name, fn in [("two_stage", two_stage), ("single_launch", single_launch)]:
        got = fn(logits)
        assert torch.equal(got, ref), f"{name} MISMATCH! {(got != ref).sum()} beda"
    print(f"Akurasi: dua varian IDENTIK dengan torch.argmax")

    # uji kasus variabel
    for name, t in [
        ("scale-1e6", torch.randn(SHAPE, device="cuda") * 1e6),
        ("tie-dominan", torch.zeros(SHAPE, device="cuda")),
        ("ekstrem", torch.full(SHAPE, -1e30, device="cuda")),
    ]:
        r = torch.argmax(t, dim=-1)
        for fn in (two_stage, single_launch):
            assert torch.equal(fn(t), r), f"{name}: MISMATCH"
        if name == "tie-dominan":
            t[7, 151000] = 1.0  # sebagian besar nol -> banyak tie
            r = torch.argmax(t, dim=-1)
            for fn in (two_stage, single_launch):
                assert torch.equal(fn(t), r), f"{name}-2: MISMATCH"
    print("Checks: seed-42 + scale + tie + ekstrem PASS ✅")

    base = bench(lambda: torch.argmax(logits, dim=-1))
    a = bench(lambda: two_stage(logits))
    b = bench(lambda: single_launch(logits))
    print(f"[Baseline torch.argmax]      {base:.2f} µs")
    print(f"[Triton two-stage]           {a:.2f} µs  ({base / a:.2f}x)")
    print(f"[Triton single-launch]       {b:.2f} µs  ({base / b:.2f}x)")
    best = min(base, a, b)
    print(f"Terbaik: {best:.2f} µs" + (" (Triton MENANG 🚀)" if best < base else " (torch masih menang)"))
