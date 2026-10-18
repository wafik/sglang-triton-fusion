'''Prefill 8B: prompt unik (uuid-hex) → tak kena radix cache, t(max_new=1) sekali jalan.
Slope = (t_long - t_short)/(tok_long - tok_short). Batas: max_total 18815 token.'''
import time, requests, uuid, statistics
URL = 'http://localhost:30000/generate'

def cold(n_chars):
    p = uuid.uuid4().hex * (n_chars // 32 + 1)
    t0 = time.perf_counter()
    r = requests.post(URL, json={'text': p, 'sampling_params':
                      {'max_new_tokens': 1, 'temperature': 0.0}}, timeout=300)
    dt = time.perf_counter() - t0
    m = r.json().get('meta_info', {})
    return dt, m.get('prompt_tokens', 0), m.get('cached_tokens', 0)

if __name__ == '__main__':
    S, L = 400, 16000
    rs = [cold(S) for _ in range(3)]
    rl = [cold(L) for _ in range(3)]
    tu_s = statistics.median(pt - ct for _, pt, ct in rs)
    tu_l = statistics.median(pt - ct for _, pt, ct in rl)
    t_s = statistics.median(d for d, _, _ in rs)
    t_l = statistics.median(d for d, _, _ in rl)
    dtok = (t_l - t_s) / (tu_l - tu_s)
    print(f'uncached: {tu_s} vs {tu_l} tok | {t_s*1000:.0f} vs {t_l*1000:.0f} ms')
    print(f'prefill: {dtok*1000:.3f} ms/token -> {1/dtok:.0f} tok/s')
