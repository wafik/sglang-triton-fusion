"""4 request paralel — kalau aggregate >> single (27.5), decode weight-bandwidth-bound."""
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = "http://localhost:30000/generate"
PROMPT = ("Jelaskan cara kerja transformator dalam bahasa Indonesia, "
          "minimal seratus kata.")


def one(_):
    t0 = time.perf_counter()
    requests.post(URL, json={"text": PROMPT,
                             "sampling_params": {"max_new_tokens": 256,
                                                 "temperature": 0.7}},
                  timeout=300).raise_for_status()
    return time.perf_counter() - t0


if __name__ == "__main__":
    ts = []
    for _ in range(2):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(4) as ex:
            durs = list(ex.map(one, range(4)))
        wall = time.perf_counter() - t0
        agg = 4 * 256 / wall
        ts.append(agg)
        print(f"wall {wall:.1f}s  aggregate {agg:.1f} tok/s  "
              f"per-req {[round(d, 1) for d in durs]}")
    print(f"median aggregate: {statistics.median(ts):.1f} tok/s (vs 27.5 single)")
