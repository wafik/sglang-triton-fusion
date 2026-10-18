"""8-way parallel — prove max_running=8 mamba slots under load."""
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
        with ThreadPoolExecutor(8) as ex:
            durs = list(ex.map(one, range(8)))
        wall = time.perf_counter() - t0
        agg = 8 * 256 / wall
        ts.append(agg)
        print(f"8-way wall {wall:.1f}s  aggregate {agg:.1f}  "
              f"per-req {[round(d, 1) for d in durs]}")
    print(f"median 8-way: {statistics.median(ts):.1f} tok/s")
