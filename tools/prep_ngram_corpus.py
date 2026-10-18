"""§20: siapkan corpus ngram eksternal (wikipedia id) utk accept-rate.
JSONL: tiap baris = JSON string dokumen. ~1500 artikel pertama.
"""
import json
import sys

from datasets import load_dataset

OUT = "/root/corpus_id.jsonl"
N_DOCS = 1500

ds = load_dataset("wikimedia/wikipedia", "20231101.id", split="train", streaming=True)
n_tok_est = 0
with open(OUT, "w", encoding="utf-8") as f:
    for i, art in enumerate(ds):
        if i >= N_DOCS:
            break
        text = art["text"]
        if len(text) < 200:
            continue
        f.write(json.dumps(text, ensure_ascii=False) + "\n")
        n_tok_est += len(text) // 3  # kasar: ~3 char/token utk id/en campur
print(f"{i} artikel, ~{n_tok_est} token kasar -> {OUT}", file=sys.stderr)
