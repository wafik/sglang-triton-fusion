#!/bin/bash
# §21 candidate 9b: ngram window larger than w8.
# draft 16 → verify graph num_tokens_per_req=16, capture bs=[1..8]
# still small (8*16=128 rows); bandwidth-bound → wider verify ~free.
# Fallback if OOM at capture: draft 12.
export SGLANG_ENABLE_JIT_DEEPGEMM=false
rm -f /root/sglang.log /root/sglang.pid
cd /root
nohup /root/venv_netra/bin/python -u -m sglang.launch_server \
  --model-path /root/models/Qwen3-8B-FP8 \
  --host 0.0.0.0 --port 30000 \
  --attention-backend triton \
  --disable-custom-all-reduce \
  --speculative-algorithm NGRAM \
  --speculative-num-steps 15 \
  --speculative-num-draft-tokens 16 \
  > /root/sglang.log 2>&1 &
echo $! > /root/sglang.pid
