#!/bin/bash
# §20: ngram w8 + corpus wikipedia id eksternal
export SGLANG_ENABLE_JIT_DEEPGEMM=false
rm -f /root/sglang.log /root/sglang.pid
cd /root
nohup /root/venv_netra/bin/python -u -m sglang.launch_server \
  --model-path /root/models/Qwen3-8B-FP8 \
  --host 0.0.0.0 --port 30000 \
  --attention-backend triton \
  --disable-custom-all-reduce \
  --speculative-algorithm NGRAM \
  --speculative-num-steps 7 \
  --speculative-num-draft-tokens 8 \
  --speculative-ngram-external-corpus-path /root/corpus_id.jsonl \
  --speculative-ngram-external-sam-budget 7 \
  > /root/sglang.log 2>&1 &
echo $! > /root/sglang.pid
