#!/bin/bash
# §22: Qwen3.5-9B-FP8 (block 128x128, lovedheart) + full winning stack
# (triton attn + fused quant+GEMV patch §13 + CUDA graph + NGRAM s7/d8).
# Hybrid linear-attn (24 linear / 8 full): mamba state in bf16 (config says fp32),
# mem-fraction 0.90 → enough slots. Weights 12.19GB on 16GB.
export SGLANG_ENABLE_JIT_DEEPGEMM=false
rm -f /root/sglang.log /root/sglang.pid
cd /root
nohup /root/venv_netra/bin/python -u -m sglang.launch_server \
  --model-path /root/models/Qwen3.5-9B-FP8 \
  --host 0.0.0.0 --port 30000 \
  --attention-backend triton \
  --disable-custom-all-reduce \
  --mem-fraction-static 0.90 \
  --mamba-ssm-dtype bfloat16 \
  --disable-prefill-cuda-graph \
  --speculative-algorithm NGRAM \
  --speculative-num-steps 7 \
  --speculative-num-draft-tokens 8 \
  > /root/sglang.log 2>&1 &
echo $! > /root/sglang.pid
