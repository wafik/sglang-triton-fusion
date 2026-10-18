#!/bin/bash
# §23: Qwen3.5-9B-GPTQ-INT4 (mssfj, gptq_v2, 7.69GB) + quant-agnostic stack
# (triton attn + CUDA graph + NGRAM s7/d8). §13 FP8 patch NOT applicable
# (GPTQ uses marlin kernels — gate won't fire). Hybrid: mamba bf16,
# prefill graph OFF (hang). ~4.5GB freed vs FP8 → more mamba slots.
export SGLANG_ENABLE_JIT_DEEPGEMM=false
rm -f /root/sglang.log /root/sglang.pid
cd /root
nohup /root/venv_netra/bin/python -u -m sglang.launch_server \
  --model-path /root/models/Qwen3.5-9B-GPTQ-INT4 \
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
