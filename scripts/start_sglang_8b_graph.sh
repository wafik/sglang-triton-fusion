#!/bin/bash
# §16: CUDA graph decode ON (tanpa --disable-cuda-graph), patch §13 aktif.
export SGLANG_ENABLE_JIT_DEEPGEMM=false
rm -f /root/sglang.log /root/sglang.pid
cd /root
nohup /root/venv_netra/bin/python -u -m sglang.launch_server \
  --model-path /root/models/Qwen3-8B-FP8 \
  --host 0.0.0.0 --port 30000 \
  --attention-backend triton \
  --disable-custom-all-reduce \
  > /root/sglang.log 2>&1 &
echo $! > /root/sglang.pid
echo "started pid $(cat /root/sglang.pid)"
