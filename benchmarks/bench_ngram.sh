#!/bin/bash
# §18 runner: wait ready → smoke + e2e + parallel
for i in $(seq 1 60); do
  grep -q 'fired up and ready' /root/sglang.log 2>/dev/null && break
  sleep 2
done
if ! grep -q 'fired up and ready' /root/sglang.log; then
  echo "SERVER NOT READY"; grep -i -m5 'error\|traceback' /root/sglang.log; exit 1
fi
echo "=== ready; spec lines ==="
grep -i -m8 'spec\|ngram' /root/sglang.log
echo "=== smoke ==="
/root/venv_netra/bin/python /root/smoke30.py
echo "=== e2e ==="
/root/venv_netra/bin/python /root/bench_e2e.py
echo "=== parallel ==="
/root/venv_netra/bin/python /root/bench_parallel.py
