#!/bin/bash

poetry run python3 -m vllm.entrypoints.openai.api_server \
  --model models/HyperCLOVAX-SEED-Text-Instruct-1.5B \
  --trust-remote-code \
  --port 8001 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --dtype half \
  --enable-chunked-prefill \
  --prefix-caching-hash-algo sha256 \
  --disable-log-requests \
  --scheduling-policy fcfs