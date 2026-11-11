# NInfer-L3 Deployment Guide

This document is written for autonomous agents (or operators) who need to deploy
NInfer from source to a working inference server. Follow the steps in order.
Every command has been verified on a clean RTX 5090 + WSL2 + Docker Desktop setup.

## Target architecture

```
┌─────────────────────────────────────────────────────┐
│  Windows 11 (32 GB RAM)                             │
│  ┌───────────────────────────────────────────────┐  │
│  │  WSL2 VM (24 GB, .wslconfig memory=24GB)      │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │  Docker container: ninfer               │  │  │
│  │  │  ├─ ninfer-serve (C++/CUDA engine)     │  │  │
│  │  │  ├─ L3 disk cache (88 GiB on volume)   │  │  │
│  │  │  └─ Port 8080 → host 9999              │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
│  GPU 0: RTX 3090 (ComfyUI rendering)               │
│  GPU 1: RTX 5090 (ninfer inference, exclusive)     │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

| Component | Minimum | Notes |
|---|---|---|
| GPU | RTX 5090 (sm_120a) | Exclusive access required for GPU 1 |
| RAM | 32 GB | 24 GB for WSL VM + 8 GB for Windows/ComfyUI |
| CUDA | 13.1+ | NVIDIA driver 610.88+ |
| Docker Desktop | Latest | WSL2 backend, NVIDIA Container Toolkit |
| Disk (NVMe) | ≥ 200 GB free | System + model + L3 cache + pagefile |
| Disk (SATA SSD) | ≥ 250 GB free | ComfyUI + pagefile (optional, for I/O isolation) |

## Step 1: Configure WSL memory

Edit `%USERPROFILE%\.wslconfig` (create if missing):

```ini
[wsl2]
memory=24GB
processors=12
swap=6GB

[experimental]
autoMemoryReclaim=gradual
```

Apply: `wsl --shutdown` then restart Docker Desktop.

> **Why 24 GB?** The engine needs ~11 GB RSS (weights mmap 12.4 GB + KV pool 6 GB
> pinned + scheduler heap). 16 GB causes page-fault thrashing (majflt > 100K/min).
> Do NOT set below 24 GB if ComfyUI renders concurrently on the same machine.

## Step 2: Configure pagefile (recommended)

Move the Windows pagefile to a dedicated SSD (ideally D: SATA, separate from the
NVMe that hosts the Docker data disk):

1. System Properties → Advanced → Performance → Settings → Advanced → Virtual Memory
2. Uncheck "Automatically manage paging file size"
3. Set your NVMe drive (C:) to "No paging file"
4. Set the SATA SSD (D:) to "System managed size"
5. Restart Windows

This isolates paging I/O from the engine's L3 cache I/O on the NVMe.

## Step 3: Clone and build

```bash
git clone https://github.com/TIGillusion/ninfer-L3.git
cd ninfer-L3

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build -j
```

Dependencies (Debian/Ubuntu):
```bash
apt-get install --yes --no-install-recommends \
  cmake ninja-build pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev
```

CUDA: NVIDIA CUDA 13.1 with `sm_120a` target. Install `cuda-nvcc-13-1` and
`cuda-cudart-dev-13-1` from NVIDIA's apt repo:
```bash
# Inside a gcc:13 container or bare metal:
apt-get install cuda-nvcc-13-1 cuda-cudart-dev-13-1
```

## Step 4: Download model artifact

```bash
pip install huggingface_hub
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer --local-dir ./models
```

Alternative models (all compatible, see README for the full table):
```bash
hf download neroued/Qwen3.6-27B-nvfp4-NInfer \
  qwen3_6_27b_nvfp4.ninfer --local-dir ./models
```

## Step 5: Prepare L3 cache directory

```bash
mkdir -p /path/to/l3cache/l3
# This directory stores the rolling KV page chain and state images.
# Budget: 88 GiB (grows on demand). Ensure the filesystem has ≥ 100 GB free.
```

## Step 6: Start the server

### Docker (recommended for production)

```bash
docker build --tag ninfer:latest .

docker run -d \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume ./models:/models:ro \
  --volume /path/to/l3cache:/prod-kvcache \
  --ipc=host \
  --ulimit memlock=-1 \
  --restart unless-stopped \
  --name ninfer \
  ninfer:latest \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
    --max-context 400000 \
    --kv-capacity 628000 \
    --max-concurrency 5 \
    --kv-dtype nvfp4 \
    --device-state-slots 2 \
    --host-state-slots 16 \
    --host-kv-mib 6144 \
    --spec mtp --draft-tokens 3 \
    --lm-head-draft \
    --vision \
    --disk-kv-path /prod-kvcache/l3 \
    --disk-kv-gib 88 \
    --disk-kv-restore \
    --admission-order prefill-first \
    --prefill-yield-max-ms 5000 \
    --pending-timeout-ms 6000 \
    --checkpoint-watermark-pct 80 \
    --min-persist-tokens 8192 \
    --max-private-continuations 32 \
    --max-shared-prefixes 16 \
    --max-long-anchors-per-continuation 8 \
    --media-cache-mib 256 \
    --preserve-thinking
```

### Bare metal (no Docker)

```bash
./build/apps/ninfer-serve ./models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 400000 \
  --kv-capacity 628000 \
  --max-concurrency 5 \
  --kv-dtype nvfp4 \
  --device-state-slots 2 \
  --host-state-slots 16 \
  --host-kv-mib 6144 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --vision \
  --disk-kv-path ./l3cache/l3 \
  --disk-kv-gib 88 \
  --disk-kv-restore \
  --admission-order prefill-first \
  --prefill-yield-max-ms 5000 \
  --pending-timeout-ms 6000 \
  --checkpoint-watermark-pct 80 \
  --min-persist-tokens 8192 \
  --preserve-thinking
```

## Step 7: Verify

```bash
# Health
curl http://127.0.0.1:8080/health
# → {"status":"ok"}

# OpenAI chat
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'

# Research: perplexity scoring
curl http://127.0.0.1:8080/v1/score \
  -H 'Content-Type: application/json' \
  -d '{"text": "The quick brown fox jumps over the lazy dog."}'

# Research: next-token distribution
curl http://127.0.0.1:8080/v1/next-token \
  -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is", "top_k": 5}'
```

## Research API reference

### POST /v1/score

Teacher-forced causal scoring. Returns per-token logprobs and perplexity.

| Field | Type | Description |
|---|---|---|
| `text` | string | Text to score (≥ 2 tokens, ≤ max_context - 1) |
| `model` | string | Model alias (optional) |

Response fields: `tokens[]`, `token_logprobs[]` (null at position 0),
`aggregate.scored_tokens`, `aggregate.perplexity`.

### POST /v1/next-token

Next-token probability distribution (full-vocab softmax, no sampling).

| Field | Type | Description |
|---|---|---|
| `messages` | array | Chat messages (mutually exclusive with `text`) |
| `text` | string | Raw text (mutually exclusive with `messages`) |
| `top_k` | int | Top-K range [1,100], default 10 |

Response fields: `top[]` (token_id, token_text, probability, logprob),
`chosen` (the actually sampled token).

### OpenAI chat logprobs

Chat Completions with `top_logprobs: N` renders the first-token distribution
as `choices[0].logprobs.content[0]` in the OpenAI format.

## Production notes

### Memory budget (24 GB VM)

| Consumer | Size | Pinned |
|---|---|---|
| Engine RSS (weights mmap + KV pool + heap) | ~11 GB | Anon heap locked via mlock-anon |
| Host KV pool (pinned) | 6 GB | cudaHostAlloc |
| Host state pool | 2.3 GB | cudaHostAlloc |
| dockerd + containerd + snowluma | ~2 GB | |
| Page cache (reclaimable) | remainder | |

### L3 disk cache lifecycle

1. **First turn**: full prefill, pages written to L3 as they are computed
2. **Subsequent turns**: only new-token pages written (delta append)
3. **Preopener anchor**: state image at prompt-end (146 MB per session)
4. **Probe anchor**: state image at admission-depth
5. **Rolling eviction**: oldest pages evicted at 88 GiB budget
6. **Restore**: disk reads (5 workers, ~440 MB/s) → pipelined H2D (transfer stream) → prefill

A 170K-token session restores in **1–5 seconds** vs 60+ seconds re-prefill.

### Research scoring internals

The `/v1/score` endpoint:
1. Tokenizes raw text (no chat template)
2. Runs a full prefill with `prompt_logprobs=true`
3. Per-chunk: copies per-position hidden → lm_head projection → `target_logprobs` CUDA kernel → pinned D2H
4. Returns per-token logprobs aligned with the tokenized sequence (position 0 = null)
5. Forces `allow_prefix_reuse=false` for deterministic full-prompt scoring

The `/v1/next-token` endpoint:
1. Runs prefill (chat template or raw text)
2. Captures the last-position logit row (483 KB BF16)
3. CPU softmax over the full 248,077-token vocabulary
4. Returns top-K with probability, logprob, and the chosen token

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Health 000, engine process alive | Docker data-plane death | `wsl --shutdown`, wait 30s, Docker Desktop restarts |
| Bootstrap fails with `getty: console=hvc0` | docker-desktop distro corrupted | `wsl --unregister docker-desktop`, restart Docker Desktop |
| majflt > 100K/min, VmRSS < 1 GB | VM RAM exhaustion → guest swap thrash | Increase VM memory or reduce concurrent load |
| exit 70 / exit 75 / exit 137 | Stall sentinel or OOM kill | Check dmesg for OOM, increase VM memory |
| Decode drops during ComfyUI render | GPU-PV driver contention | Normal under WSL2; native Windows eliminates this |
| CUDA 999 / libcuda abort in cuStreamSynchronize | GPU-PV driver bug | Update NVIDIA driver; known WSL2 limitation |
