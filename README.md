# NInfer-L3

> From-scratch C++/CUDA inference engine. Production-hardened for 24/7 agent serving
> with L3 disk-cache reuse, research API, and autonomous recovery.

A production fork of [Neroued/ninfer](https://github.com/Neroued/ninfer) hardened for
non-stop AI-agent serving on a single RTX 5090. The upstream engine provides the
C++/CUDA inference core; this fork adds the L3 disk-cache tier, research API endpoints,
autonomous recovery, and operational fixes accumulated from weeks of 24/7
multi-agent production use.

## What this fork adds over upstream

| Feature | Description |
|---|---|
| **L3 disk-cache tier** | 88 GiB NVMe-backed rolling cache with per-turn checkpoints (preopener + probe anchors), exact-prefix page-chain reuse, and state-image restore. A 170K-token session restores in 1–5 s instead of 60+ s re-prefill. |
| **Research API** | `POST /v1/score` (teacher-forced per-token logprobs + perplexity) and `POST /v1/next-token` (exact full-vocab softmax next-token distribution). Purpose-built for offline evaluation and RLHF pipelines. |
| **Transfer-stream restore** | Seed-restore H2D runs on a dedicated CUDA transfer stream, fully overlapping decode on the compute stream. Materialization no longer suppresses decode throughput. |
| **mlock-anon** | Engine heap and stacks (1.17 GB) are `mlock`-ed after warmup, preventing host swap-out under memory pressure. Named mappings (weights mmap, driver shmem) are excluded. |
| **Queue timeout 6 s** | Queue wait returns 503 in 6 s (down from 30 s) for fast agent-side retry feedback. |
| **Dual materialization slots** | Two concurrent L3 restores, with a waiter-rescue mechanism for capacity-blocked heads. |
| **Prefill yield** | Prefill yields to decode within 5 s windows, keeping decode latency bounded under multi-agent load. |
| **Auto-recover** | Watchdog script (`auto_recover.bat`) monitors engine health and triggers `wsl --shutdown` + Docker Desktop restart on failure — no UAC needed. |
| **Non-blocking stderr** | Engine stderr is set to non-blocking at startup; a full log pipe never wedges business threads. |
| **Docker commit safety** | Container binary changes survive watchdog `rm+run` via `docker commit` into the `ninfer:l3-yarn31` tag. |

## Supported models

| Model | Weights | Artifact | Hugging Face |
|---|---|---|---|
| Qwen3.6-27B | groupwise-int | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | nvfp4 | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | groupwise-int | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | nvfp4 | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | groupwise-int | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Each artifact embeds the tokenizer, chat template, and media frontend resources.

## Requirements

- 64-bit Linux (native or WSL2)
- NVIDIA GeForce RTX 5090 (sm_120a)
- CUDA Toolkit 13.1+
- CMake 3.28+, C++20 compiler, Ninja
- FFmpeg dev libraries (`libavformat>=60`, `libavcodec>=60`, `libavutil>=58`, `libswscale>=7`)
- `libcurl>=7.85`
- 32 GB+ system RAM

## Quick start

### Build from source

```bash
git clone https://github.com/TIGillusion/ninfer-L3.git
cd ninfer-L3
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Download a model artifact

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer --local-dir models
```

### Start the server

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 400000 \
  --kv-capacity 628000 \
  --max-concurrency 5 \
  --kv-dtype nvfp4 \
  --device-state-slots 2 \
  --host-state-slots 16 \
  --host-kv-mib 6144 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --disk-kv-path /prod-kvcache/l3 \
  --disk-kv-gib 88 \
  --disk-kv-restore \
  --admission-order prefill-first \
  --prefill-yield-max-ms 5000 \
  --pending-timeout-ms 6000 \
  --checkpoint-watermark-pct 80 \
  --min-persist-tokens 8192 \
  --preserve-thinking
```

### Send an OpenAI-style request

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## Research API

Two endpoints expose the model's internal probability distributions for evaluation pipelines.

### POST /v1/score

Teacher-forced causal scoring: per-token logprobs and perplexity for any text.

```bash
curl http://127.0.0.1:9999/v1/score \
  -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is Paris."}'
```

Response:

```json
{
  "token_count": 10,
  "tokens": ["The", " quick", "..."],
  "token_logprobs": [null, -10.01, -3.49, "..."],
  "aggregate": {
    "scored_tokens": 9,
    "total_nll": 14.76,
    "mean_nll": 1.64,
    "perplexity": 5.15
  }
}
```

### POST /v1/next-token

Exact next-token probability distribution (full-vocab softmax, no sampling).

```bash
curl http://127.0.0.1:9999/v1/next-token \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "1+1="}], "top_k": 5}'
```

Response:

```json
{
  "prompt_token_count": 6,
  "top_k": 5,
  "top": [
    {"token_id": 73766, "token_text": "2", "probability": 0.83, "logprob": -0.19}
  ],
  "chosen": {"token_text": "2", "logprob": -0.19}
}
```

## Docker

Build the runtime image:

```bash
docker build --tag ninfer:latest .
```

Run with GPU passthrough and L3 cache volume:

```bash
docker run -d \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume /path/to/models:/models:ro \
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
    --disk-kv-path /prod-kvcache/l3 \
    --disk-kv-gib 88 \
    --disk-kv-restore \
    --spec mtp --draft-tokens 3 \
    --lm-head-draft \
    --admission-order prefill-first \
    --prefill-yield-max-ms 5000 \
    --pending-timeout-ms 6000
```

## L3 disk cache

The L3 tier provides persistent KV reuse across agent turns:

- **Rolling page chain**: only new-token pages are written; old pages roll off
- **Per-turn checkpoints**: preopener (prompt-end) and probe (admission-depth) anchors
- **State images**: full continuation state for zero-replay restore
- **Rolling eviction**: oldest pages evicted when the 88 GiB budget is reached
- **Restore pipeline**: disk reads by 5 CPU workers, pipelined H2D on a dedicated CUDA transfer stream

Restore performance for a 170K-token session: **1–5 seconds** (vs 60+ s re-prefill).

## Production hardening

Battle-tested in 24/7 multi-agent serving with fixes accumulated from production incidents:

- Non-blocking stderr: a full log pipe never wedges business threads
- Prefill yield: decode stays responsive during long prefills
- Dual materialization slots: two concurrent L3 restores
- Waiter rescue: FIFO head rescued into waiter slot when blocked by materialization
- Transfer-stream restore: H2D on a dedicated stream, overlapping decode
- mlock-anon: engine heap pinned against host swap
- Auto-recover: health-check + `wsl --shutdown` recovery script
- scanf return check: corrected smaps parsing for mlock detection
- Escape sequence fix: literal newline/tab in C char literals corrected
- Docker commit safety: binary changes survive watchdog container recreation

## Performance

See [docs/performance.md](docs/performance.md) for benchmark methodology.

### Concurrent MTP3 decode (RTX 5090)

| Model profile | C=1 tok/s | C=8 tok/s |
|---|---:|---:|
| Qwen3.8-27B nvfp4 | 143.8 | 766.6 |
| Qwen3.6-27B nvfp4 | 202.4 | 1,146.9 |
| Qwen3.6-35B-A3B | 593.0 | 1,313.8 |

### Research API scoring

Teacher-forced perplexity on the Pangram test sentence: **5.15** (deterministic, byte-identical across restarts and container recreations).

## Documentation

- [HTTP serving + Research API](docs/serving.md)
- [CLI guide](docs/cli.md)
- [Performance](docs/performance.md)
- [Perplexity](docs/perplexity.md)
- [Resource scheduling](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Engine architecture](docs/maintainer/engine-architecture.md)

## License

[Apache License 2.0](LICENSE)
