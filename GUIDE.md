# NInfer-L3 全面指南

> 从零开始了解、部署、使用、维护 NInfer 推理引擎 —— 写给所有人群的完整参考文档。

---

## 目录

1. [什么是 NInfer](#1-什么是-ninfer)
2. [它能做什么](#2-它能做什么)
3. [支持的模型](#3-支持的模型)
4. [硬件要求](#4-硬件要求)
5. [安装部署](#5-安装部署)
6. [API 使用](#6-api-使用)
7. [研究接口](#7-研究接口)
8. [L3 磁盘缓存](#8-l3-磁盘缓存)
9. [性能数据](#9-性能数据)
10. [架构原理](#10-架构原理)
11. [运维与排障](#11-运维与排障)
12. [生产环境部署](#12-生产环境部署)
13. [安全与权限](#13-安全与权限)
14. [常见问题](#14-常见问题)
15. [已知限制](#15-已知限制)

---

## 1. 什么是 NInfer

NInfer 是一个**从零开始用 C++ 和 CUDA 编写的大模型推理引擎**，专门为单张
NVIDIA GeForce RTX 5090 显卡优化，目标是榨干这张卡的最大推理性能。

### 核心理念

- **极简专用**：一张 GPU、一个驻留模型、启动时固定 1–8 个并发请求。
  不做多卡、不做动态批处理、不做模型热切换——把所有复杂度都砍掉，
  只留下推理最需要的路径。
- **显式注册**：只支持预先注册好的模型格式（`.ninfer` 文件），
  不做运行时模型发现，不加载未注册的检查点。
- **从零实现**：注意力、KV 缓存、CUDA Graph、分页管理、调度器全部自己写，
  不依赖 vLLM/TensorRT-LLM 等推理框架。

### 与其他推理引擎的对比

| 特性 | NInfer | vLLM | TensorRT-LLM |
|---|---|---|---|
| 语言 | C++/CUDA 从零 | Python + C++ 内核 | C++ + Python 接口 |
| 专用硬件 | RTX 5090 (sm_120a) | 任意 NVIDIA | 任意 NVIDIA |
| KV 缓存 | 分页 + 磁盘 L3 缓存 + 状态镜像 | PagedAttention | KV cache |
| 推理解码 | MTP3 + CUDA Graph | 连续批处理 | Inflight Batching |
| 研究接口 | /v1/score + /v1/next-token | 无 | 无 |
| 部署模式 | 单机单卡 | 多卡分布式 | 多卡分布式 |

### 什么时候选 NInfer

- 单张 5090 + 单个大模型 + 7×24 小时不间断服务
- 需要磁盘缓存避免重复 prefill（多轮对话/agent 场景）
- 需要研究级别的概率分布接口
- 想要一个精简可控、从零构建的推理引擎

### 什么时候不选 NInfer

- 多卡多模型分布式部署
- 需要动态切换模型
- 不跑在 RTX 5090 上

---

## 2. 它能做什么

### 文本生成

支持 thinking（深度思考）和 non-thinking 两种模式。通过 OpenAI 或
Anthropic 兼容的 HTTP API 或本地 CLI 调用。支持流式和非流式输出。

### 多模态

支持图像、多图、视频和混合多模态消息。通过 `--vision` 启动参数开启。

### 推测解码 (MTP)

Multi-Token Prediction：每次解码步预测 1–5 个 token，用 CUDA Graph
加速验证。实测吞吐提升 1.6×–5.7×（取决于并发数和模型）。

### 精确前缀复用

完全相同前缀的请求可以直接复用已计算的 KV 缓存：
- Device 级复用：KV 在 GPU 显存中直接复用
- Host 级复用：状态从锁页内存恢复到 GPU
- **L3 磁盘缓存复用**：KV 页链和状态镜像持久化到 NVMe，跨重启存活

### 研究接口

两个专用端点暴露模型内部概率分布：
- `/v1/score`：教师强制打分，逐 token logprob + 困惑度
- `/v1/next-token`：下一 token 全词表精确 softmax 分布

详见第 7 节。

---

## 3. 支持的模型

NInfer 支持 5 个显式注册的模型格式（`.ninfer` artifact）：

| 模型 | 量化格式 | Artifact 文件 | 下载 |
|---|---|---|---|
| Qwen3.6-27B | groupwise-int | `qwen3_6_27b.ninfer` | [HuggingFace](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | NVFP4 | `qwen3_6_27b_nvfp4.ninfer` | [HuggingFace](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | groupwise-int | `qwen3_8_27b.ninfer` | [HuggingFace](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | NVFP4 | `qwen3_8_27b_nvfp4.ninfer` | [HuggingFace](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | groupwise-int | `qwen3_6_35b_a3b.ninfer` | [HuggingFace](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

> **NVFP4** = NVIDIA 4-bit 浮点量化格式，权重体积约为 BF16 的 1/4。
> **groupwise-int** = 分组整数量化格式。
> **35B-A3B** = MoE 架构（A3B = 激活参数 3B），额外支持 DFlash 投机解码。

每个 artifact 文件内嵌 tokenizer、chat template 和媒体前端资源，
无需额外下载。

---

## 4. 硬件要求

### GPU

| 项 | 要求 |
|---|---|
| 型号 | NVIDIA GeForce RTX 5090（Blackwell 架构，sm_120a） |
| 显存 | ≥ 32 GB GDDR7 |
| 驱动 | NVIDIA 610.88+ |
| CUDA | 13.1+ |

> **注意**：编译时强制 `sm_120a`，不兼容其他架构。
> 5090 D（国内版）理论兼容但未测试。

### 系统内存

| 项 | 最小 | 推荐 |
|---|---|---|
| 物理 RAM | 24 GB | 32 GB+ |
| WSL VM 分配 | 16 GB | 24 GB |
| 页面文件 | 系统管理（建议在独立 SATA SSD 上） | |

> **16 GB VM 实测数据**：VmRSS 451MB、majflt 105K/8min（缺页飓风）→ 换页
> 飓风导致引擎反复重启。**24 GB 实测数据**：VmSwap 0、稳定运行。因此 16 GB
> 不可用，最低 24 GB。

### 存储

| 用途 | 位置建议 | 大小 |
|---|---|---|
| 模型 artifact | NVMe 任意分区 | 18–20 GB/个 |
| L3 磁盘缓存 | NVMe（与引擎同盘或独立 NVMe） | 88–96 GB |
| Docker 数据盘 | NVMe | ≥ 150 GB |
| Windows 系统 | NVMe | 常规 |
| ComfyUI（可选） | SATA SSD（独立带宽） | ≥ 250 GB |

### 网络

引擎监听 `0.0.0.0:8080`（容器内）。宿主机映射到 `9999`。
不需要外网访问（离线可用），HuggingFace 下载需要网络。

---

## 5. 安装部署

### 5.1 从源码构建

```bash
# 克隆仓库
git clone https://github.com/TIGillusion/ninfer-L3.git
cd ninfer-L3

# 安装依赖（Debian/Ubuntu）
apt-get install --yes --no-install-recommends \
  cmake ninja-build pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev

# 配置 CUDA（需要 CUDA 13.1 + sm_120a）
export PATH=/usr/local/cuda-13.1/bin:$PATH

# 构建生产二进制
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF
cmake --build build -j

# 产物
ls build/apps/ninfer-serve   # 推理服务器
ls build/apps/ninfer         # CLI 工具
```

### 5.2 Docker 构建

```bash
# 构建镜像（需要 NVIDIA Container Toolkit）
docker build --tag ninfer:latest .

# 或使用 docker-compose
docker compose up -d
```

### 5.3 下载模型

```bash
# 使用 HuggingFace CLI
pip install huggingface_hub
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer --local-dir ./models

# 或使用 wget（NVFP4 版本，18.4 GB）
wget https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/resolve/main/qwen3_8_27b_nvfp4.ninfer \
  -O models/qwen3_8_27b_nvfp4.ninfer
```

### 5.4 启动服务器

**最小配置**（测试用）：

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 --kv-capacity 240000 --max-concurrency 2
```

**生产配置**（多 agent 长上下文 + L3 磁盘缓存 + 研究接口）：

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
  --vision \
  --disk-kv-path ./l3cache/l3 \
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
  --preserve-thinking \
  --host 0.0.0.0 --port 8080
```

启动成功标志：
```
INFO  engine ready | qwen3.8-27b/nvfp4 | total 1m 30s | weights 16.3 GiB
INFO  capacity | KV 628,032 tokens, nvfp4 | pages 9,813/31,250
INFO  listening on http://0.0.0.0:8080
```

### 5.5 Docker 部署

```bash
# 构建镜像
docker build --tag ninfer:latest .

# 运行
docker run -d \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume ./models:/models:ro \
  --volume ./l3cache:/prod-kvcache \
  --ipc=host \
  --ulimit memlock=-1 \
  --restart unless-stopped \
  --name ninfer \
  ninfer:latest \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
    --max-context 400000 --kv-capacity 628000 --max-concurrency 5 \
    --kv-dtype nvfp4 --disk-kv-path /prod-kvcache/l3 --disk-kv-gib 88 \
    --disk-kv-restore --spec mtp --draft-tokens 3 --lm-head-draft
```

---

## 6. API 使用

### 6.1 OpenAI Chat Completions（最常用）

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 1024,
    "stream": true
  }'
```

支持：
- 流式（SSE）和非流式
- 工具调用（function calling）
- 结构化输出（JSON mode）
- thinking 模式（深度思考）
- top_logprobs（首 token 概率分布）
- 多模态（图像/视频消息）

### 6.2 OpenAI Responses API

支持完整的 Responses Core 规范，包括状态管理、输入 token 计数、
SSE 流式、工具调用、取消等。

### 6.3 Anthropic Messages API

Anthropic 格式的消息接口，支持 streaming、tool_use、count_tokens。

### 6.4 研究接口

#### POST /v1/score — 困惑度打分

给定一段文本，返回逐 token 的对数概率和整体困惑度。
适用于评估模型对特定文本的"理解程度"。

```bash
curl http://127.0.0.1:9999/v1/score \
  -H 'Content-Type: application/json' \
  -d '{"text": " Paris is the capital of France."}'
```

响应字段说明：
- `tokens`：分词后的 token 文本列表
- `token_logprobs`：每个 token 的自然对数概率。位置 0 为 null（首 token
  没有条件概率）。越接近 0 = 模型越确定；越负 = 模型越"意外"。
- `aggregate.perplexity`：困惑度 = exp(mean_nll)。越低 = 模型对这段文本
  越有"信心"。高质量英文文本通常 3–15；模型没见过的内容可以到 100+。

**使用场景**：
- 评估生成的文本质量（PPL 越低越好）
- 检测 out-of-distribution 输入
- 数据集质量过滤
- 对比不同模型的"理解程度"

#### POST /v1/next-token — 下一 token 概率分布

给定上下文，返回下一个 token 的全词表精确 softmax 分布。

```bash
curl http://127.0.0.1:9999/v1/next-token \
  -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is", "top_k": 5}'
```

响应字段说明：
- `top[]`：按概率降序排列的 top-K 候选，每个包含 `token_id`、
  `token_text`（可读文本）、`probability`（0–1）、`logprob`（自然对数）
- `chosen`：实际被采样器选中的 token（本接口用 greedy）

**使用场景**：
- 分析模型的预测行为
- RLHF 奖励模型的数据采集
- 教学/演示（展示模型的"思考过程"）

#### Chat Completions + top_logprobs

在常规 chat 请求中设置 `top_logprobs: N`（N = 1–100），响应中
`choices[0].logprobs.content[0]` 会包含首 token 的 top-N 概率分布。
这是 OpenAI 标准格式，可以直接被 OpenAI SDK 解析。

---

## 7. 研究接口

（已在第 6 节详述）

---

## 8. L3 磁盘缓存

### 8.1 是什么

L3 磁盘缓存是 ninfer 的**持久化 KV 复用层**——把计算过的 KV 缓存
写到 NVMe SSD 上，下次相同前缀的请求直接从盘上读回来，跳过重复 prefill。

### 8.2 为什么需要

Agent 场景下，每轮对话的 prompt = 上一轮的完整对话 + 新消息。
没有缓存时，每轮都要重新 prefill 整个对话（可能 100K+ token，60+ 秒）。
有了 L3 缓存，只需要 prefill 新增的几百个 token，后续从缓存恢复（1–5 秒）。

### 8.3 工作原理

```
Agent 发消息 → prompt = [system + 历史对话 + 新消息]
                     ↓
              检查 L3 缓存（前缀匹配）
                     ↓
        ┌── 缓存命中 ──┐     ┌── 缓存未命中 ──┐
        │  从盘恢复 KV  │     │  全量 prefill   │
        │  1–5 秒      │     │  60+ 秒         │
        └──────┬───────┘     └───────┬─────────┘
               ↓                     ↓
        新 token prefill（几十毫秒）
               ↓
           生成回复
               ↓
        新增 KV 页写回缓存
        锚点状态镜像写回
```

### 8.4 关键概念

| 概念 | 说明 |
|---|---|
| **页链** | KV 数据按页（~505 KB/页，27 token/页）追加写入 |
| **锚点** | 在关键位置保存的状态快照，用于快速恢复 |
| **preopener 锚点** | prompt 结束位置的状态快照，每轮对话自动写入 |
| **probe 锚点** | 准入深度探索时保存的快照 |
| **状态镜像** | 146 MB/个，包含 GDN 状态 + 完整循环状态 |
| **滚动驱逐** | 缓存满 88 GiB 时，最老的页自动淘汰 |
| **恢复管线** | 磁盘读（5 worker CPU 线程）→ 流水线 H2D（独立 CUDA 流）→ GPU |

### 8.5 性能数据

| 场景 | 无缓存 | 有缓存 |
|---|---|---|
| 170K token 会话恢复 | 60+ 秒 re-prefill | **1–5 秒** |
| 新增 1K token 的下一轮 | 仍需 60+ 秒 | **毫秒级**（只 prefill 新 token）|
| 缓存命中率（多轮 agent） | 0% | **98%+** |

---

## 9. 性能数据

### 并发 MTP3 解码（RTX 5090）

| 模型 | 1 并发 tok/s | 8 并发 tok/s | 加速比 |
|---|---:|---:|---:|
| Qwen3.6-27B nvfp4 | 202.4 | 1,146.9 | 5.67× |
| Qwen3.8-27B nvfp4 | 143.8 | 766.6 | 5.33× |
| Qwen3.6-35B-A3B | 593.0 | 1,313.8 | 2.22× |

### 单请求 Prefill 速度

| 模型 | 7K token | 260K token |
|---|---:|---:|
| Qwen3.8-27B nvfp4 | 8,340 tok/s | 2,203 tok/s |
| Qwen3.6-35B-A3B | 15,544 tok/s | 5,157 tok/s |

### L3 缓存命中时的 TTFT

| 场景 | 无缓存 TTFT | 有缓存 TTFT |
|---|---|---|
| 170K 会话新 turn | 60+ 秒 | **1–5 秒** |
| 新增 1K token 的 turn | 同上 | **毫秒级** |

---

## 10. 架构原理

### 10.1 整体架构

```
┌────────────────────────────────────────────┐
│              HTTP Server (:8080)           │
│  OpenAI Chat / Responses / Anthropic       │
│  Research API (/v1/score, /v1/next-token)  │
├────────────────────────────────────────────┤
│           Generation Service                │
│  准入控制 │ 排队 │ 超时 │ 取消 │ 优先级    │
├────────────────────────────────────────────┤
│           Engine Core (C++)                │
│  调度器 │ 批处理 │ CUDA Graph │ KV 管理   │
│  L3 磁盘缓存 │ 研究接口 │ 观测           │
├────────────────────────────────────────────┤
│         CUDA Runtime (sm_120a)             │
│  GDN 注意力 │ MTP │ NVFP4 反量化         │
└────────────────────────────────────────────┘
         │
    RTX 5090 (32 GB GDDR7)
```

### 10.2 关键路径

**Prefill（处理输入）**：
1. Tokenize → embedding → GPU kernel 计算 → KV 写入缓存
2. 支持分块（chunk = 4096 token）避免一次性占用过多显存
3. L3 缓存命中时跳过已缓存的 prefill，只处理新增部分

**Decode（生成输出）**：
1. CUDA Graph 捕获整个 decode 步骤（消除 kernel launch 开销）
2. MTP3：每步预测 3 个 draft token，批量验证
3. 每秒可处理 145–1147 tok/s（取决于并发数和模型）

**Materialization（物化 = 缓存恢复）**：
1. 磁盘读取：5 个 CPU worker 线程并行 pread
2. H2D 拷贝：在专用 transfer_stream 上执行（不阻塞 decode）
3. 完成：KV 池中的页已填充，序列可以继续 decode

### 10.3 内存布局

```
GPU 显存（RTX 5090, 32 GB GDDR7）
├── 权重（NVFP4）                    16.3 GB
├── KV 池（628K token, NVFP4）        6.3 GB
├── 设备状态镜像（2 slot）             0.3 GB
├── CUDA Graph + workspace           ~4 GB
└── 运行时余量                        ~1 GB

宿主内存（WSL VM, 24 GB）
├── Host KV 池（pinned, 6 GiB）       6.0 GB
├── Host 状态池（16 slot, pinned）     2.3 GB
├── 引擎堆/栈（mlock-anon 锁定）       1.17 GB
├── 权重流式加载缓冲                   有界/复用
├── dockerd + containerd              ~1 GB
├── snowluma（QQ bot）                ~1 GB
├── 页缓存（可回收）                  其余
└── VM swap（6 GB, C: swap.vhdx）     应急
```

---

## 11. 运维与排障

### 11.1 常用监控命令

```bash
# 引擎健康
curl http://127.0.0.1:9999/health

# 实时吞吐
docker logs ninfer --since 1m | grep throughput

# GPU 利用率
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# 错误/楔死
docker logs ninfer --since 1h | grep -c "ERROR\|wedged"

# L3 缓存命中率（最近请求）
docker logs ninfer --since 1h | grep -oP 'cache \d+ \(\d+\.\d+%\)'
```

### 11.2 故障排查表

| 症状 | 可能原因 | 处理 |
|---|---|---|
| health 返回 000 | 网络转发断了 | `wsl --shutdown`，等 30s 自动重启 |
| 引擎反复重启 | VM 内存不足 → OOM kill | 确认 VM ≥ 24G，检查 dmesg OOM |
| 引擎卡死（host 0%，GPU 空） | GPU-PV 通道挂起 | 哨兵 180s 自动重启，等恢复 |
| decode 变慢但没死 | 并发高或上下文长 | 正常，等 prefill 完成 |
| 首次请求很慢 | 权重首次从盘加载 | 正常，等待 |
| L3 缓存 miss | 新会话或被驱逐 | 正常，首次全量 prefill |
| bootstrap 失败 (getty) | docker-desktop 发行版损坏 | `wsl --unregister docker-desktop` + 重启 Docker Desktop |
| Docker CLI 卡死 | 数据面死亡 | `wsl --shutdown`，等 Docker 自愈 |
| C 盘满 | wsl-crashes 转储 | 清理 `%TEMP%\wsl-crashes` |

### 11.3 Docker 数据面死亡

这是 WSL2 Docker Desktop 的已知问题：backend 进程死亡后，dockerd 和
容器还在 VM 里运行，但宿主机无法通过 API 管道访问它们。

**症状**：docker CLI 卡死或报错、端口转发失效、容器日志停止更新。
但引擎进程本身还在跑（/proc 可见、RSS 正常）。

**修复**：
```bash
wsl --shutdown
# 等 30 秒，Docker Desktop 自动重启 VM 和容器
# 如果 Docker Desktop 本体也没了：
# 双击 E:\vllm\恢复Docker服务.bat（会弹 UAC）
```

### 11.4 docker-desktop 引导失败

**症状**：`getty: cmdline has console=hvc0 but does not exist in /etc/securetty`

**修复**（无需 UAC）：
```bash
wsl --unregister docker-desktop
# Docker Desktop 会自动重建发行版
```

如果 Docker Desktop 本体也没跑：
```powershell
Start-Process -FilePath 'C:\Program Files\Docker\Docker\Docker Desktop.exe' -Verb RunAs
```

### 11.5 pagefile 配置

建议 pagefile 放在与引擎 L3 缓存**不同的物理盘**上：
- 引擎 L3 在 Ti600 (NVMe) → pagefile 放 SATA SSD
- 这样换页 I/O 不会与 L3 缓存 I/O 竞争同一块盘的队列

---

## 12. 生产环境部署

### 12.1 Docker Compose 生产编排

```yaml
version: "3.8"
services:
  ninfer:
    build: .
    image: ninfer:latest
    restart: unless-stopped
    runtime: nvidia
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models:ro
      - l3cache:/prod-kvcache
    ipc: host
    ulimits:
      memlock: -1
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command:
      - ninfer-serve
      - /models/qwen3_8_27b_nvfp4.ninfer
      - --max-context=400000
      - --kv-capacity=628000
      - --max-concurrency=5
      - --kv-dtype=nvfp4
      - --disk-kv-path=/prod-kvcache/l3
      - --disk-kv-gib=88
      - --disk-kv-restore
      - --spec=mtp --draft-tokens=3 --lm-head-draft
      - --admission-order=prefill-first
      - --prefill-yield-max-ms=5000
      - --pending-timeout-ms=6000
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://127.0.0.1:8080/health"]
      interval: 30s
      timeout: 5s
      start_period: 180s

volumes:
  l3cache:
```

### 12.2 安全网

1. **auto_recover.bat**：每 60s 探测 health，连续失败 → `wsl --shutdown` 自愈
2. **看门狗**：监控容器状态，异常时 `docker restart`
3. **哨兵**：引擎内 180 秒无进展自动退出（让 Docker 重启策略接管）

### 12.3 镜像更新流程

```bash
# 构建新版本
cd nb-build && ninja  # 或重新 cmake --build

# 部署到运行容器
docker cp nb-build:/build/apps/ninfer-serve ninfer:/usr/local/bin/ninfer-serve.new
docker exec ninfer mv /usr/local/bin/ninfer-serve.new /usr/local/bin/ninfer-serve
docker restart ninfer

# commit 进镜像（防止容器重建丢失）
docker commit ninfer ninfer:l3-yarn31

# 验证
docker run --rm --entrypoint md5sum ninfer:l3-yarn31 /usr/local/bin/ninfer-serve
```

---

## 13. 安全与权限

### 13.1 网络安全

- 引擎监听 `0.0.0.0:8080`（容器内）——如果需要仅本机访问，
  改为 `--host 127.0.0.1`
- 研究接口（/v1/score、/v1/next-token）与常规 API 共享鉴权
- 无内置 API key——生产环境应在前面放反向代理（nginx/caddy）加认证

### 13.2 文件权限

- 模型 artifact：只读挂载（`:ro`）
- L3 缓存：读写挂载，建议使用专用目录
- Docker socket：不要暴露给不可信进程

### 13.3 Docker 安全

- 使用 `--restart unless-stopped`（不是 `always`，避免系统启动时意外启动）
- `--ipc=host` 是 NVIDIA GPU 共坐内存必需的，但有安全影响
- 不要在不信任的容器中运行引擎

---

## 14. 常见问题

### Q: 和 vLLM 比有什么优势？

NInfer 是从零构建的专用引擎，针对单张 5090 上的 Qwen 模型做了极致优化。
vLLM 是通用框架，支持多模型多卡但单卡性能不如专用引擎。如果你的场景
正好匹配（5090 + Qwen + 单模型），NInfer 的性能更好、部署更简单。

### Q: 支持哪些 GPU？

只支持 RTX 5090（sm_120a）。5090 D 理论兼容但未测试。不支持 A100/H100/
4090 等其他卡。

### Q: 多轮对话的缓存怎么工作？

每轮对话结束后，新增 KV 页和状态锚点自动写入 L3 磁盘缓存。下一轮
如果前缀匹配（system + 历史对话不变），直接从缓存恢复 KV，只 prefill
新增的 token。缓存命中时 TTFT 从 60+ 秒降到 1–5 秒。

### Q: 困惑度（PPL）是什么意思？

困惑度衡量模型对一段文本的"熟悉程度"。PPL = exp(平均负对数似然)。
- PPL = 1：模型完全预测正确（不可能）
- PPL = 5–15：模型很熟悉（正常文本）
- PPL = 50+：模型很意外（乱码、外语文本等）
- PPL = 1000+：完全乱码或非自然语言

### Q: 为什么 /v1/score 返回的 token_logprobs[0] 是 null？

位置 0 的 token 没有前缀上下文，因此没有条件概率。这是数学上的限制
（P(第一个token) 需要一个起始分布，但因果语言模型没有起始分布）。

### Q: L3 缓存满了怎么办？

88 GiB 预算满时，最老的页自动驱逐。不需要手动清理。如果你想完全重置
缓存：清空 L3 缓存目录并重启引擎。

### Q: 能在 Windows 上原生运行吗？

目前不能。引擎需要 Linux CUDA 运行时。WSL2 提供了 GPU-PV 虚拟显卡
支持，但有已知的稳定性问题（GPU-PV 通道挂起导致引擎重启）。长期方案
是移植到原生 Windows CUDA。

### Q: 多个 L3 缓存可以共享吗？

可以。多个 Engine 实例可以指向同一个 L3 缓存目录（只读），实现缓存共享。
但写并发需要注意（建议每个引擎用自己的 L3 目录）。

---

## 15. 已知限制

| 限制 | 说明 |
|---|---|
| 单 GPU | 不支持多卡推理、模型并行、流水线并行 |
| 单模型 | 一个引擎实例只加载一个模型，不支持运行时切换 |
| 固定并发 | 启动时固定 1–8 个并发，不支持动态扩缩 |
| sm_120a | 编译硬编码 RTX 5090 架构，不兼容其他 GPU |
| Qwen 系列专用 | 只支持 Qwen3.6/3.8 系列模型 |
| WSL2 限制 | GPU-PV 驱动偶发不稳定（通道挂起），每次自愈约 3 分钟 |
| 32 GB RAM 下限 | 引擎 11 G + 系统 4 G + ComfyUI 峰值 15 G ≈ 30 G，余量小 |
| 研究接口无鉴权 | /v1/score 和 /v1/next-token 无内置认证，需反向代理保护 |

---

*文档版本：2026-11-10 | 基于 NInfer-L3（TIGillusion/ninfer-L3）*
*引擎版本：ec745df0（mlock-anon + transfer_stream + 研究接口）*
