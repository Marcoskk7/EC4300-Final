# GPU 节点与服务

两台单卡实例（RTX 5090，32 GB，sm_120），在同一跳板机之后。**20049/20050 是 SSH 端口，不是 HTTP 端口**：每个节点上的 vLLM 只绑定各自的 `127.0.0.1`，通过 SSH 隧道访问。API key 不写入仓库；服务在内网中不鉴权。

| 实例 | SSH 端口 | 实例内 vLLM | 本地隧道 | 配置 | 工作目录 |
|---|---|---|---|---|---|
| `qwen27b` | 20049 | `127.0.0.1:8000` | 18000 | `configs/models/qwen3.8-27b.yaml` | `/opt/data/private/isaaclab/ec4300` |
| `qwen9b` | 20050 | `127.0.0.1:8001` | 18001 | `configs/models/qwen3.5-9b.yaml` | `/opt/data/private/isaaclab/ec4300-9b` |

```bash
make tunnel           # scripts/tunnel.sh up；幂等
make tunnel-status    # 哪些端点可达，返回的 model id 是什么
make untunnel
```

`HYPOSPACE_SSH_HOST`（默认 `10.1.3.100`）、`HYPOSPACE_SSH_USER`（默认 `isaac`）、`HYPOSPACE_PORT_27B` / `HYPOSPACE_PORT_9B` 可覆盖默认值；私钥默认走 `~/.ssh/config`，也可用 `HYPOSPACE_SSH_KEY=/path/to/ec4300_team` 指定。私钥线下分发，绝不提交。

在节点**上**运行时不需要隧道，直接传 guest URL：

```bash
make eval MODEL=configs/models/qwen3.8-27b.yaml ARGS="--base-url http://127.0.0.1:8000/v1 --tag onnode"
```

## 必须加的服务参数

- `--attention-backend TRITON_ATTN`：镜像里没有 FlashInfer cubins，默认后端在第一次 decode 时就会挂。
- `VLLM_USE_FLASHINFER_SAMPLER=0`：镜像没有 `nvcc`，FlashInfer 的 sampler 会走 JIT。
- 27B NVFP4 必须 `--enforce-eager`，否则 CUDA graph capture 阶段 OOM。
- 9B 以 bf16 从 `/opt/data/private/isaaclab/ec4300-9b/hf-cache/Qwen3.5-9B-bf16` 加载。

`scripts/serve/start_vllm_{9b,27b}.sh` 是节点上运行脚本的副本（2026-11-23 取回）；以节点上的原件为准。两者分别使用 `--max-num-seqs 1`（27B）/ `4`（9B），这是做吞吐优化时第一个该动的参数。

## 显存分配

每个节点一张 32 GB 卡，因此服务占用的显存必须显式设置：

| 9B 服务 `--gpu-memory-utilization` | 服务占用 | 留给其他进程 |
|---|---|---|
| 0.85（当前脚本） | 约 27 GB | 约 5 GB |
| 0.45 | 约 15 GB | 约 17 GB |

**决策：训练时停服务。** 9B bf16 权重 16.7 GiB，0.85 时服务占约 27.7 GB，QLoRA 需要 10–16 GB，两者无法共存。流程固定为：

```bash
make serve-stop      # scripts/vllm9b.sh stop，训练前
# ... 训练、保存 adapter ...
make serve-start     # 评测前重新起服务
```

训练进程退出后再启动服务，并等它加载完成再跑 `make smoke`：加载中发出的请求会失败，runner 会重试，等于悄悄消耗预算。27B 服务属于 harness track，这个脚本不会碰它。

服务配置记录在 `meta.json`；只有配置一致时才能比较不同 run。

## 3D 提示词

除非 system prompt 明令禁止，Qwen 会先写一段散文再给结构，而答案在 `max_tokens` 内写不完。`tools/run_eval.py` 对 3D 域发送 "Answer with the requested Structure/Layer format only, no explanation."。允许在模型配置里覆盖 `system_prompts`，但先看 `parse_success_rate`：低于 0.9 时 VR/NR/RR 没有意义。

## 已知坑

- 任何 prompt、`max_tokens`、thinking 模式的改动都会改变 parse rate，而不只是内容。这些字段会自动写入 `meta.json`；只有字段一致时才能比较 run。
- 任何服务改动后先跑 `make smoke`。parse rate 为 0 是格式问题，不是能力问题。
- `runs/<run>/checkpoints/` 已被 gitignore，任务完成时会删除。
- 节点无法访问 GitHub；用 SSH 同步本仓库（`rsync -a --exclude .venv --exclude .build --exclude runs ./ isaac@<host>:/opt/data/private/isaaclab/ec4300-9b/code/`），不要 `git pull`。
