# EC4300 PlatforMAX 团队访问说明

- `ec4300_team`：ed25519 私钥

## 使用

```bash
chmod 600 ec4300_team

# 直连
ssh -i ec4300_team -p 20049 isaac@10.1.3.100   # qwen27b
ssh -i ec4300_team -p 20050 isaac@10.1.3.100   # qwen9b
```

当前两台机器

| 机器 | 端口 | 内容 | 环境位置 |
|---|---:|---|---|
| qwen27b | 20049 | 27B NVFP4，vLLM 在 `127.0.0.1:8000`，模型名 `qwen3.8-27b` | 持久卷 `/opt/data/private/isaaclab/ec4300` |
| qwen9b | 20050 | 9B bf16，vLLM 在 `127.0.0.1:8001`，模型名 `qwen3.5-9b` | 持久卷 `/opt/data/private/isaaclab/ec4300-9b` |

9B 启动 vLLM：

```bash
/opt/data/private/isaaclab/ec4300-9b/scripts/start_vllm_9b.sh
# 模型 qwen3.5-9b，端口 8001
```

9B 的五个 zero-shot 结果：

```text
/opt/data/private/isaaclab/ec4300-9b/results/zero_9b_*.json
```

27B 启动 vLLM：

```bash
/opt/data/private/isaaclab/ec4300/scripts/start_vllm_27b.sh
```

27B 的五个结果在 `/opt/data/private/isaaclab/ec4300/results/zero_27b_*.json`。
