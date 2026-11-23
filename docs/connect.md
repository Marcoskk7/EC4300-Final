# Team access

Both GPU instances sit behind one jump host. **Ports 20049/20050 are SSH, not HTTP**; vLLM
listens on the guest's loopback. `make tunnel` forwards them to 18000 (27B) and 18001 (9B).

```bash
export HYPOSPACE_SSH_KEY=/path/to/ec4300_team   # optional, if not in ~/.ssh/config
make tunnel
make tunnel-status
```

| Instance | SSH port | vLLM in the guest | Tunnel | Model | Workspace |
|---|---:|---|---:|---|---|
| `qwen27b` | 20049 | `127.0.0.1:8000` | 18000 | `qwen3.8-27b` (NVFP4) | `/opt/data/private/isaaclab/ec4300` |
| `qwen9b` | 20050 | `127.0.0.1:8001` | 18001 | `qwen3.5-9b` (bf16) | `/opt/data/private/isaaclab/ec4300-9b` |

The private key is distributed out of band and must never be committed. Serving flags, GPU
sharing, and restart commands are in `hardware.md`.
