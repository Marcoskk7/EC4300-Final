# GPU nodes and serving

Two single-GPU instances (RTX 5090, 32 GB, sm_120) behind the same jump host. **Ports 20049/20050 are SSH ports, not HTTP**: each node runs vLLM bound to its own
`127.0.0.1`, and we reach it through an SSH tunnel. API keys are never stored in this repo; the
servers run without auth on the internal network.

| Instance | SSH port | vLLM in the guest | Local tunnel | Config |
|---|---|---|---|---|
| `qwen27b` | 20049 | `127.0.0.1:8000` | 18000 | `configs/models/qwen3.8-27b.yaml` |
| `qwen9b` | 20050 | `127.0.0.1:8001` | 18001 | `configs/models/qwen3.5-9b.yaml` |

```bash
make tunnel           # scripts/tunnel.sh up; idempotent
make tunnel-status    # which endpoints answer, and with which model id
make untunnel
```

`HYPOSPACE_SSH_HOST` (default `10.1.3.100`), `HYPOSPACE_SSH_USER` (default `isaac`), and
`HYPOSPACE_PORT_27B` / `HYPOSPACE_PORT_9B` override the defaults.

Running *on* a node needs no tunnel: pass the guest URL directly,

```bash
make eval MODEL=configs/models/qwen3.8-27b.yaml ARGS="--base-url http://127.0.0.1:8000/v1 --tag onnode"
```

## Serving flags that are not optional

- `--attention-backend TRITON_ATTN`: the image ships no FlashInfer cubins; the default backend
  dies on the first decode.
- `VLLM_USE_FLASHINFER_SAMPLER=0`: no `nvcc` in the image, and FlashInfer's sampler JITs.
- 27B NVFP4 needs `--enforce-eager`, otherwise CUDA graph capture OOMs.
- 9B runs bf16 from `/opt/data/private/isaaclab/ec4300-9b/hf-cache/Qwen3.5-9B-bf16`
  (script: `ec4300-9b/scripts/start_vllm_9b.sh`).

## GPU sharing

One 32 GB card per node, so serving memory is explicit:

| 9B serving `--gpu-memory-utilization` | Serving | Left for other processes |
|---|---|---|
| 0.85 (current script) | ~27 GB | ~5 GB |
| 0.45 | ~15 GB | ~17 GB |

The serving configuration is recorded in `meta.json`; compare runs only when it matches.

To score a LoRA adapter without merging weights, serve with
`--enable-lora --lora-modules <name>=<path>` and evaluate with
`make eval ... ARGS="--served-model <name> --tag <name>"`.

## 3D prompt

Qwen writes prose before the structure unless the system prompt forbids it, and the answer never
fits in `max_tokens`. `tools/run_eval.py` sends "Answer with the requested Structure/Layer
format only, no explanation." for the 3D domain. Overriding `system_prompts` in a model config is
allowed, but check `parse_success_rate` first: below 0.9 the VR/NR/RR numbers are meaningless.

## Pitfalls

- Any prompt, `max_tokens`, or thinking-mode change moves parse rate, not only content. All of it
  goes into `meta.json` automatically; compare runs only when those fields match.
- `make smoke` first after any serving change. Parse rate 0 is a format problem, not capability.
- `runs/<run>/checkpoints/` is gitignored and deleted when a task finishes.
- The nodes cannot reach GitHub; sync this repo over SSH (`rsync -a --exclude .venv --exclude
  .build --exclude runs ./ isaac@<host>:/opt/data/private/isaaclab/ec4300-9b/code/`) rather than
  `git pull`.
