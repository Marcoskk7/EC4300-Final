# 团队共享知识库（现象 + 原因 + 做法）

## 环境

- Homebrew 默认 Python 跑不动这个仓库，用 `uv venv -p 3.11 .venv`（当前 3.11.16）。
- 依赖比上游 README 多一个 `sympy`（Boolean 域必需）；画图工具还要 `matplotlib`。
- 官方 config 的 `api_keys` 留空时回落读环境变量 `OPENROUTER_API_KEY`；key 不要写进 yaml。
- `third_party/HypoSpace` 必须与上游一致；`LocalLLM` 等改动只在 `patches/hypospace-local-llm.patch`，运行时由 `tools/bench.py` 复制到 `.build/hypospace` 再打补丁。直接改 `third_party` 会把上游代码混进 diff。

## 数据 / 评测

- `|H_O| = 1` 的实例白送分（VR=NR=RR=1）。causal n=4 占 21%，n=3 占 36%，boolean extended/full 各 2–3%，3D 和 basic 为 0。
- 主实验 causal 用 `--gt-filter 8-200 --n-samples 200 --seed 33550336`，实例 id 固化在 `data/splits/causal_n4_test.json`。**随机 200 个实例的结果（`results/legacy/`）与主实验不可比**，因为一半实例几乎没有信息量。
- 预算 `n_queries = max(1, int(|H_O| * multiplier))`；causal 默认 2.0，3D/Boolean 1.0。不同预算的 RR 不可比。
- 3D 响应格式是每层按行输出（`Layer 1:` 后跟 grid_size 行，每行 grid_size 个 0/1）。压成一行 9 个数会被解析成 1×9，VR=0。
- 3D hash 会先裁掉顶部全零层，"多垫空气"不算新假设。
- Boolean VR 不检查空间归属（`validate_expression("No idea") -> True`），对我们的模型影响 0–4.5 点，集中在 full 档。报数字时两套口径都给，见 `docs/protocol.md` §5。

## 本地服务

- 27B NVFP4 在 32GB 卡上必须 `--enforce-eager`，否则 CUDA graph capture OOM。
- 镜像没有 nvcc：`VLLM_USE_FLASHINFER_SAMPLER=0` + `--attention-backend TRITON_ATTN`。9B 不加也会在解码时 `FlashInfer backend is not available` 并退出。
- 3D 的 system prompt 必须禁止解释，否则 256 token 到不了 `Structure:`，parse rate = 0。
- `temperature` 别设 0，否则每条 query 退化成同一假设，NR 崩掉。
- 同一模型不同服务配置的数字不可混用：27B 在 3D 上重跑同一 30 个实例，VR/NR/RR 从 70.5/67.2/45.4 变成 73.5/70.7/52.3。

## 记录

- 每次评测一个 run 目录（`runs/`），`meta.json` 自动记录子集、预算、服务配置、commit；榜单用 `make report` 生成，不手写。
- 数字从 JSON 的 `statistics` 取，不手抄。
