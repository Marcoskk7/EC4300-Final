# 文档

| 文件 | 回答什么 |
|---|---|
| `protocol.md` | 评测协议：预算、子集、seed、留出集、汇报规则。 |
| `baselines.md` | §0 论文 vs 本地对照总表；论文 Table 1–6、Qwen3.5-9B / Qwen3.8-27B 基线、失败模式分解。 |
| `hardware.md` | GPU 节点、vLLM 端点与必须加的参数、3D prompt。 |
| `lessons.md` | 已经踩过的坑：环境、数据格式、服务。 |
| `papers/HypoSpace_arXiv_2510.15614.pdf` | 论文 PDF，Table 1–6 的来源。 |

`results/leaderboard.{md,csv}` 由 `make report` 从 `runs/` 生成；`baselines.md` 是带论文数字和注意事项的注释版快照。
