# 在 HypoSpace 上提升 LLM 的假设集合生成

课程项目（EC4300）。基准 **HypoSpace**（ICML 2026 spotlight）：衡量 LLM 能否在观测欠定的情况下，枚举三个域（因果图、3D 体素重建、Boolean 基因交互）中与观测一致的全部假设所组成的**集合**。

三个指标，均以百分比报告：

| 指标 | 含义 |
|---|---|
| **VR** 有效性 | 与观测一致的提议占比 |
| **NR** 新颖性 | 不重复的提议占比 |
| **RR** 恢复率 | 被恢复的可容许集合 $\mathcal{H}_O$ 占比 |

预算按实例计算：`n_queries = max(1, floor(|H_O| * multiplier))`。不同 multiplier 下的 RR 不可比，所以下表每一行都标注预算。

## 方法概况

强模型在 causal 和 3D 上的 VR 已饱和，论文也表明温度、top-p 和 3× 预算都不能提升 RR（其 Table 6）。因此提升空间在于**预算怎么花**：失败来自重复和无效提议，而不是模型容量不足。

两个服务端点：

- Qwen3.5-9B（端口 18001）——基座权重，经隧道评测。
- Qwen3.8-27B（端口 18000）——推理侧改动（历史假设记忆、覆盖引导 prompt、批处理、前缀复用）。上游 runner 串行发请求，且每轮重发全部历史，引擎侧确有优化空间。

两者用同一协议、同一命令评测，数字可比。

## 结果

零样本基线，`configs/suites/main.yaml`，seed 33550336（`make report` 从 `runs/` 重新生成此表）：

| 任务 | 实例数 | 预算 | 模型 | parse | VR | NR | RR |
|---|---|---|---|---|---|---|---|
| causal n=4 | 200 | 2.0× | Qwen3.5-9B | 100.0 | 25.9 | 38.1 | 8.3 |
| causal n=4 | 200 | 2.0× | Qwen3.8-27B | 95.7 | 28.7 | 42.0 | 20.7 |
| 3D g3/b3 | 129 | 1.0× | Qwen3.5-9B | 100.0 | 57.1 | 42.5 | 22.7 |
| 3D g3/b3 | 129 | 1.0× | Qwen3.8-27B | 99.9 | 71.7 | 69.9 | 49.8 |
| boolean basic | 35 | 1.0× | Qwen3.5-9B | 100.0 | 80.8 | 32.7 | 26.6 |
| boolean basic | 35 | 1.0× | Qwen3.8-27B | 100.0 | 70.9 | 71.1 | 49.5 |
| boolean extended | 78 | 1.0× | Qwen3.5-9B | 100.0 | 66.1 | 32.1 | 16.5 |
| boolean extended | 78 | 1.0× | Qwen3.8-27B | 100.0 | 50.5 | 56.4 | 36.0 |
| boolean full | 80 | 1.0× | Qwen3.5-9B | 100.0 | 59.7 | 14.3 | 8.0 |
| boolean full | 80 | 1.0× | Qwen3.8-27B | 100.0 | 43.3 | 37.4 | 20.0 |

causal 使用 `--gt-filter 8-200`：`|H_O| = 1` 的实例是白送分，已排除。每一行使用同一批 200 个实例 id（`data/splits/causal_n4_test.json`）。

## 常用命令

```bash
make setup                        # 由 env/bench.txt 创建 .venv（Python 3.11）
make test                         # 评测器自检，脚本化 LLM，不联网
make tunnel                       # 到两个 GPU 节点的 SSH 隧道（见 docs/hardware.md）
make tunnel-status
make smoke                        # 每个域 2 个实例 × 2 个固定 query（默认 9B）
make smoke MODEL=configs/models/qwen3.8-27b.yaml   # 对 27B 冒烟（走 18000 隧道）
make eval MODEL=configs/models/qwen3.5-9b.yaml SUITE=configs/suites/main.yaml
make eval MODEL=... ARGS="--served-model qwen3.5-9b-alt --tag alt-v1"
make report                       # 由 runs/ 生成 results/leaderboard.{md,csv}
make monitor                      # GPU、vLLM 队列、单次 run 进度
make monitor WATCH=30
```

`make eval` 可续跑：对同一 `--run-dir` 重跑会跳过已完成任务。每次 run 在 `runs/<date>_<model>_<suite>[_<tag>]/meta.json` 中记录解析后的端点、采样设置、预算与 git commit；`report.py` 只读这些文件，数字无需手抄。

## 目录结构

```
configs/models/          每个服务端点一个文件（url、采样、thinking 模式）
configs/suites/          评测协议：任务、子集、预算、seed
scripts/tunnel.sh        SSH 隧道：18000 -> 27B 节点，18001 -> 9B 节点
data/                    固定 seed 的基准实例 + 留出实例 id
third_party/HypoSpace/   上游基准，固定版本（见 third_party/UPSTREAM.md）
patches/                 我们对上游的改动（patch 形式）
tools/bench.py           .build/hypospace = 上游原样 + patch（gitignored）
tools/run_eval.py        唯一评测入口
tools/report.py          榜单生成
tools/monitor.py         实时状态
tools/sanity_check.py    评测器正确性检查
runs/                    每次评测一个目录（结果 + 日志 + 出处）
```

## 诚实性声明

- 基线和方法在同一任务上使用同一预算；multiplier 出现在每张表里。
- 3D 与 Boolean 全量评测；causal 使用 `data/splits/causal_n4_test.json` 中的 200 个难实例；被评测的 id 已提交，可检查泄漏。
- 评测器已知问题：Boolean validator 不检查提议是否属于假设空间，在观测全为 0 的 7/35 个 basic 实例上，无意义字符串也能拿 VR=1。我们报告官方数字并标注受影响的行。
- 本地模型的量化/服务方式与论文 API 模型不同；这些行是开放权重参照，不是论文表格的复现。
