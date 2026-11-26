# 在 HypoSpace 上提升 LLM 的假设集合生成

课程项目（EC4300）。基准 **HypoSpace**（ICML 2026 spotlight）：衡量 LLM 能否在观测欠定的情况下，枚举三个域（因果图、3D 体素重建、Boolean 基因交互）中与观测一致的全部假设所组成的**集合**。

三个指标，均以百分比报告：

| 指标 | 含义 |
|---|---|
| **VR** 有效性 | 与观测一致的提议占比 |
| **NR** 新颖性 | 不重复的提议占比 |
| **RR** 恢复率 | 被恢复的可容许集合 $\mathcal{H}_O$ 占比 |

预算按实例计算：`n_queries = max(1, floor(|H_O| * multiplier))`。不同 multiplier 下的 RR 不可比，所以下表每一行都标注预算。

## Total新增：9B 反馈与自适应探索实验

已完成三个领域、五组任务的四方法双种子评测，共 97,952 次正式请求。实现、复现命令与限制见 [运行说明](docs/adaptive_evaluation.md)，完整结果见 [正式报告](docs/adaptive_results_zh.md)，机器可读摘要见 `results/adaptive_9b/`。

这些实验重跑了匹配基线，并修正了 Boolean 验证器的一致性问题；不要与下方历史基线混用。UCB 并非所有任务最优，固定修复是重要对照。

## Marcos新增：9B 上的多方向探索, 基于 9 个 trick 进一步涨点

在上一节的四种固定策略之上，根据Marcos 的研究思路继续搜索**推理流程**。agent 只改 `autoresearch/policy.py`（prompt、历史呈现、修复反馈、变异、动作选择），预算、验证器、dev/test 划分和评分全部锁定；每轮在 dev 实例上评一次，分数变好保留、变差回退。不训练、不改权重、不改服务。

dev 与 test 按实例 id 不相交：causal 沿用已有的 200 个 test id，dev 取 32 个；3D 与 Boolean 各对半分。循环共 35 次实验（另有 4 次重跑校验），keep 门槛取实测噪声（同一版本重跑相差 0.33–0.81）的 2 倍，中途从 0.8 提高到 1.6。

### 总体结果

dev 分数 **19.59 → 33.29（+13.7）**。最终有效的不是单一方法，而是 6 次通过门槛的改动叠加成的复合策略。

### 最终评测

留出集只在循环结束后跑一次；全量与正式报告同口径。

| 任务 | repair_only |    组合方法全量 |   组合方法 留出 | 留出实例数 |
|---|---:|----------:|----------:|---:|
| causal | 11.28 | **24.20** | **24.20** | 200 |
| 3d | 22.68 | **44.10** | **44.56** | 65 |
| boolean_basic | 27.14 | **43.43** | **41.85** | 18 |
| boolean_extended | 20.15 | **27.19** | **24.85** | 39 |
| boolean_full | 11.88 | **18.26** | **19.51** | 40 |

- **全量**：causal 为 200 个 test id，3D/Boolean 为全部实例；其中 3D/Boolean 有一半实例用于挑选策略，这一列偏乐观。
- **留出**：只用循环中从未见过的实例，是干净的数字。
- 两列相差不超过 2.4 个百分点，说明提升主要来自可迁移的流程改动，而不是对 dev 实例的过拟合；留出集的 3D/Boolean 只有 18–65 个实例，区间会比较宽。

### 通过门槛的改动

| # | 改动 | dev 分数 | 增量 | 主要受益 |
|---|---|---:|---:|---|
| — | repair_only（起点） | 19.59 | — | — |
| 1 | 重复时明确告诉模型"和之前某条重复了" | 22.02 | +2.43 | 全面 |
| 2 | causal 把观测改写成可达性约束 + 历史按 key 去重 | 24.39 | +2.37 | causal 13.5 → 19.7 |
| 3 | 探索时以已找到的有效答案为起点做单点变异 | 26.45 | +2.06 | 3D、extended |
| 4 | 3D 用紧凑高度表代替逐层列出 + 列高提示 | 27.69 | +1.24 | 3D 31 → 39 |
| 5 | 单点编辑变异 + causal 变异优先（70%） | 30.22 | +2.53 | causal 25.8 |
| 6 | boolean 已用表达式列表放末尾 + 最少用过的变异基座 | 33.29 | +3.07 | basic 48.3 |

### 单独没过门槛、组合后产生效果的改动

| 改动 | 单独增量 | 后来并入 |
|---|---:|---|
| causal 可达性提示 | +0.49（causal +2.7） | #2（24.39） |
| 历史按 key 去重 | +0.51（3D +2.9） | #2（24.39） |
| 3D 列高提示 | +0.49（3D +1.65） | #4（27.69） |
| causal 变异优先 p=0.7 | +1.21（causal +2.1） | #5（30.22） |
| 单点编辑变异（仅 causal/3D） | +1.49（basic +4.3） | #5（30.22） |
| boolean 已用列表置末尾 | +1.43（bool +4.7/+3.5/+1.9） | #6（33.29） |

### 试过但无效的方向

| 方向 | 结果 |
|---|---|
| 信用分配：UCB 在 mutate/repair 间分配、causal 变异概率调到 1.0、boolean 变异优先 | −0.34 / −0.10 / −0.18；简单固定规则优于自适应 |
| 同一想法换表示：causal 末尾列已用边集合；basic 只放已用列表、去掉上游历史 | causal −5.7；basic −9；而 boolean 用表达式列表、3D 用高度表都是正收益 |
| 把有效答案直接塞进修复 prompt | −0.27 / −0.79 |
| 变异时同时展示其他有效答案 | 3D −3.3，信息过多反而分散 |
| 把变异排到重复反馈前面 | −1.03 |

两个反复出现的结论：

- **"告诉模型试过什么"的表示形式比信息本身更关键**：同一份已用集合，在 Boolean 用表达式列表、在 3D 用高度表是正收益，在 causal 用边集合并列在末尾则是明显负收益。
- **自适应不如固定规则**：三次信用分配尝试都没有超过手工设定的固定优先级。

### 在 27B 上复用的耗时

同一套 harness 可以切到 27B（`AR_RUN=27b autoresearch/remote_run.sh`），但 27B 服务是 `--max-num-seqs 8`，一轮 dev 明显更慢：

| 服务 | 并发 | 完成请求 | 生成 token | 输入 token | dev 一轮 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 32 | 约 10.0 req/s | 约 397 tok/s | 约 9,187 tok/s | 约 10 分钟 |
| Qwen3.8-27B | 8 | 1.68 req/s | 85 tok/s | 764 tok/s | 约 57 分钟 |

差在批大小而不是单条速度：每路流的生成速度接近（27B 约 10.6 tok/s，9B 约 12.4 tok/s），但 8 与 32 的并发把总吞吐拉开约 5 倍。27B 显存已用 30.2/32 GB，KV cache 余量很小，提高并发需要重启服务并有 OOM 风险。

9B 一行取自 32 并发压测与 dev 一轮实测，27B 一行为运行中两分钟的 vLLM 指标增量。**这套改动是否在 27B 上同样有效，尚未验证**；27B 的 3D 零样本已到 49.8 RR，提升空间比 9B 小。

### 复现

```bash
.venv/bin/python autoresearch/harness.py                          # dev，约 5,760 次请求
AUTORESEARCH_FINAL=1 .venv/bin/python autoresearch/harness.py --split test
```

完整规则见 `autoresearch/program.md`，评测入口见 `autoresearch/harness.py`。

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
