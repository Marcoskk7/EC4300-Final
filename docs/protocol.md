# 评测协议

协议由两份 suite 组成：causal 主基线用 `configs/suites/causal_full.yaml`（n=4 全量 847 条），3D 与 Boolean 用 `configs/suites/main.yaml`。所有被报告的数字都来自这两份。若某次 run 没用它们，数字必须单列一张表并写明差异。

## 1. 预算

$$n\_queries = \max\left(1, \left\lfloor |\mathcal{H}_O| \cdot \text{multiplier} \right\rfloor\right)$$

按实例计算，其中 $|\mathcal{H}_O|$ 是与观测一致的假设数。query 越多，恢复整个集合的机会越大，因此**不同 multiplier 下的 RR 不可比**。论文本身也报告了 3× 预算不能解决 RR（其 Table 6），所以我们在同预算下比较，而不是靠加预算买覆盖。

| 任务 | multiplier | 原因 |
|---|---|---|
| causal n=4 | 2.0 | 上游 `run_causal_benchmark.py` 的默认值（`query_multiplier=2.0`） |
| 3D、boolean | 1.0 | 这两个域的上游默认值 |

## 2. 任务与子集

| 任务 | 数据集 | 实例数 | 说明 |
|---|---|---|---|
| `causal_n4_all`（**主基线**） | `data/causal/n4.json` | 847（全量） | 不加 `\|H_O\|` 过滤，含 175 条 `\|H_O\| = 1` 的实例。 |
| `causal_n4`（旧，参照） | `data/causal/n4.json` | 847 个中取 200 | `--gt-filter 8-200`：去掉 `\|H_O\| = 1` 的实例。与 `causal_n4_all` 不可比。 |
| `3d` | `data/3d/g3_h3_b3.json` | 129（全量） | 俯视图，grid 3，最大高度 3，最大方块数 3 |
| `boolean_basic` | `data/boolean/basic_d2.json` | 35（全量） | 深度 2，basic 算子集 |
| `boolean_extended` | `data/boolean/extended_d2.json` | 78（全量） | 深度 2，extended 算子集 |
| `boolean_full` | `data/boolean/full_d2.json` | 80（全量） | 深度 2，全部算子 |

seed 统一为 `33550336`，任何机器上抽到的实例相同。旧的 200 个 causal 实例 id 已提交在 `data/splits/causal_n4_test.json`；全量 847 条就是 `n4.json` 的全部实例。

### max_tokens 截断

`max_tokens` 为 1024。若某次 query 达到上限（`finish_reason == "length"`）且解析失败，runner 不重试，只在该实例的 `truncated_queries` 里记下，结果顶层的 `truncation` 汇总截断次数和实例 id。全量跑完后，用 `--only-ids <截断 id 列表> --max-tokens 2048` 只重跑这些实例，再用 `tools/replace_samples.py` 按实例整条替换、重算统计。替换过的结果在 `replaced_samples` 里记下实例 id 和重跑时的 `max_tokens`，榜单的 max_tokens 列也会标出（如 `1024; 187 rerun @2048`）。

### 并发

causal 可用 `--shards N` 把实例按轮转切成 N 片并行跑（vLLM 需 `--max-num-seqs >= N`），`tools/merge_shards.py` 合并后的实例顺序与不分片一致，统计按上游算法重算。并发时 vLLM 会把多个请求拼批计算，在 temperature 0.7 下与串行运行不是逐位相同，但两者本来都是随机的。分片数记录在 `meta.json` 的 `shards`。

## 3. 留出集

causal 主基线评测 `n4.json` 的全部 847 条，因此**任何 n=4 causal 实例都不能进训练数据**；训练用 n=3 或其他节点数生成的数据。旧的 200 条 id 仍在 `data/splits/causal_n4_test.json`。泄漏可查：任何见过被评测实例的模型，其结果行作废。

## 4. 可比性规则

- causal 必须使用 `configs/suites/causal_full.yaml`，3D/Boolean 必须使用 `configs/suites/main.yaml`，否则该行不能与主基线比较。
- 服务配置必须一致（量化方式、`max_tokens`、thinking 模式）；这些全部记录在 `meta.json`。
- 每次 run 只改一个变量：同时改方法、prompt 和预算的 run 证明不了任何东西。

## 5. 汇报与 Boolean 评测器问题

- 每次 run：`runs/<date>_<model>_<suite>[_<tag>]/meta.json` + 每个任务一个 JSON。
- 榜单是生成的，不手写：`make report`。
- RR 旁边必须给预算列，VR/NR 旁边必须给 parse rate。
- Boolean 的 validator 不检查提议是否属于假设空间，因此在观测全为 0 的 7/35 个 basic 实例上，垃圾字符串也能拿 VR=1。报告官方数字，并标注受影响的行。
- 本地量化模型是开放权重参照，不是论文的复现。
