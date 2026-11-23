# 评测协议

唯一协议是 `configs/suites/main.yaml`，所有被报告的数字都来自它。若某次 run 没用它，数字必须单列一张表并写明差异。

## 1. 预算

$$n\_queries = \max\left(1, \left\lfloor |\mathcal{H}_O| \cdot \text{multiplier} \right\rfloor\right)$$

按实例计算，其中 $|\mathcal{H}_O|$ 是与观测一致的假设数。query 越多，恢复整个集合的机会越大，因此**不同 multiplier 下的 RR 不可比**。论文本身也报告了 3× 预算不能解决 RR（其 Table 6），所以我们在同预算下比较，而不是靠加预算买覆盖。

| 任务 | multiplier | 原因 |
|---|---|---|
| causal n=4 | 2.0 | 该域上游默认值；两个基线都用了它 |
| 3D、boolean | 1.0 | 这两个域的上游默认值 |

## 2. 任务与子集

| 任务 | 数据集 | 实例数 | 说明 |
|---|---|---|---|
| `causal_n4` | `data/causal/n4.json` | 847 个中取 200 | `--gt-filter 8-200`：去掉 `\|H_O\| = 1` 的白送分实例。 |
| `3d` | `data/3d/g3_h3_b3.json` | 129（全量） | 俯视图，grid 3，最大高度 3，最大方块数 3 |
| `boolean_basic` | `data/boolean/basic_d2.json` | 35（全量） | 深度 2，basic 算子集 |
| `boolean_extended` | `data/boolean/extended_d2.json` | 78（全量） | 深度 2，extended 算子集 |
| `boolean_full` | `data/boolean/full_d2.json` | 80（全量） | 深度 2，全部算子 |

seed 统一为 `33550336`，任何机器上抽到的实例相同。200 个 causal 实例 id 已提交在 `data/splits/causal_n4_test.json`。

## 3. 留出集

被评测的实例 id 已提交（causal 见 `data/splits/causal_n4_test.json`），泄漏可查：任何见过被评测实例的模型，其结果行作废。

## 4. 可比性规则

- 必须使用同一份 `configs/suites/main.yaml`，否则该行不进榜单。
- 服务配置必须一致（量化方式、`max_tokens`、thinking 模式）；这些全部记录在 `meta.json`。
- 每次 run 只改一个变量：同时改方法、prompt 和预算的 run 证明不了任何东西。

## 5. 汇报与 Boolean 评测器问题

- 每次 run：`runs/<date>_<model>_<suite>[_<tag>]/meta.json` + 每个任务一个 JSON。
- 榜单是生成的，不手写：`make report`。
- RR 旁边必须给预算列，VR/NR 旁边必须给 parse rate。
- Boolean 的 validator 不检查提议是否属于假设空间，因此在观测全为 0 的 7/35 个 basic 实例上，垃圾字符串也能拿 VR=1。报告官方数字，并标注受影响的行。
- 本地量化模型是开放权重参照，不是论文的复现。
