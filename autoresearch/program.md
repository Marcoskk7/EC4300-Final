# autoresearch: HypoSpace 推理流程优化

目标：在**同一请求预算**下，改进 Qwen3.5-9B 生成假设集合的流程，提高 dev 集上的平均 RR。不训练、不改权重、不改服务。

## 文件

| 文件 | 权限 | 内容 |
|---|---|---|
| `autoresearch/policy.py` | **唯一可改** | 每一轮的 messages 如何构造、如何利用上一轮结果（prompt、历史呈现、修复反馈、动作选择、结构探索） |
| `autoresearch/harness.py`、`autoresearch/final_eval.py`、`autoresearch/remote_run.sh` | 只读 | dev/test 划分、预算、LLM 调用、解析与校验、评分 |
| `autoresearch/test_harness.py` | 只读 | 离线检查 |
| `tools/`、`third_party/`、`configs/`、`data/`、`patches/` | 只读 | 上游基准与已有评测 |
| `autoresearch/results.tsv` | 只追加，不提交 | 实验记录 |

`policy.py` 只能 import `math random re itertools collections dataclasses typing statistics functools`，不能用 `open/eval/exec/__import__`，也不能访问 dunder 属性。harness 运行前会检查这些，违规直接报错。

## 预算规则（不可绕过）

- 每个实例的请求数由 harness 固定：causal 为 `2 × |H_O|`，其余为 `1 × |H_O|`。**一次 `next_messages()` 对应一次请求，对应一个候选。**
- policy 拿不到 `|H_O|`、预算、真解，也拿不到其他实例的信息。每个 (实例, seed) 都新建一个 Policy，不允许跨实例学习。
- 温度 0.7、`max_tokens` 1024、关闭 thinking，都由 model config 锁定。
- 要求模型一次输出多个候选没有用：upstream parser 每条回复只取一个候选。
- 超过 8192 上下文时请求被拒，但仍然消耗预算（输出里的 `context_rejected`）。历史越长越容易触发，需要自己控制 prompt 长度。

## 允许的方向（举例）

- 改写 system/user prompt，调整历史的呈现方式（全量、只列有效的、压缩、按结构分组）
- 修复反馈的粒度与措辞，以及何时修复、何时探索
- 动作选择策略（固定规则、UCB 变体、按重复率或错误率切换、按 domain 分开处理）
- 结构探索：基于已找到的有效解做局部变换（加边、删边、翻转边、移动方块、替换算子），提示模型去补没覆盖到的区域
- 用 `info.public`（节点、最大边数、网格大小、最大高度、变量、算子、深度）在本地枚举或检查假设空间，以此构造提示。可以用观测在本地筛选，但**最终提交的候选必须来自模型这一轮的回复**，harness 只解析回复本身

**禁止**：在 prompt 里直接写出完整的答案让模型抄。例如在本地枚举出所有与观测一致的 DAG，然后逐条要求模型复述。这样等于绕过了 LLM，结果不可报告。判定标准：如果把 LLM 换成"原样复读 prompt 里第一个候选"的程序，分数也一样高，那就是作弊。提示"方向"可以（比如"尝试 A->B 方向未出现过的图"），给出完整答案不行。

## 准备（每次新开一轮）

1. 约定 tag（例如 `nov26`），确认当前在分支 `autoresearch/<tag>` 上，然后读本文件、`harness.py`、`policy.py`。
2. 确认节点可用：`autoresearch/remote_run.sh --smoke > run.log 2>&1; grep -E "^(score|exit_code)" run.log`。评测**在 9B 节点本机跑**（vLLM 在 localhost，约 10 分钟一轮）；本机经 Mac mini 转两跳只负责推送 `policy.py` 和取回结果。连不上或报 `LOCKED FILES DIFFER` 就停下来告诉人。不要自己重启 vLLM，也不要改节点上的其他文件。
3. 离线测试：`.venv/bin/python -m unittest autoresearch/test_harness.py`
4. 如果 `autoresearch/results.tsv` 不存在，就创建，只写表头：
   `commit	score	causal	3d	basic	extended	full	status	description`
5. 跑一次未修改的 `policy.py` 作为 baseline，记为 `keep`。人已经在节点上跑过两次 repair_only 的 dev baseline，分别是 19.46 和 19.13，可以作为参照。

## 实验循环（直到满足终止条件）

1. 改 `policy.py`，每次只改一个想法，然后 `git commit`。
2. 运行：`autoresearch/remote_run.sh > run.log 2>&1`（不要用 tee，不要把输出灌进上下文）。脚本会把 `policy.py` 推到节点，在节点上脱离 SSH 后台运行（SSH 断了也不影响），每 30 秒查询一次，结束后把输出写进 run.log，并把 run 目录拉回本机 `runs/autoresearch/`。调用 Bash 时设 timeout 为 600000，并用 `run_in_background`，等它结束的通知；不要另开一个等待循环。
3. 读结果：`grep -E "^(score|rr_|context_rejected|truncated|transport_retries|seconds)" run.log`。没有输出说明崩了，用 `tail -n 50 run.log` 查原因。`transport_retries` 只是网络重试（连接断开或 5xx，请求没有到达模型），不影响分数，但数量多时要检查隧道。
4. 往 `results.tsv` 追加一行（score 和各任务 RR 保留 2 位小数，status 为 keep、discard 或 crash）。
5. **keep 条件**：score 比当前最好高出 **≥ 0.8**（噪声门槛，见下文），且 `context_rejected` 不增加。否则 `git reset --hard HEAD~1`。
   - 分数差不多（±0.8 以内）但代码明显更简单：keep。
   - 某个任务大幅上涨、另一个任务下跌：按 score 判断，但要在 description 里写明，之后可以尝试按 domain 分别处理。
6. 每 keep 5 次左右，原样重跑一次当前最好版本，记为 `rerun`，检查是否只是运气。重跑分数比记录低超过 1 的话，把它的记录分数改成两次平均。

一轮约 5,760 个请求，在节点上大约 10 分钟（实测 577 秒）。节点侧有 25 分钟超时（`exit_code: 124`），超时按 crash 处理；prompt 变长会让一轮变慢，这也是一种代价。崩溃如果是笔误就修了重跑；如果想法本身走不通，就记 crash，然后换下一个。

**不要中途停下来问人**：循环开始后不要问人要不要继续。没有想法时就重读 `harness.py` 里校验器给出的错误类型，看 `runs/autoresearch/<run>/jobs.jsonl` 里的失败模式（重复？无效？哪类错误最多？），组合之前差一点通过的改动，或者尝试更激进的思路。

## 终止条件

每跑完一个实验、写完 results.tsv 之后检查一次。满足下面任意一条就停止循环：

1. **2 小时没有提升**。"提升"指某次实验被 keep，并且 score 比之前最好的高出 ≥ 0.8；"代码更简单、分数持平"的 keep 不算提升。从最近一次提升的实验 commit 时间算起（`git log -1 --format=%ci <hash>`），到现在 ≥ 2 小时就停。baseline 那一行算作第一次提升。
2. **没有思路了**。必须先确认以下方向都试过了：去重和重复反馈、历史呈现方式、修复反馈、动作选择、结构探索、按 domain 分别处理、组合之前差一点通过的改动。然后再在 results.tsv 写一行 `stop`，在 description 里列出还剩什么没试、为什么判断不值得试。不能用这一条来逃避第 1 条：只要还能想出没试过的方向，就继续。

停止后按顺序做：

1. 确认当前 HEAD 就是 results.tsv 里分数最高的 keep 版本（不是的话，`git checkout <hash> -- autoresearch/policy.py` 恢复它，并 commit 成 `final: best policy <hash>`）。
2. 最终评测（只跑这一次，约 30–45 分钟）：`AR_FINAL=1 autoresearch/remote_run.sh > final.log 2>&1`，用 `run_in_background` 跑，timeout 设为 600000；超时被转到后台时等通知即可。会对两个 seed 跑两份数据：
   - **留出**：循环中从未用过的实例（causal 200 个 test id，3D/Boolean 各一半），这是干净的数字；
   - **全量**：留出 + 3D/Boolean 的 dev 那一半，实例集合与 `全领域工程优化正式结果.md` 相同。
3. 生成对比表：`<主仓库 .venv 的 python> autoresearch/final_eval.py --report runs/autoresearch/<final.log 里的 run_dir>`，结果写到该目录的 `final.md`。
4. 最后一条消息汇报：停止原因、实验总数、dev 上的最好分数和 baseline 19.59 的对比、保留下来的改动，以及 `final.md` 的表格。

## 噪声门槛从哪来

同一份 repair_only 代码跑了两次，score 为 19.46 和 19.13，相差 0.33。单个任务的差值最大到 1.0（extended）。keep 门槛 0.8 约为噪声的 2.4 倍。如果你在 rerun 里看到更大的波动，就把门槛改成观察到的最大差值的 2 倍，并在 results.tsv 里加一行说明。

## 27B 环境

同一套 harness、同一份 `policy.py`，只把模型换成 Qwen3.8-27B：

- 评测：`AR_RUN=27b autoresearch/remote_run.sh`（节点 20049，服务 8 并发）。其余用 `AR_WORKERS`、`AR_MODEL` 覆盖。
- 一轮比 9B 慢约 4 倍：9B 约 10 分钟，27B 约 40 分钟（节点侧超时 90 分钟，9B 是 25 分钟）。
- dev/test 划分按实例 id、与模型无关，和 9B 完全相同，两边的分数可以直接对比。
- **停止窗口放宽到 4 小时**没有提升（27B 上 4 小时大约等于 9B 上 2 小时的实验数）。
- 起点用 9B 循环的最优策略：`git show 236bef9:autoresearch/policy.py > autoresearch/policy.py`，先测它在 27B dev 上相对 repair_only 的涨幅，确认方法能迁移，再继续搜索。
- 27B 的 3D 零样本已经到 49.8 RR，提升空间比 9B 小；boolean basic 27B 反而低于 9B（70.9 vs 80.8），是可能的机会。

## 划分说明

test 和 dev 按实例 id 划分（causal 沿用团队已有的 200 个 test id）。循环中禁止使用 test，只有上面的最终评测会用到，而且只跑一次。
