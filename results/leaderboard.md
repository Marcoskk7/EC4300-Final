# 榜单

由 `make report` 从 `runs/*/` 生成。VR/NR/RR/parse 单位为 %，每个任务内按 RR 降序。
只有预算相同的行才能比较。

## causal_n4

| run | 模型 | tag | 预算 | max_tokens | 实例数 | parse | VR | NR | RR | 每样本 token | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261122_qwen3.8-27b_main_zero | qwen3.8-27b | zero | x2 | 1024 | 200 | 95.7 | 28.7 | 42.0 | 20.7 | 38435 |  |
| 20261122_qwen3.5-9b_main_zero | qwen3.5-9b | zero | x2 | 1024 | 200 | 100.0 | 25.9 | 38.1 | 8.3 | 31865 |  |

## causal_n4_all

| run | 模型 | tag | 预算 | max_tokens | 实例数 | parse | VR | NR | RR | 每样本 token | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261123_qwen3.8-27b_causal_full | qwen3.8-27b |  | x2 | 1024; 187 rerun @2048 | 847 | 99.2 | 31.9 | 68.3 | 43.0 | 16005 | afb625b6 |
| 20261123_qwen3.5-9b_causal_full | qwen3.5-9b |  | x2 | 1024 | 847 | 99.9 | 24.2 | 63.7 | 22.9 | 13596 | 0367a292 |

## 3d

| run | 模型 | tag | 预算 | max_tokens | 实例数 | parse | VR | NR | RR | 每样本 token | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261122_qwen3.8-27b_main_zero | qwen3.8-27b | zero | x1 | 1024 | 129 | 99.9 | 71.7 | 69.9 | 49.8 | 27849 |  |
| 20261122_qwen3.5-9b_main_zero | qwen3.5-9b | zero | x1 | 1024 | 129 | 100.0 | 57.1 | 42.5 | 22.7 | 24316 |  |

## boolean_basic

| run | 模型 | tag | 预算 | max_tokens | 实例数 | parse | VR | NR | RR | 每样本 token | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261122_qwen3.8-27b_main_zero | qwen3.8-27b | zero | x1 | 1024 | 35 | 100.0 | 70.9 | 71.1 | 49.5 | 2610 |  |
| 20261122_qwen3.5-9b_main_zero | qwen3.5-9b | zero | x1 | 1024 | 35 | 100.0 | 80.8 | 32.7 | 26.6 | 2396 |  |

## boolean_extended

| run | 模型 | tag | 预算 | max_tokens | 实例数 | parse | VR | NR | RR | 每样本 token | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261122_qwen3.8-27b_main_zero | qwen3.8-27b | zero | x1 | 1024 | 78 | 100.0 | 50.5 | 56.4 | 36.0 | 4595 |  |
| 20261122_qwen3.5-9b_main_zero | qwen3.5-9b | zero | x1 | 1024 | 78 | 100.0 | 66.1 | 32.1 | 16.5 | 3772 |  |

## boolean_full

| run | 模型 | tag | 预算 | max_tokens | 实例数 | parse | VR | NR | RR | 每样本 token | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261122_qwen3.8-27b_main_zero | qwen3.8-27b | zero | x1 | 1024 | 80 | 100.0 | 43.3 | 37.4 | 20.0 | 9516 |  |
| 20261122_qwen3.5-9b_main_zero | qwen3.5-9b | zero | x1 | 1024 | 80 | 100.0 | 59.7 | 14.3 | 8.0 | 7598 |  |
