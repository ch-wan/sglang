# Training-Free Looped Transformer 实现日志

## 目标

在 SGLang 中实现论文 [Training-Free Looped Transformers](https://arxiv.org/abs/2605.23872) 的 inference-time loop，并复现 Dense model 的实验行为。实现目标是不修改模型权重，在连续的中间 decoder layers 上执行 block-anchored damped Euler refinement。

## 实现进展

### 配置接口

- 将多个 loop server 参数收敛为单个 `--loop-config` JSON 参数。
- 支持 `start`、`end`、`k`、`mode`、`strategy`、`anchor_beta`、`decode_mode` 和 `cache_strategy`。
- 保留 `loop_start`、`loop_end`、`loop_k` 作为输入别名。
- 对窗口边界、`k`、beta、策略和 decode/cache 组合做启动期校验。

### 公共插件

公共实现位于 `python/sglang/srt/models/looped_transformer.py`，模型接入通过 `LoopModelAdapter` 提供：

- initial state；
- 单层执行 callback；
- prefix/finalize traversal；
- hidden/residual stream 的统一 blend。

插件负责 block-anchored 算法、anchor blend、trial pass 的 KV policy，以及 prefill/decode 的执行策略。这样 Qwen2、Qwen3 和 Llama 不需要各自复制 loop 算法。

### 模型接入

- Qwen2 Dense：已接入。
- Qwen3 Dense：已接入，并复用 Qwen2 layer traversal。
- Llama：已接入公共 plugin；适配了 Llama attention 的 trial KV 写入控制。
- Qwen3 MoE：尚未开放，仍需要单独验证 expert routing 和 EP/DP cache 语义。

## 验证记录

### 基础验证

- loop plugin 单元测试：8 tests passed。
- Python AST、ruff、isort、ruff format、codespell 和 pre-commit passed。
- Qwen2.5-1.5B、Qwen2.5-7B、Qwen3-1.7B、Qwen3-8B 和 Llama-3.1-8B 均完成 Hopper serving smoke test。
- Qwen3-8B 和 Llama-3.1-8B 的 baseline/loop 服务均可健康启动并生成结果。

### Paper recipe 的 Qwen3-8B 外推

Paper recipe 使用：`K=6`、`anchor_beta=0.5`、`block_anchored`、`cache_strategy=first`、`decode_mode=bypass`。Qwen3-8B 使用与 Qwen3-4B 相同的四层窗口 `[15, 19)`，完成完整 16-task suite。

相对 baseline，主要 acc 变化如下：

| Task | Baseline | Looped | Delta |
| --- | ---: | ---: | ---: |
| ARC-Challenge | 0.55546075 | 0.56740614 | +0.01194539 |
| CommonsenseQA | 0.78542179 | 0.78787879 | +0.00245700 |
| MMLU elementary math | 0.69841270 | 0.71428571 | +0.01587302 |
| MMLU college math | 0.58000000 | 0.61000000 | +0.03000000 |
| PIQA acc | 0.76659412 | 0.76605005 | -0.00054407 |
| SciQ acc | 0.96700000 | 0.96600000 | -0.00100000 |
| WinoGrande | 0.67561168 | 0.68745067 | +0.01183899 |

16 个主 acc 指标中，10 个提升、3 个持平、3 个下降。该结果是 paper recipe 在 Qwen3-8B 上的外推，不是论文原始模型的严格复现。

### Llama 类 paper recipe 验证

Paper 没有 Llama 实验，因此没有原始 Llama window。对 32 层 Llama-3.1-8B 采用中间四层 `[14, 18)`，其余配置与 paper recipe 相同，并完成完整 16-task suite。

关键结果：

| Task | Baseline | Looped | Delta |
| --- | ---: | ---: | ---: |
| MMLU elementary math | 0.49206349 | 0.49735450 | +0.00529101 |
| MMLU high school physics | 0.44370861 | 0.46357616 | +0.01986755 |
| ARC-Challenge | 0.53924915 | 0.54522184 | +0.00597270 |
| PIQA acc | 0.80359086 | 0.80195865 | -0.00163221 |
| LAMBADA acc | 0.46516592 | 0.45934407 | -0.00582185 |
| WinoGrande | 0.70639305 | 0.70244672 | -0.00394633 |

结果是混合的：数学、ARC 和 physics 有提升，但 LAMBADA、formal logic 和 WinoGrande 有下降。因此当前结果只证明 Llama adapter 可运行，不能说明该窗口是 Llama 的最佳配置。

## 重要限制

- `decode_mode=bypass` 是论文 loglikelihood 评测路径；`decode_mode=full` 仍是实验性 serving 路径。
- decode-full 当前通过固定 token slot 的 KV 覆写实现，不提供完整的跨 slot stash/restore。
- loop window 不能跨 pipeline-parallel rank。
- CUDA graph、chunked prefill、prefix-cache 命中以及 MoE expert routing 仍需专项 cache invariant 测试。
- Instruct 模型通过 completion API 做 lm-eval 时没有应用 chat template，相关分数主要用于 baseline/loop 相对比较。

## 产物位置

Hopper 上的主要结果目录：

- Qwen3-8B paper-style：`/scratch/loop-results/qwen3-8b-paper-eval-30012`、`/scratch/loop-results/qwen3-8b-paper-eval-30013`
- Llama paper-style：`/scratch/loop-results/llama-8b-paper-eval-30014`、`/scratch/loop-results/llama-8b-paper-eval-30015`
- Qwen2.5-7B：`/scratch/loop-results/q25-7b-eval-30006`、`/scratch/loop-results/q25-7b-eval-30007`

## 后续工作

1. 增加 KV cache slot、prefix-cache 和 chunked-prefill correctness tests。
2. 对 Qwen3-8B 和 Llama 做 layer-window sweep，避免把单一窗口误认为模型通用最优配置。
3. 补齐 Qwen3 MoE 的 layer-mode 与 EP/DP 测试。
4. 评估固定 loop config 下的 CUDA graph 和性能优化。
5. 在合并前将 serving 参数、完整 suite 命令和结果 JSON 纳入可重复的 Hopper benchmark script。
