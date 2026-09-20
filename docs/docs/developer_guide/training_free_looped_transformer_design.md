# Training-Free Looped Transformer：SGLang 实现设计

## 1. 目标与范围

在不修改模型权重的前提下，在推理时重复执行一段连续的中间 Transformer layers。当前支持 Qwen2/Qwen3 Dense 的 prefill loop，以及实验性的 decode-full loop，默认使用论文推荐的 block-anchored damped Euler：

```text
natural = g(x0)
h = x0 + (natural - x0) / K
repeat K - 1 times:
    y = g(h)
    h = h + (y - h) / K
output = beta * natural + (1 - beta) * h
```

当前明确不支持：跨 pipeline-parallel rank 的 loop window、CUDA graph 动态 loop、非连续 layer window，以及除 `block_anchored` 外的高级迭代策略。decode-full 会在同一个当前 token cache slot 上覆写 trial KV，暂不支持跨 slot 的 cache stash/restore。

## 2. 用户接口

模型运行参数收敛为一个 JSON 配置：

```text
--loop-config JSON
```

例如：

```json
{"start":15,"end":19,"k":6,"mode":"block","strategy":"block_anchored","anchor_beta":0.5,"decode_mode":"bypass","cache_strategy":"first"}
```

配置解析阶段校验窗口、`K`、策略和 cache/decode 约束；`k=1` 时退化为普通
forward，并且第一阶段的窗口必须完整位于当前 PP rank。

loop 开启时默认关闭或绕过不支持动态控制流的 CUDA graph；后续再增加固定配置下的 graph capture。

## 3. 模型接入点

Qwen3 当前复用 `Qwen2Model` 的 layer traversal，Qwen3 MoE 也采用相同的 decoder-layer 接口。公共 loop plugin 位于：

```text
python/sglang/srt/models/looped_transformer.py
```

`TrainingFreeLoopPlugin` 负责：

- loop 配置与连续窗口校验；
- block-mode 和 layer-mode 的迭代；
- anchor blend；
- loop trial pass 的 cache policy；
- decode 状态与 cache stash policy。

模型适配只需构造 `LoopModelAdapter`，提供 initial state、单层执行和 finalize
三个 callback；因此 Qwen3、Llama 或其他 decoder-only 模型不需要复制 loop
算法。模型 attention 只需识别统一的
`forward_batch._loop_disable_kv_cache` 上下文标记来抑制 trial pass 的 KV 写入。

## 4. hidden/residual 状态

SGLang 的 decoder layer 返回 `(hidden_states, residual)`，而论文公式作用于完整 residual stream。不能只对 `hidden_states` 做插值。

设计一个逻辑上的 loop state：

```python
LoopState(hidden_states, residual)
```

每次 layer 或 block 执行都返回新的 `LoopState`。blend 必须保持下一层看到的完整 stream 等于：

```python
stream = hidden_states + residual
```

实现时优先复用既有 residual layout，避免破坏 LayerCommunicator 的 TP/DP/EP scatter 约定。需要单独覆盖以下情况：首层 `residual is None`、普通 residual fusion、MoE reduce-scatter，以及 pipeline proxy tensors。

## 5. KV cache 语义

loop trial pass 不能重复产生 KV cache entry。

### Prefill

1. pre-loop layers 正常执行并写入 cache；
2. loop window 的 natural pass 和后续 trial pass 禁止写 cache；
3. loop 完成后执行一次 canonical loop-window pass，只写入每层一次 KV；
4. post-loop layers 正常执行。

### Decode full（当前实验实现）

1. 每个 loop layer 记录进入 loop 前的 cache 长度；
2. trial pass 可以读取历史 KV，但禁止保留本次写入；
3. 每次 trial 后恢复该 layer 的 cache 长度；
4. loop 完成后执行一次 stash pass，写入 canonical KV；
5. post-loop layers 正常执行。

当前实现通过统一 loop context 控制 attention：prefill trial pass 禁止写 KV，decode-full trial pass 覆写当前 token 的固定 cache slot。后续仍需增加 cache-slot 不变量和 prefix-cache 命中场景测试。

## 6. Dense 与 MoE 策略

第一阶段：

- Dense：`block` + `block_anchored`；
- MoE：暂不开放，避免专家路由和 cache 语义同时引入。

第二阶段：

- MoE 默认 `layer` mode；
- 每层连续执行 K 次迭代，再进入下一层；
- 验证 EP/DP 下专家路由、通信和 residual layout。

论文和原始实现显示，MoE 使用 block-mode 容易造成跨迭代的 expert routing thrashing，因此不应把 Dense 默认策略直接复用给 MoE。

## 7. 测试计划

### 单元测试

- `K=1` 与普通 layer traversal 一致；
- 非连续 window、越界 window、非法 K/beta 被拒绝；
- damped Euler 和 anchor blend 的小 tensor 公式测试；
- hidden/residual blend 保持完整 stream；
- loop disabled 时无额外分支行为。

### 模型 correctness

- Qwen3 Dense baseline 与 `K=1` logits 对比；
- prefill loop 的 deterministic logits；
- decode bypass 与 baseline 生成结果一致；
- loop trial 不导致 KV cache 长度增长；
- canonical pass 每个 loop layer 只写一个 cache entry。

### 并行与性能

- TP、DP attention、PP 单 rank window；
- chunked prefill；
- CUDA graph 被正确绕过；
- loop 开启后的额外 latency、显存和吞吐；
- 不同 batch size 与 prefix-cache 命中率。

## 8. 分阶段交付

### PR 1：公共 plugin 与 Qwen2/Qwen3 Dense

实现配置、参数校验、`block_anchored`、Qwen2/Qwen3 adapter、基础测试和 decode-full experimental path。

### PR 2：KV cache 与 serving correctness

补齐 prefix-cache、chunked prefill、batch decode 的 cache invariant 测试，并评估 decode-full 的 slot 覆写策略。

### PR 3：Qwen3 MoE

加入 layer-mode、EP/DP 测试和 MoE 性能基线。

### PR 4：性能优化

固定配置 CUDA graph、chunked prefill、PP 支持及更多模型族。

## 9. 当前决策

- 不直接移植原仓库的 monkey-patch forward；
- 不在第一版开放全部数值迭代策略；
- decode-full 目前仍需显式配置，不默认对 decode token 做 K 倍 loop；
- 不允许 loop window 跨 PP rank；
- 先保证 cache 和 residual 语义正确，再做性能优化。

## 10. Dense Qwen3 复现命令

在 Hopper/H200 上，使用 `Qwen3-4B-Base`、loop window `[15, 19)`、`K=6`、
`beta=0.5`，启动一个普通服务和一个 loop 服务：

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path /scratch/models/Qwen3-4B-Base --tp-size 1 --port 30000

PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model-path /scratch/models/Qwen3-4B-Base --tp-size 1 --port 30001 \
  --loop-config '{"start":15,"end":19,"k":6,"mode":"block",\
"strategy":"block_anchored","anchor_beta":0.5,\
"decode_mode":"bypass","cache_strategy":"first"}'
```

服务 ready 后，对两个端口分别运行相同的 lm-eval 命令（这里只用 `limit=100`
作为快速 smoke test）：

```bash
lm_eval --model local-completions \
  --tasks arc_easy,hellaswag,openbookqa \
  --model_args model=/scratch/models/Qwen3-4B-Base,\
base_url=http://127.0.0.1:${PORT}/v1/completions,\
num_concurrent=8,max_retries=3,tokenized_requests=false \
  --gen_kwargs '{"temperature":0,"max_gen_toks":16}' \
  --limit 100 --batch_size 8
```

已在 Hopper/H200 实测：baseline 的三项 `acc` 为 `0.74 / 0.52 / 0.31`，
loop 的三项 `acc` 为 `0.75 / 0.52 / 0.32`；对应 `acc_norm` 为
`0.77 / 0.71 / 0.40` 和 `0.80 / 0.70 / 0.38`。原始 Paper 的 HF 评测在同一
模型、同一任务和同一 limit 下为 baseline `0.74 / 0.51 / 0.31`、loop
`0.76 / 0.52 / 0.32`，因此 SGLang 已达到 Dense smoke-test 的同量级结果。

另外按 Paper 示例扩展到 16 个 task（每个 task 100 条）后，baseline/loop
的关键结果仍保持一致量级：`arc_challenge acc=0.47/0.48`、
`arc_easy acc_norm=0.77/0.80`、`hellaswag acc_norm=0.71/0.70`、
`openbookqa acc=0.31/0.33`，其余任务也未出现异常退化。完整 JSON 产物位于
Hopper 的 `/scratch/loop-results/full-suite-30000/` 和
`/scratch/loop-results/full-suite-30001/`。
