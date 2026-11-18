# Training-Free Looped Transformer：SGLang 实现设计

## 1. 目标与范围

在不修改模型权重的前提下，在推理时重复执行一段连续的中间 Transformer layers。当前支持 Qwen2/Qwen3/Llama Dense 的 prefill loop，以及实验性的 decode-full loop，默认使用论文推荐的 block-anchored damped Euler：

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

执行 loop 时校验窗口、`K`、策略和 cache/decode 约束；`k=1` 时退化为普通
forward，并且第一阶段的窗口必须完整位于当前 PP rank。

当前模型接入没有统一强制关闭 CUDA graph；固定配置下的 graph 数值一致性和 cache 语义仍需专项验证。

## 3. 模型接入点

Qwen3 当前复用 `Qwen2Model` 的 layer traversal，Qwen3 MoE 也采用相同的 decoder-layer 接口。公共 loop plugin 位于：

```text
python/sglang/srt/models/looped_transformer.py
```

`TrainingFreeLoopPlugin` 负责：

- loop 配置与连续窗口校验；
- 当前支持的 block-mode 迭代；
- anchor blend；
- loop trial pass 的 cache policy；
- 实验性 decode-full 标记与固定 token slot 的 KV 覆写；尚未实现 cache stash/restore。

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
2. loop window 的 natural pass 正常写 cache；
3. 后续 trial pass 禁止写 cache，最终 stream 按 beta 混合；
4. post-loop layers 正常执行。

### Decode full（当前实验实现）

1. natural pass 写入当前 token 的固定 cache slot；
2. trial pass 读取历史 KV 并允许覆写当前 slot；
3. loop 后执行 anchor blend，再运行 post-loop layers；
4. 尚未提供原始作者实现的 snapshot/restore 等价性验证。

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

## 10. 可移交的复现入口

完整实验矩阵、模型与数据 revision、执行脚本、全部可恢复分数及已知缺口见
[远端复现 journal](training_free_looped_transformer_journal.md)。

复现工具位于 `benchmark/looped_transformer/`，输出目录由参数指定。
历史原始结果与环境没有完整归档；不能仅凭 aggregate acc 接近就宣称
精确复现了论文或所有历史实验。
