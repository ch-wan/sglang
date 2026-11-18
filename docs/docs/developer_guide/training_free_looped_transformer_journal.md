# Training-Free Looped Transformer：远端复现 journal

## 交接目标和已知缺口

本 journal 与 `benchmark/looped_transformer/` 是远端 agent 的完整交接入口。命令从公开 checkout 的根目录执行，资产和结果由参数指定，不依赖前一台开发机器或历史节点的目录。

区分三个目标：

1. 重跑已知模型、配置、任务集和样本数，生成新的原始结果。
2. 对照 `historical-results.json` 中记录下来的全部历史分数。
3. 精确重现旧运行的所有浮点值。

目前只能为前两项目标提供可执行流程。历史 H200 节点已不可用；原始环境 lock、模型/数据 revision、容器 digest、完整启动参数和 samples 没有归档。Qwen3-1.7B 与 Qwen2.5-1.5B 还缺少精确 loop JSON。作者原始代码仓库现返回 404，归档 patch 的 commit/哈希没有保存。这些缺口不能用猜测填补。

本次提交的资产 lock 是交接时新解析的 6 个模型和 16 个 task 的 immutable revisions，供此后的重复运行使用；它不是恢复出的历史 lock。新运行与历史分数存在偏差时，须保留偏差与版本信息。

## 来源与更正

- 论文：[Training-Free Looped Transformers v1](https://arxiv.org/html/2605.23872v1)。
- 作者仓库：[Training-Free-Looped-Transformer](https://github.com/L-z-Chen/Training-Free-Looped-Transformer)。交接更新时 API 与 Git remote 均返回 404。
- 实现基线：`2f6ceec8dfc8ab5804d5b81ff24c8b8ec65e9e74`。
- 发布分支：[cheng/dev/raining-free-looped-transformer](https://github.com/ch-wan/sglang/tree/cheng/dev/raining-free-looped-transformer)。名称按用户指定保留。
- 新 replay 固定 `lm-eval[api]==0.4.11`；[对应源码](https://github.com/EleutherAI/lm-evaluation-harness/tree/27988a293647d5853e48edea291640b4af54740c)。当时的安装版本没有恢复，不能将此值写成历史环境证据。

更正先前报告：

- 论文实际包含 **Llama-3.2-1B/3B**。我们运行的 **Llama-3.1-8B-Instruct** 属于额外扩展；“论文不包含 Llama”是错误说法。
- 当时 README 的 `K=6, beta=0.5, block_anchored, first, bypass` recipe 不是论文所有表格的统一 setting。论文附录存在不同窗口、策略、shot count 和 decode/cache 组合。
- Qwen3-8B/Llama-3.1-8B 的 suite16 是 README recipe 的模型外推实验；它们不验证论文主表中的 MMLU-Pro/GPQA，也没有复现全部论文实验。
- 只测有限窗口无法证明 Llama 当前窗口是最优或非最优；之前“不是最佳配置”的结论没有充分证据。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| experiments.json | 实验 ID、模型、suite、limit、完整 loop JSON、配置来源 |
| assets.lock.json | 模型 SHA、数据集 SHA、task YAML SHA256 |
| historical-results.json | 可恢复的历史 baseline/loop/stderr；缺失格不填造 |
| requirements-eval.txt | 新 replay 的 harness 依赖 |
| requirements-runtime.in | 从此分支 runtime spec 提取的 Linux 依赖及 replay/build 工具 |
| requirements-linux.lock.txt | Linux x86_64 / Python 3.12 的 248 个固定版本及包 SHA256 |
| reproduce.py | 准备、服务生命周期、评测调度、汇总 |
| evaluate.py | 从锁定的数据 snapshot 加载 task，调用 harness，可选作者 HF patch |

以上文件均在 `benchmark/looped_transformer/`。执行器核对 5 个实现文件相对实现基线 commit 的哈希，每次运行保存实际源码哈希、Git HEAD 与 dirty status。

## 实验矩阵

窗口是 0-based、左闭右开。已知 loop 都使用 `mode=block`、`strategy=block_anchored`、`anchor_beta=0.5`、`cache_strategy=first`。baseline 不传 `--loop-config`。

| ID | Model | Suite | Limit/task | Window | K | Decode |
| --- | --- | --- | ---: | --- | ---: | --- |
| qwen3-4b-smoke100 | Qwen/Qwen3-4B-Base | smoke3 | 100 | [15,19) | 6 | bypass |
| qwen3-4b-suite100 | Qwen/Qwen3-4B-Base | suite16 | 100 | [15,19) | 6 | bypass |
| qwen3-1p7b-suite100 | Qwen/Qwen3-1.7B-Base | extended5 | 100 | 未归档 | 未归档 | 未归档 |
| qwen25-1p5b-suite100 | Qwen/Qwen2.5-1.5B-Instruct | extended5 | 100 | 未归档 | 未归档 | 未归档 |
| qwen25-7b-suite100 | Qwen/Qwen2.5-7B-Instruct | extended5 | 100 | [12,16) | 4 | full |
| qwen3-8b-suite100 | Qwen/Qwen3-8B | extended5 | 100 | [12,16) | 4 | full |
| llama31-8b-suite100 | NousResearch/Meta-Llama-3.1-8B-Instruct | extended5 | 100 | [12,16) | 4 | full |
| qwen3-8b-suitefull | Qwen/Qwen3-8B | suite16 | 全量 | [15,19) | 6 | bypass |
| llama31-8b-suitefull | NousResearch/Meta-Llama-3.1-8B-Instruct | suite16 | 全量 | [14,18) | 6 | bypass |

`smoke3`：

```text
arc_easy, hellaswag, openbookqa
```

`extended5`：

```text
piqa, winogrande, sciq, commonsense_qa, mmlu_elementary_mathematics
```

`suite16`：

```text
arc_easy, arc_challenge, hellaswag, piqa, winogrande, lambada_openai,
sciq, openbookqa, commonsense_qa, truthfulqa_mc1,
mmlu_elementary_mathematics, mmlu_high_school_mathematics,
mmlu_high_school_physics, mmlu_college_mathematics,
mmlu_formal_logic, mmlu_abstract_algebra
```

共同评测参数：0-shot、batch size 8、单并发、seeds 0/1234/1234/1234、client max length 2048、Hugging Face tokenizer backend、`tokenized_requests=False`、completion API、不应用 chat template。保持原 task prompt。Instruct 模型的结果不应与标准 chat leaderboard 混用。

新 replay 显式指定 bfloat16、TP=1。历史启动没有完整记录 attention backend，不将新的 backend 默认值写成历史事实。历史 suite16 打印 86,438 个候选请求/组；这个值不是样本数，实际请求数可能随 harness 变化。

## 在新远端建立 worktree 和环境

推荐 Linux x86_64、Python 3.12、H100/H200 Hopper、兼容 CUDA 13 的 driver。源码安装还需要 Rust/Cargo >=1.85、C/C++ 编译器、Python development headers、pkg-config 和 OpenSSL development headers；可使用仓库 Dockerfile 提供的工具链。原运行使用 H200、TP=1。一张空闲 GPU 可以顺序运行全部配置；建议至少 150 GB 可用存储和 64 GB 主机内存。

在你选择的持久存储目录：

```bash
git clone --branch cheng/dev/raining-free-looped-transformer \
  https://github.com/ch-wan/sglang.git looped-source
cd looped-source
HANDOFF_REV="$(git rev-parse HEAD)"
git worktree add --detach ../looped-replay "$HANDOFF_REV"
cd ../looped-replay
git rev-parse HEAD
git status --short

python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --require-hashes -r benchmark/looped_transformer/requirements-linux.lock.txt
python -m pip install --no-deps --no-build-isolation -e python
pre-commit install
python -m pip check
PYTHONPATH=python python -c 'import torch, sglang; print(torch.__version__, torch.cuda.is_available(), sglang.__file__)'
```

解析一次 `HANDOFF_REV` 后保存该 SHA，后续不要追随变化的分支 HEAD。Linux lock 已完成 resolver 校验，固定 torch 2.13.0、transformers 5.12.1、FlashInfer 0.6.18、sglang-kernel 0.4.7、lm-eval 0.4.11 及传递依赖。它是新的 replay 环境，不是原运行环境。遇到 wheel、driver 或编译错误要保留输出；不要静默改版本后声称环境相同。新的 Hopper 全量安装与推理尚未重跑。

如果要有意生成新的环境 lock，保留旧文件并对照差异：

```bash
uv pip compile --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --no-header --no-annotate \
  --output-file artifacts/new-runtime.lock.txt \
  benchmark/looped_transformer/requirements-runtime.in
```

如果使用容器，从同一提交构建并记录 image ID：

```bash
docker build -f docker/Dockerfile --build-arg BRANCH_TYPE=local \
  --build-arg BUILD_TYPE=dev -t looped-transformer-replay .
docker image inspect looped-transformer-replay --format '{{.Id}}'
```

将镜像导出或发布到可访问 registry，并记录 digest。容器内要有可用的 Git metadata；不能因为 linked worktree 的 gitfile 指向容器外目录而无法校验源码，也不能导入镜像里另一份 SGLang。原镜像 digest 未恢复，新构建不能叫作历史镜像。

## 模型与数据准备

```bash
python benchmark/looped_transformer/reproduce.py list
python benchmark/looped_transformer/reproduce.py run --dry-run
python benchmark/looped_transformer/reproduce.py prepare \
  --lock benchmark/looped_transformer/assets.lock.json \
  --output artifacts/looped-transformer
```

模型用完整 SHA 下载进 Hub cache；数据按 task 保存为可携带的 snapshot，附带 lock。评测阶段用这些 snapshot，不再依赖旧节点。准备失败的半成品目录会被拒绝使用；同 lock 的完整目录可复用。

Llama 使用 NousResearch repo 是因为当时官方 Meta repo 的下载凭证没有授权，不能静默换模型。公开资产通常不需要 token；限流/授权需要 `HF_TOKEN` 时在远端环境中设置，不写入命令参数和共享归档。

只验证数据准备、不下载权重：

```bash
python benchmark/looped_transformer/reproduce.py prepare --datasets-only \
  --output artifacts/dataset-check
```

若要重新解析 revisions，写新文件：

```bash
python benchmark/looped_transformer/reproduce.py lock \
  --lock artifacts/new-assets.lock.json
```

之后 prepare/run 都必须显式使用该新 lock。改 revision 是一次新实验，不覆盖已提交的 lock。

## 执行实验

### 最小验证

```bash
PYTHONPATH=python python -m pytest -q test/manual/models/test_looped_transformer.py
PYTHONPATH=python python benchmark/looped_transformer/reproduce.py run \
  --experiments qwen3-4b-smoke100 --gpu 0 --port 30100
```

执行器启动 baseline，等待 HTTP health 200 后运行 harness，停止 baseline，再启动 loop，运行相同评测。只监听 loopback。退出和错误时清理本次启动的进程组，不按通用进程名批量 kill。

loop JSON：

```json
{"start":15,"end":19,"k":6,"mode":"block","strategy":"block_anchored","anchor_beta":0.5,"decode_mode":"bypass","cache_strategy":"first"}
```

每个实验/backend/mode 目录必须是新目录，避免覆盖历史结果。重试用新的 `--output`，或只选缺失的 `--modes baseline` / `--modes loop`。新 output 也要先 prepare。

### 全部已知配置

```bash
PYTHONPATH=python python benchmark/looped_transformer/reproduce.py run \
  --experiments qwen3-4b-smoke100 qwen3-4b-suite100 \
    qwen25-7b-suite100 qwen3-8b-suite100 llama31-8b-suite100 \
    qwen3-8b-suitefull llama31-8b-suitefull \
  --gpu 0 --port 30100
```

只跑两组全量：

```bash
PYTHONPATH=python python benchmark/looped_transformer/reproduce.py run \
  --experiments qwen3-8b-suitefull llama31-8b-suitefull \
  --gpu 0 --port 30100
```

可在两张空闲 GPU 上按不同实验 ID 并行；必须用不同端口，不得同时写同一个实验目录。全量组不传 limit，执行器核对每个 task 的 effective count 与完整 split 对齐。

### 缺失配置的两组

baseline 可独立执行：

```bash
PYTHONPATH=python python benchmark/looped_transformer/reproduce.py run \
  --experiments qwen3-1p7b-suite100 qwen25-1p5b-suite100 \
  --modes baseline --gpu 0 --port 30100
```

loop 会在启动前报出参数缺失。恢复 JSON 后可通过 `--config-overrides` 提供一个 experiment ID 到完整 loop JSON 的对象。执行器将这类结果标为 `reconstructed-override`，不冒充历史配置。默认 all 也会明确报出缺项，不能把跳过这两组称为全部完成。

### 作者 HF reference

历史 Qwen3-4B/Qwen3-1.7B 有 HF reference 分数；作者源码当前不可下载且历史哈希缺失。取得可用归档后，计算并核实文件哈希：

```bash
sha256sum author-archive/loop_qwen3.py
PYTHONPATH=python python benchmark/looped_transformer/reproduce.py run \
  --experiments qwen3-4b-smoke100 --backend paper-hf \
  --paper-source author-archive/loop_qwen3.py \
  --paper-sha256 "$AUTHOR_PATCH_SHA256" --gpu 0
```

`AUTHOR_PATCH_SHA256` 必须来自实际归档。执行器校验后调用作者的 `patch_qwen3_with_loop`，使用相同窗口/K/beta/cache/decode，HF backend 用 HFLM 与 SDPA。归档缺失、哈希不同或接口不兼容都会失败，不自动替换实现。Qwen3-1.7B 另有历史 loop 配置缺失的问题。

## 产物和自动汇总

```text
artifacts/looped-transformer/
  assets.lock.json
  datasets/<task>/...
  <experiment>/<backend>/baseline/
    environment.json
    experiment.json
    assets.lock.json
    commands.json
    status.json
    server.log
    eval.log
    dataset-fingerprints.json
    results.json
    samples-<task>.jsonl
  <experiment>/<backend>/loop/...
  summary.json
  summary.md
  historical-differences.json
```

保存 pip freeze、实际命令、Git 状态、源码哈希、GPU/driver、数据 fingerprint、逐样本预测和 running/complete/failed 状态。运行日志可能包含执行节点自己的目录信息，但那不是其他节点需要依赖的路径。

```bash
python benchmark/looped_transformer/reproduce.py summarize
```

acc、acc_norm 分别汇总；stderr 单独保存，不当作主指标统计。Markdown 8 位小数，JSON 保留计算精度。历史差值为本次分数减去历史分数，不将任意 tolerance 自动解释为显著性。

在产物根目录生成校验和，再发布归档：

```bash
cd artifacts/looped-transformer
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
tar -czf ../looped-transformer-results.tar.gz .
```

下一位执行者应将 archive 发布到可访问的对象存储或 Release，并记录实际 URL 和 SHA256。不能用某台机器的磁盘目录替代下载入口。目前没有可公开下载的历史原始 JSON；本次提供历史数值和重新生成完整产物的方法。

## 可恢复的全部历史数值

以下内容由 `historical-results.json` 生成。每一项的证据来源是本对话保留的命令输出；不是重新测量。缺少 stderr 显示为 “—”。full-suite 的 23 个 metric pair 均保留，早期 suite100 未恢复的格不补填。

### qwen3-4b-smoke100 / sglang

| Task / metric | Baseline | Loop | Delta | Baseline stderr | Loop stderr |
| --- | ---: | ---: | ---: | ---: | ---: |
| arc_easy/acc | 0.74000000 | 0.75000000 | +0.01000000 | — | — |
| arc_easy/acc_norm | 0.77000000 | 0.80000000 | +0.03000000 | — | — |
| hellaswag/acc | 0.52000000 | 0.52000000 | +0.00000000 | — | — |
| hellaswag/acc_norm | 0.71000000 | 0.70000000 | -0.01000000 | — | — |
| openbookqa/acc | 0.31000000 | 0.32000000 | +0.01000000 | — | — |
| openbookqa/acc_norm | 0.40000000 | 0.38000000 | -0.02000000 | — | — |

### qwen3-4b-smoke100 / paper-hf

Archived author patch and its historical SHA256 are unavailable.

| Task / metric | Baseline | Loop | Delta | Baseline stderr | Loop stderr |
| --- | ---: | ---: | ---: | ---: | ---: |
| arc_easy/acc | 0.74000000 | 0.76000000 | +0.02000000 | — | — |
| hellaswag/acc | 0.51000000 | 0.52000000 | +0.01000000 | — | — |
| openbookqa/acc | 0.31000000 | 0.32000000 | +0.01000000 | — | — |

### qwen3-4b-suite100 / sglang

Only these four suite100 metrics survived in the design journal. Do not substitute smoke100 numbers for the remaining cells.

| Task / metric | Baseline | Loop | Delta | Baseline stderr | Loop stderr |
| --- | ---: | ---: | ---: | ---: | ---: |
| arc_challenge/acc | 0.47000000 | 0.48000000 | +0.01000000 | — | — |
| arc_easy/acc_norm | 0.77000000 | 0.80000000 | +0.03000000 | — | — |
| hellaswag/acc_norm | 0.71000000 | 0.70000000 | -0.01000000 | — | — |
| openbookqa/acc | 0.31000000 | 0.33000000 | +0.02000000 | — | — |

### qwen3-1p7b-suite100 / sglang

Exact historical loop configuration is unavailable; loop replay fails until an explicit reconstructed override is supplied.

| Task / metric | Baseline | Loop | Delta | Baseline stderr | Loop stderr |
| --- | ---: | ---: | ---: | ---: | ---: |
| commonsense_qa/acc | 0.73000000 | 0.72000000 | -0.01000000 | — | — |
| mmlu_elementary_mathematics/acc | 0.53000000 | 0.53000000 | +0.00000000 | — | — |
| piqa/acc | 0.77000000 | 0.77000000 | +0.00000000 | — | — |
| piqa/acc_norm | 0.78000000 | 0.77000000 | -0.01000000 | — | — |
| sciq/acc | 0.98000000 | 0.98000000 | +0.00000000 | — | — |
| sciq/acc_norm | 0.98000000 | 0.98000000 | +0.00000000 | — | — |
| winogrande/acc | 0.67000000 | 0.71000000 | +0.04000000 | — | — |

### qwen3-1p7b-suite100 / paper-hf

Exact loop configuration and archived author patch SHA256 are unavailable.

| Task / metric | Baseline | Loop | Delta | Baseline stderr | Loop stderr |
| --- | ---: | ---: | ---: | ---: | ---: |
| commonsense_qa/acc | 0.74000000 | 0.72000000 | -0.02000000 | — | — |
| mmlu_elementary_mathematics/acc | 0.53000000 | 0.51000000 | -0.02000000 | — | — |
| piqa/acc | 0.77000000 | 0.77000000 | +0.00000000 | — | — |
| piqa/acc_norm | 0.80000000 | 0.81000000 | +0.01000000 | — | — |
| sciq/acc | 0.98000000 | 0.98000000 | +0.00000000 | — | — |
| sciq/acc_norm | 0.97000000 | 0.98000000 | +0.01000000 | — | — |
| winogrande/acc | 0.69000000 | 0.66000000 | -0.03000000 | — | — |

### qwen25-1p5b-suite100 / sglang

Exact historical loop configuration is unavailable.

| Task / metric | Baseline | Loop | Delta | Baseline stderr | Loop stderr |
| --- | ---: | ---: | ---: | ---: | ---: |
| commonsense_qa/acc | 0.77000000 | 0.77000000 | +0.00000000 | — | — |
| mmlu_elementary_mathematics/acc | 0.47000000 | 0.47000000 | +0.00000000 | — | — |
| piqa/acc | 0.77000000 | 0.79000000 | +0.02000000 | — | — |
| piqa/acc_norm | 0.77000000 | 0.77000000 | +0.00000000 | — | — |
| sciq/acc | 0.95000000 | 0.96000000 | +0.01000000 | — | — |
| sciq/acc_norm | 0.96000000 | 0.97000000 | +0.01000000 | — | — |
| winogrande/acc | 0.72000000 | 0.71000000 | -0.01000000 | — | — |

### qwen25-7b-suite100 / sglang

| Task / metric | Baseline | Loop | Delta | Baseline stderr | Loop stderr |
| --- | ---: | ---: | ---: | ---: | ---: |
| commonsense_qa/acc | 0.82000000 | 0.82000000 | +0.00000000 | 0.03861229 | 0.03861229 |
| mmlu_elementary_mathematics/acc | 0.69000000 | 0.68000000 | -0.01000000 | 0.04648232 | 0.04688262 |
| piqa/acc | 0.83000000 | 0.84000000 | +0.01000000 | 0.03775252 | 0.03684529 |
| piqa/acc_norm | 0.84000000 | 0.86000000 | +0.02000000 | 0.03684529 | 0.03487351 |
| sciq/acc | 0.97000000 | 0.98000000 | +0.01000000 | 0.01714466 | 0.01407053 |
| sciq/acc_norm | 0.99000000 | 1.00000000 | +0.01000000 | 0.01000000 | 0.00000000 |
| winogrande/acc | 0.71000000 | 0.72000000 | +0.01000000 | 0.04560480 | 0.04512609 |

### qwen3-8b-suite100 / sglang

| Task / metric | Baseline | Loop | Delta | Baseline stderr | Loop stderr |
| --- | ---: | ---: | ---: | ---: | ---: |
| commonsense_qa/acc | 0.76000000 | 0.84000000 | +0.08000000 | 0.04292347 | 0.03684529 |
| mmlu_elementary_mathematics/acc | 0.65000000 | 0.68000000 | +0.03000000 | 0.04793725 | 0.04688262 |
| piqa/acc | 0.82000000 | 0.81000000 | -0.01000000 | 0.03861229 | 0.03942772 |
| piqa/acc_norm | 0.83000000 | 0.84000000 | +0.01000000 | 0.03775252 | 0.03684529 |
| sciq/acc | 0.99000000 | 0.99000000 | +0.00000000 | 0.01000000 | 0.01000000 |
| sciq/acc_norm | 0.99000000 | 0.99000000 | +0.00000000 | 0.01000000 | 0.01000000 |
| winogrande/acc | 0.78000000 | 0.79000000 | +0.01000000 | 0.04163332 | 0.04093602 |

### llama31-8b-suite100 / sglang

| Task / metric | Baseline | Loop | Delta | Baseline stderr | Loop stderr |
| --- | ---: | ---: | ---: | ---: | ---: |
| commonsense_qa/acc | 0.76000000 | 0.75000000 | -0.01000000 | 0.04292347 | 0.04351941 |
| mmlu_elementary_mathematics/acc | 0.47000000 | 0.43000000 | -0.04000000 | 0.05016136 | 0.04975699 |
| piqa/acc | 0.78000000 | 0.78000000 | +0.00000000 | 0.04163332 | 0.04163332 |
| piqa/acc_norm | 0.82000000 | 0.81000000 | -0.01000000 | 0.03861229 | 0.03942772 |
| sciq/acc | 0.98000000 | 0.99000000 | +0.01000000 | 0.01407053 | 0.01000000 |
| sciq/acc_norm | 0.97000000 | 0.98000000 | +0.01000000 | 0.01714466 | 0.01407053 |
| winogrande/acc | 0.70000000 | 0.70000000 | +0.00000000 | 0.04605662 | 0.04605662 |

### qwen3-8b-suitefull / sglang

| Task / metric | Baseline | Loop | Delta | Baseline stderr | Loop stderr |
| --- | ---: | ---: | ---: | ---: | ---: |
| arc_challenge/acc | 0.55546075 | 0.56740614 | +0.01194539 | 0.01452123 | 0.01447801 |
| arc_challenge/acc_norm | 0.56484642 | 0.57764505 | +0.01279863 | 0.01448799 | 0.01443414 |
| arc_easy/acc | 0.83417508 | 0.83838384 | +0.00420876 | 0.00763171 | 0.00755322 |
| arc_easy/acc_norm | 0.80765993 | 0.81691919 | +0.00925926 | 0.00808756 | 0.00793559 |
| commonsense_qa/acc | 0.78542179 | 0.78787879 | +0.00245700 | 0.01175342 | 0.01170420 |
| hellaswag/acc | 0.57179845 | 0.57657837 | +0.00477992 | 0.00493807 | 0.00493091 |
| hellaswag/acc_norm | 0.74875523 | 0.75512846 | +0.00637323 | 0.00432843 | 0.00429132 |
| lambada_openai/acc | 0.65204735 | 0.64913643 | -0.00291092 | 0.00663608 | 0.00664889 |
| lambada_openai/perplexity | 4.59398247 | 4.56964915 | -0.02433332 | 0.13869483 | 0.13856871 |
| mmlu_abstract_algebra/acc | 0.58000000 | 0.59000000 | +0.01000000 | 0.04960450 | 0.04943111 |
| mmlu_college_mathematics/acc | 0.58000000 | 0.61000000 | +0.03000000 | 0.04960450 | 0.04902071 |
| mmlu_elementary_mathematics/acc | 0.69841270 | 0.71428571 | +0.01587301 | 0.02363698 | 0.02326651 |
| mmlu_formal_logic/acc | 0.61111111 | 0.62698413 | +0.01587302 | 0.04360315 | 0.04325506 |
| mmlu_high_school_mathematics/acc | 0.51111111 | 0.51111111 | +0.00000000 | 0.03047801 | 0.03047801 |
| mmlu_high_school_physics/acc | 0.70198675 | 0.70198675 | +0.00000000 | 0.03734536 | 0.03734536 |
| openbookqa/acc | 0.31200000 | 0.31800000 | +0.00600000 | 0.02074060 | 0.02084757 |
| openbookqa/acc_norm | 0.42200000 | 0.42400000 | +0.00200000 | 0.02210904 | 0.02212299 |
| piqa/acc | 0.76659412 | 0.76605005 | -0.00054407 | 0.00986925 | 0.00987724 |
| piqa/acc_norm | 0.77638738 | 0.77801959 | +0.00163221 | 0.00972149 | 0.00969612 |
| sciq/acc | 0.96700000 | 0.96600000 | -0.00100000 | 0.00565181 | 0.00573384 |
| sciq/acc_norm | 0.95800000 | 0.96200000 | +0.00400000 | 0.00634636 | 0.00604918 |
| truthfulqa_mc1/acc | 0.36597307 | 0.36597307 | +0.00000000 | 0.01686294 | 0.01686294 |
| winogrande/acc | 0.67561168 | 0.68745067 | +0.01183899 | 0.01315723 | 0.01302756 |

### llama31-8b-suitefull / sglang

| Task / metric | Baseline | Loop | Delta | Baseline stderr | Loop stderr |
| --- | ---: | ---: | ---: | ---: | ---: |
| arc_challenge/acc | 0.53924915 | 0.54522184 | +0.00597269 | 0.01456630 | 0.01455151 |
| arc_challenge/acc_norm | 0.55716724 | 0.56228669 | +0.00511945 | 0.01451557 | 0.01449757 |
| arc_easy/acc | 0.82323232 | 0.82786195 | +0.00462963 | 0.00782764 | 0.00774614 |
| arc_easy/acc_norm | 0.79755892 | 0.80555556 | +0.00799664 | 0.00824516 | 0.00812108 |
| commonsense_qa/acc | 0.76248976 | 0.76167076 | -0.00081900 | 0.01218367 | 0.01219811 |
| hellaswag/acc | 0.59759012 | 0.59858594 | +0.00099582 | 0.00489381 | 0.00489183 |
| hellaswag/acc_norm | 0.78968333 | 0.79047998 | +0.00079665 | 0.00406701 | 0.00406134 |
| lambada_openai/acc | 0.46516592 | 0.45934407 | -0.00582185 | 0.00694905 | 0.00694291 |
| lambada_openai/perplexity | 12.79514099 | 13.28141364 | +0.48627265 | 0.41178592 | 0.42779661 |
| mmlu_abstract_algebra/acc | 0.39000000 | 0.39000000 | +0.00000000 | 0.04902071 | 0.04902071 |
| mmlu_college_mathematics/acc | 0.32000000 | 0.33000000 | +0.01000000 | 0.04688262 | 0.04725816 |
| mmlu_elementary_mathematics/acc | 0.49206349 | 0.49735450 | +0.00529101 | 0.02574807 | 0.02575095 |
| mmlu_formal_logic/acc | 0.48412698 | 0.47619048 | -0.00793650 | 0.04469882 | 0.04467063 |
| mmlu_high_school_mathematics/acc | 0.41851852 | 0.42592593 | +0.00740741 | 0.03007801 | 0.03014914 |
| mmlu_high_school_physics/acc | 0.44370861 | 0.46357616 | +0.01986755 | 0.04056528 | 0.04071636 |
| openbookqa/acc | 0.35400000 | 0.35800000 | +0.00400000 | 0.02140758 | 0.02146143 |
| openbookqa/acc_norm | 0.46000000 | 0.46400000 | +0.00400000 | 0.02231133 | 0.02232498 |
| piqa/acc | 0.80359086 | 0.80195865 | -0.00163221 | 0.00926923 | 0.00929821 |
| piqa/acc_norm | 0.81284004 | 0.81338411 | +0.00054407 | 0.00910027 | 0.00909008 |
| sciq/acc | 0.97400000 | 0.97300000 | -0.00100000 | 0.00503481 | 0.00512809 |
| sciq/acc_norm | 0.95500000 | 0.95600000 | +0.00100000 | 0.00655881 | 0.00648892 |
| truthfulqa_mc1/acc | 0.37943696 | 0.38066095 | +0.00122399 | 0.01698704 | 0.01699763 |
| winogrande/acc | 0.70639305 | 0.70244672 | -0.00394633 | 0.01279940 | 0.01284909 |

Delta 从已四舍五入的历史分数计算，最后一位可能与当时直接从原 JSON 算出的差值不同。Qwen3-8B full 的 16 个主 acc 为 10 升、3 平、3 降；这是描述性统计，不是显著性检验。

## 实现语义、已知风险与验收

公共 plugin 位于 `python/sglang/srt/models/looped_transformer.py`。Qwen2/Qwen3/Llama adapter 提供 initial state、单层 callback 和 finalize；blend 作用于 `hidden_states + residual`。

prefill 中 natural pass 正常写 KV，K-1 次 trial pass 禁止写 KV，最后按 beta 混合 stream。没有额外 canonical 写回 pass。decode-full 允许 trial 覆写当前 token slot；它尚未证明与作者的 snapshot/restore 等价。bypass 不在 decode 中执行 loop。

仍需独立验证 prefix cache、chunked prefill、batch decode、TP/DP/PP correctness、固定配置 CUDA graph、MoE 路由和 HF 的逐样本 loglikelihood 一致性。健康检查、短文本生成和 aggregate acc 接近不能代替这些测试。

远端 agent 完成后必须交付：

1. 本次源码 SHA、环境版本、GPU/driver、模型和数据 revision、真实命令。
2. baseline/loop 均为 complete；effective sample count 与 limit/完整 split 一致。
3. 所有 results、samples、失败日志、stderr、历史差值和 SHA256 校验清单。
4. 可下载的 artifact URL；另一台节点能取得并核验，不依赖当前节点目录。
5. 逐项解释模型 revision、prompt、数值 kernel 或配置重建造成的偏差。
6. 对缺失历史 JSON/环境/作者归档的部分保留缺口说明，不把它们标为精确复现成功。

本次更新实际验证了 Linux 248-package 环境的依赖解析、harness 导入、6 个模型与 16 个 task revision 解析、16 个数据 snapshot 的全新下载和保存、真实 harness/worker 对 16 个 task 的 48 条逐样本输出、失败结果排除、零分/零 stderr 保留、缺失配置拒绝和所启动进程的清理。模型边界在工具检查中使用 dummy 输出，这些不是新的模型精度结果。没有重新运行 Hopper 全量推理。历史 “8 tests passed” 属于此前实现测试，不是新的 GPU 验证结果。
