# DeepSeek V4.1 Flash on Strata：三卡部署测试 / Three-GPU benchmark notes

[中文](#中文) | [English](#english)

## 中文

记录一台三 NVIDIA 显卡服务器于 2026-11-08 测试 DeepSeek-V4.1-Flash（ISTA-DASLab 2.25 bpw / IQ2_XS 完整专家量化）的历史数据，包括约 256K 长上下文、预填充、DSpark 和工具调用。数据来自同一机器，不代表其他硬件或任务；部署现已停止，不能当作当前服务状态。

### 环境与数据说明

| 项目 | 已知信息 |
|---|---|
| CPU / 内存 | AMD EPYC 7K62，48 核 96 线程；系统可见内存约 91 GiB |
| 显卡 | 2 × RTX 3090 24 GiB，1 × RTX 4090 D 48 GiB |
| 系统 | Linux；NVIDIA 驱动 595.99.02（后续同机核验值） |
| 模型 | DeepSeek-V4.1-Flash，ISTA-DASLab 2.25 bpw / IQ2_XS，目标为完整 6 专家方案 |
| 上下文 | 262,144 token（输入和输出共享） |
| 历史引擎 | 0.1.55 与 0.1.58；0.1.58 切换检查另有记录。各个历史文件的完整运行配置未全部保存 |
| 已保存的早期配置 | `--dspark on --dspark-conf 0.65 --tp 2 --ssd-cache 80`，三张 GPU；这是早期配置快照，并非所有测试均已证明使用完全相同参数 |

[逐条脱敏记录](data/measurements.jsonl)含 24 个来源文件的 109 条记录，保留成功、启动和错误状态、使用量、timings、缓存及可确认的完成状态。原始提示、模型回答、工具参数、请求 ID、密钥和机器网络信息未发布。来源文件名是历史标签，不等于完整配置或独立验证。

### 256K 与 32K 预填充

以下每行是**一次**实际请求，构造重复文本并检查三个标记。预填充速率使用引擎报告的 `prompt_per_second`，生成速度使用 `predicted_per_second`，墙钟为客户端计时。自然结束不等于对复杂推理的质量验证；28 个输出 token 的生成速度容易波动。

| 来源 | 输入 token | 已缓存 | 预填充 token/s | 预填充秒 | 生成 token/s | 墙钟秒 | 结束 / 标记 |
|---|---:|---:|---:|---:|---:|---:|---|
| `long-32k.jsonl` | 30,056 | 0 | 1,051.5 | 28.58 | 36.7 | 29.42 | stop / 3/3 |
| `long-32k-chunk4096-cold.jsonl` | 30,060 | 0 | 1,039.5 | 28.92 | 74.3 | 29.37 | stop / 3/3 |
| `long-32k-cache45.jsonl` | 30,058 | 0 | 1,054.0 | 28.52 | 25.7 | 29.68 | stop / 3/3 |
| `long-32k-chunk4096.jsonl` | 30,056 | 30,055 | 7.2 | 0.14 | 46.0 | 0.82 | stop / 3/3 |
| `long-256k.jsonl` | 259,556 | 0 | 1,227.2 | 211.50 | 47.0 | 212.60 | stop / 3/3 |
| `engine-compare/old-long.jsonl` | 259,560 | 0 | 1,254.6 | 206.89 | 34.8 | 208.24 | stop / 3/3 |
| `engine-compare/new-long.jsonl` | 259,560 | 0 | 1,248.6 | 207.87 | 34.3 | 209.16 | stop / 3/3 |

缓存行实际只预填充了 **1 个新 token**，不能把 7.2 token/s 与冷预填充速率比较。old/new 文件各只有一次近 256K 请求、输出很短；历史记录不足以据此判断哪个引擎普遍更快。

### DSpark 对比：短提示

两道中文任务各重复两次，温度 0、最多生成 384 token；每组 4 次，其中各任务第二次命中了前缀缓存。窗口实验的固定顺序为 5 → 3 → 7；置信度实验固定窗口 5，顺序为 0.65 → 0.5 → 0.8。未交替随机化，因此不应把平均值差异当作确定的性能优势。接受率为接受草稿总数 / 草稿总数，并非答案正确率。

| 设置 | 生成 token/s，每次 | 平均 | 接受 / 草稿 | 接受率 |
|---|---|---:|---:|---:|
| 窗口 3 | 37.7、52.6、38.8、50.3 | 44.9 | 799 / 890 | 89.8% |
| 窗口 5 | 37.6、53.8、37.9、53.3 | 45.7 | 841 / 979 | 85.9% |
| 窗口 7 | 38.3、50.8、38.3、50.9 | 44.6 | 818 / 966 | 84.7% |
| 窗口 5、置信度 0.50 | 36.6、53.9、39.1、53.4 | 45.8 | 926 / 1,178 | 78.6% |
| 窗口 5、置信度 0.65 | 37.7、51.6、39.6、51.3 | 45.1 | 826 / 975 | 84.7% |
| 窗口 5、置信度 0.80 | 36.7、51.9、38.0、49.5 | 44.0 | 704 / 757 | 93.0% |

前 3 行和后 3 行来自不同批次，不能视为同一实验。窗口 5 的短测试均值较高，但差距小于单次波动，不能证明它在所有上下文、采样配置或质量任务上最优。

### 工具调用与多轮

历史记录中存在 `tool_calls` 响应、工具结果续写以及三轮对话记录；多个对应文件的最后一轮包含测试代号和计算后的预算。该记录可以说明这些请求有返回，不能替代对工具参数、复杂工具链或长期稳定性的完整评测。`soak.jsonl` 保留 11 轮 usage、缓存和耗时，但原始回答已脱敏移除，不能据此单独复核语义正确性。

### 复测

先自行部署可访问的兼容 API。这个仓库不包含模型权重、安装器或可直接启动的服务配置；复现原始硬件调度需另行准备并核实。`STRATA_CONFIG_LABEL` 等人为标识不等于运行状态验证。长测试消耗较多内存和时间；启动新负载前先确认服务空闲。

```bash
git clone https://github.com/blakeccb/strata-deepseek-flash-benchmarks.git
cd strata-deepseek-flash-benchmarks
export STRATA_BASE_URL='http://127.0.0.1:8096'
export STRATA_MODEL='deepseek-v4.1-flash'
export STRATA_API_KEY='<your key>'
python3 scripts/long_test.py 17300 unique-run-01
python3 scripts/summarize.py
```

请从安全存储读取密钥，避免把实际值写入 shell 历史或提交到 Git。每次用独立标签并检查 `cached_tokens`；最好交替运行 A/B/B/A、多次重复，记录实际引擎版本、启动参数、温度、失败及预填充/生成/墙钟时间。脚本生成的是合成文本，不能证明任意真实文档可达到同等表现。

---

## English

This repository records historical DeepSeek-V4.1-Flash tests on a three-NVIDIA-GPU machine on 2026-11-08, including near-256K prefill, DSpark tuning and tool/multi-turn requests. The deployment is no longer running; these are not current service metrics.

### Environment and provenance

| Item | Known configuration |
|---|---|
| CPU / RAM | AMD EPYC 7K62, 48 cores / 96 threads; about 91 GiB RAM visible to OS |
| GPUs | 2 × RTX 3090 24 GiB and 1 × RTX 4090 D 48 GiB |
| System | Linux; NVIDIA driver 595.99.02 (verified later on the same host) |
| Model | DeepSeek-V4.1-Flash, ISTA-DASLab 2.25 bpw / IQ2_XS, targeting the full six-expert setup |
| Context | 262,144 tokens shared by input and output |
| Historical engines | 0.1.55 and 0.1.58, with a separate 0.1.58 cutover check; full runtime settings were not retained for every record |
| Saved earlier configuration | `--dspark on --dspark-conf 0.65 --tp 2 --ssd-cache 80`, using three GPUs. This snapshot does not prove every experiment used identical settings |

[Sanitized records](data/measurements.jsonl) contain 109 records from 24 source files, including start/error states, usage, timing, cache and available finish status. Prompts, responses, tool arguments, request IDs, keys and network identifiers are omitted. Source filenames are historical labels, not complete configuration evidence.

### Long-context prefill

Each row is one request using repeated synthetic text and three marker checks. Prefill and decode rates are engine-reported; wall time is client-measured. The very short 28-token completion is a poor basis for stable decode comparisons.

| Source | Input | Cached | Prefill token/s | Prefill s | Decode token/s | Wall s | Finish / markers |
|---|---:|---:|---:|---:|---:|---:|---|
| `long-32k.jsonl` | 30,056 | 0 | 1,051.5 | 28.58 | 36.7 | 29.42 | stop / 3/3 |
| `long-32k-chunk4096-cold.jsonl` | 30,060 | 0 | 1,039.5 | 28.92 | 74.3 | 29.37 | stop / 3/3 |
| `long-32k-cache45.jsonl` | 30,058 | 0 | 1,054.0 | 28.52 | 25.7 | 29.68 | stop / 3/3 |
| `long-32k-chunk4096.jsonl` | 30,056 | 30,055 | 7.2 | 0.14 | 46.0 | 0.82 | stop / 3/3 |
| `long-256k.jsonl` | 259,556 | 0 | 1,227.2 | 211.50 | 47.0 | 212.60 | stop / 3/3 |
| `engine-compare/old-long.jsonl` | 259,560 | 0 | 1,254.6 | 206.89 | 34.8 | 208.24 | stop / 3/3 |
| `engine-compare/new-long.jsonl` | 259,560 | 0 | 1,248.6 | 207.87 | 34.3 | 209.16 | stop / 3/3 |

The cached row processed **one new prompt token**, so its 7.2 token/s must not be compared with cold prefill. The old/new near-256K samples are one run each with very short answers and do not establish a general engine ranking.

### DSpark: short prompts

Two Chinese tasks were each repeated twice at temperature 0 with a 384-token output cap; the second repeat of each task hit the prefix cache. Window order was 5 → 3 → 7, while the confidence experiment held the window at 5 and ran 0.65 → 0.5 → 0.8. These are not randomized or interleaved. Acceptance is accepted drafts divided by all drafted tokens, not answer accuracy.

| Setting | Individual decode token/s | Mean | Accepted / drafted | Acceptance |
|---|---|---:|---:|---:|
| Window 3 | 37.7, 52.6, 38.8, 50.3 | 44.9 | 799 / 890 | 89.8% |
| Window 5 | 37.6, 53.8, 37.9, 53.3 | 45.7 | 841 / 979 | 85.9% |
| Window 7 | 38.3, 50.8, 38.3, 50.9 | 44.6 | 818 / 966 | 84.7% |
| Window 5, confidence 0.50 | 36.6, 53.9, 39.1, 53.4 | 45.8 | 926 / 1,178 | 78.6% |
| Window 5, confidence 0.65 | 37.7, 51.6, 39.6, 51.3 | 45.1 | 826 / 975 | 84.7% |
| Window 5, confidence 0.80 | 36.7, 51.9, 38.0, 49.5 | 44.0 | 704 / 757 | 93.0% |

The first and second groups are separate experiments. Differences are smaller than run-to-run variation and do not establish a universally optimal setting.

### Tools and multi-turn

Historical files contain `tool_calls` responses, continuations after tool results, and three-turn conversations; several final turns contained the requested project code and calculated budget. These observations do not establish correct tool arguments, complex agent workflows, or sustained stability. `soak.jsonl` contains 11 turns of sanitized usage, cache and elapsed metrics, but the responses were removed, so their semantic correctness cannot be independently checked from this repository.

### Reproduce

Set up a compatible API separately. No model weights, installer or ready-to-run service configuration are included; recreating the original placement requires additional preparation and verification. A manual configuration label is not proof of actual runtime settings. Check the server is idle before running a heavy long-context workload.

```bash
git clone https://github.com/blakeccb/strata-deepseek-flash-benchmarks.git
cd strata-deepseek-flash-benchmarks
export STRATA_BASE_URL='http://127.0.0.1:8096'
export STRATA_MODEL='deepseek-v4.1-flash'
export STRATA_API_KEY='<your key>'
python3 scripts/long_test.py 17300 unique-run-01
python3 scripts/summarize.py
```

Read the real key from secure storage rather than writing it into shell history or Git. Use unique tags, inspect `cached_tokens`, interleave A/B/B/A runs, and record actual engine/configuration, temperature, failures, prefill, decode and wall time. This synthetic prompt does not establish performance on arbitrary real documents.

Independent of the upstream [Strata project](https://github.com/Niko1221/Strata). No license has been declared for this repository.
