# 教案模板 v2（TEMPLATE）

> **读者画像：刚毕业的大学生**。有基础编程概念，没接触过 LLM infra。每个名词第一次出现必须当场用一句白话讲清；每个机制从"它是什么 → 为什么需要 → 怎么工作 → 数字 → 什么时候用"完整串一遍，不允许只丢结论。
>
> 原版（miniinfra README）回答"这个优化在受控条件下值多少"；教案回答"它是什么、为什么存在、在产品里长什么样、在方案里什么时候用、怎么讲给外行、学完你会什么"。教案不复制原版，短引用（≤25 字/处）注明出处。

## 质量自检（每章写完过一遍）

1. 一个完全没接触过 infra 的应届生，只读这一章能否明白每个名词？
2. 是否从原理到方案**串成了线**（不是词条罗列）？
3. 数字是否都能在 README/教材里找到出处？
4. 掌握目标是否可验证（能画/能算/能答）？

## 每章结构（骨架固定、血肉灵活）

开头必有：

1. **`### chNN · 章名（训练篇 N）`** 标题行。
2. **图解**：`<!--fig:chNN-->` 占位符（构建时内联 SVG，规范见下）。

**叙述主体（必须）**——用连贯段落（每段 2–4 句），按这个顺序讲：

- **它是什么**：一句话定位 + 名词卡。
- **名词卡**：本章首次出现的术语逐个白话定义（如 Token、Prefill、Paged、AllReduce……）。放一个小表格或紧凑列表，每条 ≤30 字。
- **为什么需要它**：没有它会怎么样？用一个具体的小场景（"你给同事发一条消息，发现模型每次回复都要重新读一遍你之前说的所有话"）。
- **怎么工作（拆到计算步骤级）**：不允许只说"机制 A 替代了机制 B"；必须把机制拆成一步步可跟随的计算/数据流（如注意力 = 出题 → 配对打分 → 取内容三步；流水线 = 依序填哪些块）。baseline 和 variant 对比着讲；关键数据结构/流程用 `代码` 标注。涉及 Prefill/Decode/Prefill-Decode 干扰的地方，必须回到两阶段时间线讲清"哪一步算多少、读多少"。
- **数字**：README 实测里最能说明问题的 1–3 个数字（不许改数值，标来源章节）。
- **边界**：什么情况下不成立/收益消失（README「已知边界」挑最重要的 1–2 条）+ `[易错]` 常见误解一条。

**应用层（每章尽量都有，确无关联可省略并说明）**：

- **产品对应**：具体产品名+模块（vLLM/SGLang/TensorRT-LLM/LMCache/FSDP/Megatron-LM/DeepSpeed 等）。README 提到的直接写；没提的标 `[业界通识]`。
- **方案视角**：什么客户/业务需要它、什么时候不需要、现场先问哪个问题（判据落到"看哪个指标"）。
- **讲给客户的一句话**：可直接用于售前/汇报，不夸大。

**结尾必有**：

- **掌握目标**：1–2 条可验证行为，动词限「画出/算出/答出/判断/区分」。
- **深挖入口**：`帖子「帖名」（日期）· 教材 pNN · 仓库 chapter_NN`（训练篇无教材页；ch30–43 写"帖子未发，读仓库"）。
- **延伸阅读**（可选但鼓励）：从下面的**固定链接池**里选 1–2 条（禁止编 URL；选与本章机制最贴的，格式：`[标题](URL)`）：
  - 注意力/Transformer 图解（讲 Q/K/V 最直观）：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  - KV Cache 问答式精讲：[Sebastian Raschka: What is a KV cache](https://sebastianraschka.com/faq/docs/kv-cache.html)
  - Prefill vs Decode 两阶段：[Redis: Prefill vs Decode Explained](https://redis.io/blog/prefill-vs-decode/)
  - 中文 KV Cache 详解：[阿里云开发者：理解 KV Cache](https://developer.aliyun.com/article/1736244)
  - 推理优化总览（含 PD 分离/量化/投机解码）：[NVIDIA: Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
  - Paged KV 显存管理论文：[vLLM PagedAttention (arXiv 2309.06180)](https://arxiv.org/abs/2309.06180)
  - 训练显存切分论文：[ZeRO (arXiv 1910.02054)](https://arxiv.org/abs/1910.02054)
  - 训练并行论文：[Megatron-LM (arXiv 1909.08053)](https://arxiv.org/abs/1909.08053)
  - 流水线并行论文：[GPipe (arXiv 1811.06965)](https://arxiv.org/abs/1811.06965)
  - FlashAttention 论文：[arXiv 2205.14135](https://arxiv.org/abs/2205.14135)

**可选（强烈鼓励）**：

- **算一算**：一个能在脑子里完成的小算例（如"70B 模型 BF16 权重 = 70×2 = 140GB"）。事实须为可推导算术或 README 数字。

## 图解 SVG 规范（每章一张）

- 画布 `viewBox="0 0 760 H"`，H ∈ 180–320。
- 结构：上行 = baseline（灰调），下行 = variant（强调调），左端行标签，右端结果标签；变量位置用 `var(--warning)` 虚线框或加粗箭头标出。
- 元素只允许：`rect`（rx=6）、`text`（.lbl 12.5 / .sub 10.5 / .tag 10.5）、`path` 箭头（marker-end）。
- 颜色只用主题变量：`--fg --muted --border --surface --accent --warning`。
- 每框 ≤10 字、箭头小字 ≤8 字、总文字 ≤90 字；`role="img"` + `aria-label="chNN 原理图解：<主题>"`。
- 存 `src/fig/chNN.svg`，md 里 `<!--fig:chNN-->` 引用（**不要放进反引号**）。

## 红线

- 机制事实、数字、边界以 `chapter_NN/README.md` 为准；教材 PDF（推理篇）可补理论表述，数字以 README 为准。
- 不虚构 benchmark；不确定的业界事实标 `[业界通识]` 或不写。
- 每章正文（除图解）≈ 700–1000 汉字；不写过渡废话。
