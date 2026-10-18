# 训练篇（ch16–ch29）：从单卡 SFT 到数据并行

主线是 `Qwen/Qwen3-0.6B-Base` 全参数 SFT：一章只引入一个变量，每章都有 baseline/variant 两条路径、正确性检查与 Benchmark 入口。单卡章节与推理篇同一类 12GB 显卡，多卡覆盖单机 2/4 卡与双节点 4+4；多卡聚合显存不是统一显存，性能还受 PCIe、NUMA、NCCL 与节点间网络约束。仓库只提供机制实现与复现入口，不附带可外推的实验结论。

### ch16 · 手写Qwen3-0.6B全参数SFT（训练篇 01）

**先回答最基本的问题：一次全参数 SFT 的一步，到底由哪些状态和计算构成？**

SFT 的全称是 Supervised Fine-Tuning（监督微调）：拿一批"输入对话 → 期望回答"的数据，在预训练权重上继续训练，让基座学会按对话格式回答。训练框架把 Forward、Backward、优化器、断点恢复都封装成黑盒，出了数值问题无从下手。本章不用 Transformers、Trainer 或任何训练框架，只用 Python 和 PyTorch 手写从数据编码到可恢复 Checkpoint 的完整链路，为全系列建立 baseline。

<!--fig:ch16-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 全参数 SFT | 模型里所有权重都参与训练，区别于只训一部分的"参数高效微调" |
| labels | 每个位置"该预测出的正确 Token"编号表；Loss 就是拿模型输出跟它对答案 |
| Causal LM Loss | 接龙打分：位置 t 的输出跟 labels[t+1] 对答案，只许看过去 |
| Shifted Cross Entropy | 上面那条"错一位对答案"的具体损失实现 |
| 梯度裁剪 | 梯度向量太长时按比例缩短，防止一步走得太猛（本章阈值 1） |
| AdamW | 最常用的参数更新算法：按梯度的一阶/二阶动量走步，附带权重衰减 |
| Scheduler | 学习率随训练进度变化的日程表 |
| Checkpoint | 训练状态的完整存档（权重+优化器状态等），恢复后能接着跑下一个 Step |
| Eager Attention | 注意力的朴素实现：把完整分数矩阵算出来存住，后续章节会替换它 |

**为什么需要它**：训练框架把"一步"藏成一个函数调用。你要研究"训练系统每一笔开销花在哪"，就得先有一条每行代码都能打开看、数学上可验证的链路——否则后面任何优化的对照都无从谈起。

**这一章怎么工作**：组件全部手写——直接解析 `tokenizer.json` 的 Byte-level BPE、RMSNorm/RoPE/GQA/SwiGLU 模型、Safetensors 权重加载，固定基座 `Qwen3-0.6B-Base@da87bfb6`。一步的构成：数据编码 → Forward → Shifted Cross Entropy → Backward → 全局梯度裁剪（阈值 1）→ AdamW → Scheduler → Checkpoint。训练设置刻意保守：单卡、FP32、Micro-batch 1（一次只喂一条样本）、固定序列长度、Eager Attention，除 Padding 外完整对话 Token 都参与 Loss。训练数据是从 MiniMind `sft_t2t_mini.jsonl` 校验大小与 SHA256 后按固定行号选出的 32 条确定性样本。

**数字**（chapter_16 README）：数据文件校验 1,739,201,170 字节后选 32 条样本；正确性靠三条证据链支撑——手算 Shifted Cross Entropy、小模型有限差分梯度对照、Checkpoint 恢复后复现同一个下一 Step。

**算一算**：0.6B 参数 FP32 训练，光是主权重就占 0.6×10⁹ × 4 字节 ≈ 2.4 GB；再加梯度和 AdamW 的两份动量状态（各同尺寸），一份参数至少要背上 4 份 FP32 存储 ≈ 9.6 GB——这就是为什么后续章节要不断在显存账本上做文章。

**边界与常见误区**：本章刻意不追求效率，所有 Benchmark 只是 baseline 入口，不构成可外推结论。[易错] "Loss 在降"不是实现正确的证据——正确性只能靠手算、有限差分和恢复复现三条证据支撑；Loss 下降在梯度算错时同样会发生。

**产品对应**：相当于把 HF Trainer/accelerate 的内部拆开看一遍 `[业界通识]`；Checkpoint 模块对应框架的 resume-from-checkpoint。

**方案视角**：评估客户的训练方案，先问"你们的等价基线是什么、怎么验证的"。没有可验证基线的优化报告，数字再好也无法归因。

**讲给客户的一句话**：先跑通一条数学上可验证的最小训练链路，后面的每一次优化都有等价性对照可查。

**掌握目标**：能画出单步训练的完整数据流（编码→Forward→Loss→Backward→裁剪→AdamW→Checkpoint）；能区分哪些证据证明"梯度对"、哪些证明"断点恢复对"。

**深挖入口**：帖子「SFT」（9/20）· 仓库 chapter_16

### ch17 · Assistant-only Loss（训练篇 02）

**同一个问题换一面：模型到底该跟谁学？**

第 16 期的 Loss 让除 Padding 外完整对话的 Token 都成了预测目标——System 提示词、User 的问题、Assistant 的回答，模型都在花梯度学"怎么提问"。SFT 真正想教的是"怎么回答"，所以要把监督信号精确切到回答上。本期保持对话文本、`input_ids`、模型、优化器一切不变，只改 `labels` 中哪些位置置为 `ignore_index=-100`。

<!--fig:ch17-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| ignore_index=-100 | PyTorch Loss 约定的"此题不计分"标记：labels 里值为 -100 的位置不产生 Loss 与梯度 |
| Loss Mask | 决定哪些位置参与训练的掩码，就是把不要的位置写成 -100 |
| Assistant-only Loss | 只监督 Assistant 回答内容的掩码方案，本章 variant |
| Tool Call | 模型发出的"调用工具"指令文本，本轮也纳入监督 |
| Tool Response | 工具返回给模型的结果，属环境输出，不监督 |
| 结束哨兵 | 截断时编码器额外补入的结束标记，非真实消息边界，不监督 |
| `<\|im_end\|>` | Qwen3 模板里一条消息的真实结束符，Assistant 消息的它参与监督 |

**为什么需要它**：一句话定位——监督集合的边界决定模型学什么。baseline 让模型"复述整段对话"，variant 让模型"学会回答"；数据、模型、优化器一模一样，只有掩码不同。

**这一章怎么工作**：baseline 监督除 Padding 外的完整对话；variant 只监督三样东西——Assistant 正文、它发出的 Tool Call 序列、该消息真实的 `<|im_end|>`。被屏蔽（置 -100）的位置包括：`<|im_start|>assistant\n` 头部、System、User、Tool Response、消息间换行、Padding、截断补入的结束哨兵。模板边界来自固定模型 revision 的 `tokenizer_config.json`，Jinja2 上游渲染与手写渲染要求逐字符一致（模板 SHA256 固定为 `87a2728c…`）。注意：前向 Logits 和计算 Shape 完全不变，变的只是监督集合——所以 Loss 数值、梯度方向和一次参数更新都会改变，这不是 bug，是目标变了。

**数字**（chapter_17 README）：32 条样本与第 16 期完全一致；两种掩码方案的监督 Token 数量差异由 `compare_correctness.py` 的 Mask 统计量化；唯一变量是 labels 中非 -100 的位置。

**边界与常见误区**：模板边界依赖固定 revision 的 tokenizer 配置，遇到 `reasoning_content` 等无法无损表示的字段直接报错而不是静默吞掉。[易错] 把掩码理解成"只是少算一些 Loss"——它改变的是梯度的方向与每次更新的内容，不是训练量的简单缩放。

**产品对应**：训练框架里的 loss mask / chat template 概念 `[业界通识]`；数据标注流程中"哪段参与训练"的字段设计。

**方案视角**：收到微调报价单时，问一句"你们监督 Assistant 段还是整段对话"。两种选择都合法，但效果预期和计费 Token 数完全不同，判据是客户要"模仿回答"还是"学会格式"。

**讲给客户的一句话**：同一份数据，监督掩码的设计直接决定模型学到"回答"还是"复述"。

**掌握目标**：能画出一轮对话的 Token 序列并标出两种掩码方案各自监督的 span；能判断一段给定文本在两种方案下是否参与 Loss。

**深挖入口**：帖子「Assistant-only Loss」（9/21）· 仓库 chapter_17

### ch18 · 固定Mini-batch（训练篇 03）

**样本怎么进模型，才能既快又不偷偷改变训练目标？**

一条更新内逐个样本 Forward/Backward 太慢——GPU 眼里的矩阵太小。把多个样本拼成一个二维 Batch 一次算，批量矩阵计算就能吃满并行度。但变长样本长度不一，拼 Batch 必须 Padding（把短样本用无意义 Token 补齐到最长），而且一不留神会在合并时暗中改变训练目标。

<!--fig:ch18-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Mini-batch | 一次参数更新用的一小批样本 |
| Padding | 给短样本补无意义 Token，把所有样本凑成等长矩形 |
| 右 Padding | 补在序列末尾的 Padding 方式，便于用 Mask 屏蔽 |
| Loss reduction | 把所有 Token 的 Loss 归并成一个数的规则——按 Token 平均还是按样本平均 |
| token_mean | 全部有效监督 Token 等权平均：L = Σℓ / Σnᵢ |
| sample_mean | 每个样本先求平均，再让每个样本等权：(1/B)Σ(1/nᵢ)Σℓ |
| 物理槽位 | 实际送进模型的 Token 总格数，含 Padding |
| 有效监督 Token | labels 不为 -100 的真实训练目标 |

**为什么需要它**：变长样本下 `token_mean` 和 `sample_mean` 不是同一个训练目标。如果逐样本路径和批量路径用了不同的加权，合 Batch 前后训练目标就悄悄变了——这在报表上看不出来，只会表现为"效果莫名变差"。

**这一章怎么工作**：baseline 是一组 `[1, Tᵢ]` 调用逐样本累加梯度；variant 右 Padding 到组内最大长度后一次 `[B, max(Tᵢ)]` 调用，两条路径用相同样本、Labels、有效监督 Token 和一次 AdamW Step。逐样本路径要复现 `token_mean`，第 i 个样本的 mean Loss 必须乘权重 `nᵢ / Σnᵢ` 后再 Backward（nᵢ 是它的有效监督 Token 数）；用 `sample_mean` 时则每个样本乘 `1/B`。两种权重绝不能混用。代价侧：物理槽位从 `ΣTᵢ` 变成 `B×max(Tᵢ)`，Padding 槽位和激活显存都增加。

**算一算**：一组三个样本长度 `T=[900, 200, 300]`：物理槽位 `3×900=2700`，有效 Token 只有 1400，Padding 浪费约 48%。这就是"吞吐要用有效监督 Token/s 衡量"的原因。

**数字**（chapter_18 README）：Benchmark 扫 Batch Size 1/2/4/8；`physical_tokens`、`non_padding_tokens`、`supervised_tokens` 三本账分开记；输入未用 Pinned Memory，数据在测量前已编码进 Host 内存，故磁盘读取不列为热路径指标。

**边界与常见误区**：批量计算更快但有 Padding 浪费，这正是 ch19 分桶的引子。[易错] "样本平均后再平均"看起来最自然，却把长短样本强行等权，与按 Token 等权是两个训练目标——长短样本监督 Token 数悬殊时偏差可复现。

**产品对应**：DataLoader 的 collate 函数与 padding 策略 `[业界通识]`；训练配置里的 `group_by_length` 选项是它的后续。

**方案视角**：看供应商吞吐指标时，追问分母是物理 Token、非 Padding Token 还是监督 Token——三者可差数倍，判据是拿同一批数据的 supervised Token/s 对齐。

**讲给客户的一句话**：合 Batch 不是免费提速，是用 Padding 槽位换批量矩阵计算，账要算监督 Token 的吞吐。

**掌握目标**：能算出给定长度组 `T=[900, 200, 300]` 的物理槽位数与浪费比例；能区分 `token_mean` 与 `sample_mean` 并写出等价加权。

**深挖入口**：帖子「固定Mini-batch」（9/23）· 仓库 chapter_18

### ch19 · 按长度分桶（训练篇 04）

**Padding 浪费是 Batch 内长度差造成的——能不能让"队友"长度接近，又不牺牲随机性？**

合 Batch 之后，长短样本混在一组时，短样本要陪长样本填满整行槽位。想让 Batch 内长度接近，办法是按长度排序组批；但完全按长度排序会得到一个几乎固定的训练顺序，数据随机性就没了。折中方案：在有限的 Shuffle 窗口内排序。

<!--fig:ch19-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Batch Sampler | 决定"哪些样本组成一个 Batch"的组件 |
| Shuffle Window | 只在窗口内排序的"局部随机"范围，本章旋钮 |
| 分桶 | 窗口内按长度排序后切块组批，让同批长度接近 |
| batch_length_spread_mean | Batch 内最长与最短长度之差的均值——直接决定 Padding 量 |
| companion_repeat_ratio | 跨 Epoch 同批共现样本对的重复率，衡量随机性损失 |
| plan | 一次 Epoch 的批次划分方案，是种子等输入的纯函数，可重放 |
| Epoch | 全部训练数据过一遍 |

**为什么需要它**：一个具体场景：Batch Size 4，长度 [510, 505, 60, 55] 四条样本同批，物理槽位 2040 里近七成是 Padding——GPU 在替无意义的 Token 算前向。分桶让这四条各自找到长度相近的队友。

**这一章怎么工作**：baseline 用固定种子全局 Shuffle 后顺序切块；variant 用**同一个**随机排列切成大小为 `window` 的窗口，窗口内按编码长度稳定排序（等长样本保持随机顺序）再切块，最后用同一随机流打乱 Batch 顺序。窗口越大，Batch 内长度越接近、Padding 越少；代价是 Batch 成员越来越由长度决定（`companion_repeat_ratio` 趋近 1），全排序还可能带来可测的训练顺序损失。窗口必须不小于且整除 Batch Size；`window == batch_size` 时每个窗口就是一个随机块，排序不改变成员，分桶与 baseline 完全一致——这个退化性质本身就是正确性对照。plan 是 `(lengths, seed, epoch, batch_size, window)` 的纯函数，结果 JSON 保存完整 plan，任何一次运行都能重放。

**数字**（chapter_19 README）：固定 512 条样本、Batch Size 4，窗口扫描 {4(随机), 8, 16, 32, 64, 128, 256, 512}；学习率 2e-5 无调度器、种子 20260912；脚本断言每 Epoch 监督 Token 总量等于编码语料总量、物理 Token 总量等于长度投影值。

**算一算**：512 条样本、batch 4，全 Epoch 有 128 个 Batch。窗口从 4 开到 512，意味着"排序范围"从 1 个 Batch 扩大到整个 Epoch——`batch_length_spread_mean` 单调下降，`companion_repeat_ratio` 单调上升，窗口就是这个交换的连续旋钮。

**边界与常见误区**：窗口受"整除 Batch Size"约束，不能任意取值；收益上限由语料自然长度分布决定。[易错] "分桶破坏了随机性"只说对一半——全局排列仍是随机的，被牺牲的是跨窗口的成员共现多样性，且这个损失可用 `companion_repeat_ratio` 实测。

**产品对应**：HF Trainer 的 `group_by_length`、各框架的 length-grouped batching `[业界通识]`。

**方案视角**：长文本混合语料且按 Token 计费时，分桶几乎总是划算；问供应商窗口取多大、有没有报告 Padding 比例随窗口的变化曲线。

**讲给客户的一句话**：分桶用一点样本共现的随机性，换回被 Padding 吃掉的算力，窗口大小可调。

**掌握目标**：能画出 window=8、batch=4 时 16 条样本的分组过程；能判断一个窗口取值是否合法并解释退化性质。

**深挖入口**：帖子「分桶」（9/28）· 仓库 chapter_19

### ch20 · Sequence Packing（训练篇 05）

**分桶缩小了长度差，但短样本仍独占一行——能不能让它们拼满一行，还互相看不见？**

分桶的极限是：只要一个样本独占一行，它自己内部的长度差没人管。Sequence Packing 把多个不切分的样本装进同一个固定容量的物理序列，把 Padding 槽位换成有效 Token。代价是必须严防"串门"：样本 A 的 Token 绝不能被样本 B 的 Attention 看见。

<!--fig:ch20-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Pack | 多个样本共居的一个物理序列，容量固定 |
| 装包（First-Fit） | 贪心地把样本依次塞进还能装下的 Pack，样本不切分 |
| Position ID | 每个 Token 的位置编号，RoPE 靠它编码相对距离 |
| Block-diagonal Mask | 块对角注意力掩码：同一样本内部因果可见，跨样本全屏蔽 |
| 加性 Mask | 用 `finfo(dtype).min` 加到分数上的屏蔽方式，Softmax 权重精确为 0 |
| 边界 Label 截断 | 每个样本首个 Label 置 -100，防止上一样本末 Token"预测"下一样本首 Token |
| 负对照 | 故意用错误机制（如 EOS 直拼不 Mask）验证隔离确实必要 |

**为什么需要它**：三个正确性约束缺一不可，否则训练语义悄悄出错：① Position ID 必须从 0 重置——RoPE 是相对位置编码，不平移则后续样本全部位置相对独立执行发生平移；② 跨样本与 Padding 位置以加性最小值屏蔽——Softmax 权重精确为 0，这是隔离能逐位验证的原因；③ Causal Shift 会让位置 t 预测 labels[t+1]，不截断边界 Label，前一样本的末 Token 就会把下一个样本的首 Token 当目标。简单的 EOS 拼接加跨样本 Attention 只作负对照。

**这一章怎么工作**：baseline 沿用 Padding Batch（Batch Size 4，右 Padding）；variant 用固定种子 Shuffle 后 First-Fit 装包生成 plan（`(lengths, capacity, seed, epoch)` 的纯函数），`build_pack` 拼接 Token、重置 Position ID、截断边界 Label，`build_block_diagonal_mask` 造块对角加性掩码。模型 `forward` 新增可选 `position_ids` 与 `attention_mask_4d` 注入点，缺省时与之前逐位一致。

**数字**（chapter_20 README）：语料中 26% 样本短于 512 Token，天然能形成多样本 Pack；baseline 每 Epoch 128 Step，Packing 的 Step 数由装包结果决定——Step 数不同是 Packing 的固有属性，不是实现偏差。

**算一算**：容量 512 的一个 Pack 装下长度 [300, 180] 两条样本：Padding Batch 里它们占两行共 `2×300`（按 300 与 180 各自成行即 600）槽位，Pack 只占 512 槽位且几乎无 Padding——省下的正是行与行之间的补齐。

**边界与常见误区**：收益由长度分布决定——长度整齐的语料没什么可装；每步固定成本（Optimizer、Kernel 启动）随 Pack 数线性放大，单 Pack 每步可能仍慢于 4 行 Padding Batch。[易错] 只拼 Token 不做 Mask 是"看似省了槽位、实则训练语义已错"的最典型事故。

**产品对应**：packing + block-diagonal attention `[业界通识]`；数据引擎的 sequence packing 选项与多种 Mask 实现。

**方案视角**：语料长度方差大时 Packing 收益最大；问供应商"packing 的 mask 方案是什么、有没有逐位隔离验证"，而不是只看吞吐提升。

**讲给客户的一句话**：Packing 把 Padding 槽位变成有效 Token，正确性靠位置重置、块对角 Mask、边界 Label 三条锁住，缺一条训练语义就悄悄错了。

**掌握目标**：能画出两个样本共居一个 Pack 的 Block-diagonal Mask 结构；能答出三条约束分别防止哪种错误。

**深挖入口**：帖子「Sequence Packing」（9/30）· 仓库 chapter_20

### ch21 · 离线Token Shard（训练篇 06）

**同一个数据集每个 Epoch 都被重复渲染和编码——这种确定性计算为什么占着训练热路径？**

Chat Template 渲染和 BPE 编码（把文本切成 Token 编号）是纯确定性计算：同样的输入永远得到同样的输出。它们却发生在训练的每一步数据路径上。把编码搬到训练前一次性做完，训练只做读取——但必须保证读回来的和在线编码逐 Token 一致，坏了能被发现。

<!--fig:ch21-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Tokenize | 文本 ↔ Token 编号的转换，BPE 是具体算法 |
| 在线编码 | 训练进程里现读现渲染现编码的 baseline 方式 |
| Shard | 离线写好的二进制 Token 数据文件，训练按索引随机读 |
| revision | 模型/数据的确切版本号，绑进文件头防止版本错配 |
| FNV-1a 64 | 一种快速非加密哈希，这里只用来检测意外损坏 |
| 校验和 | 给每条记录算的"指纹"，读回时重算比对 |
| O(1) 随机读 | 按记录号直接跳到位置读取，不用顺序扫描 |
| materialize(index) | 两条路径暴露的同一接口：给下标返回编码结果 |

**为什么需要它**：一个具体场景：同样的 512 条样本，第 3 个 Epoch 又被完整渲染、编码了一遍——GPU 在等 CPU 做它第 1 个 Epoch 就做过的同一件事。而且"数据到底是哪个版本"如果没有落在字节上，复现实验就全靠口头承诺。

**这一章怎么工作**：baseline 在线读 JSONL、逐条渲染编码；variant 训练前一次性编码成二进制 Shard。格式：头部固定 100 字节——magic `MISFTSKD`、格式版本、40 字节模型 revision、40 字节数据 revision、最大长度、记录数；每条记录以 source_line 和长度开头，接 input_ids（u32）与 labels（i32，含 -100），以 FNV-1a 64 校验和结尾；文件尾部是"索引区起始位置"自描述指针加记录偏移表，支持 O(1) 随机读。读取端强制校验：magic/版本/revision 不符、校验和不匹配、索引错位、文件截断，一律抛错而非静默回退。`OnlineCorpus` 与 `ShardCorpus` 暴露同一 `materialize(index)` 接口，逐 Token 一致性可全量比对。

**数字**（chapter_21 README）：头部 100 字节；512 条样本逐 Token 全量一致性比对；每条记录校验和为 FNV-1a 64 位。

**边界与常见误区**：收益大小取决于在线编码成本占训练 Step 的比例，需要实测而非默认可观。[易错] 把 FNV-1a 当成完整性保证的全部——它不是密码学哈希，不防恶意篡改；revision 绑定和文件大小/SHA256 校验是另一层。

**产品对应**：预 tokenize 数据格式，如 tokenized datasets / 预编码数据缓存 `[业界通识]`。

**方案视角**：多 Epoch 长训练或数据集跨机分发时，离线 Shard 几乎稳赚；一次性小实验则要权衡构建成本，判据是"在线编码时间 ÷ Step 时间"。

**讲给客户的一句话**：数据文件自带版本与校验，训练只做读取和验证——复现实验从数据格式这一层就有据可查。

**掌握目标**：能画出 Shard 的头部/记录/索引三段布局；能区分"校验和防什么"与"revision 绑定防什么"。

**深挖入口**：帖子「离线Token Shard」（10/1）· 仓库 chapter_21

### ch22 · DataLoader预取与异步H2D（训练篇 07）

**GPU 算完一步就闲着等下一批数据——这份"等待"能不能藏进"计算"的时间窗里？**

Shard 读取、校验、组批和 Host-to-Device 拷贝仍发生在每步关键路径上。同步路径里，GPU 计算和数据准备严格串行；而"读取第 k+1 批"和"计算第 k 批"本来毫无依赖，完全可以同时进行。

<!--fig:ch22-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| H2D | Host-to-Device：把数据从内存拷到显存 |
| 预取 | 趁 GPU 算当前批，提前准备下一批 |
| Worker 线程 | 后台专门负责物化未来 Batch 的线程 |
| Pinned Memory | 锁页内存：操作系统不许换出的内存，DMA 可直接异步拷贝；普通内存的"异步拷贝"会退化为同步 |
| 队列深度 | 最多缓存几个待用 Batch，本章为 2，限制宿主内存占用 |
| CUDA Stream | GPU 上的任务队列，不同 Stream 的任务可并行 |
| 双缓冲 | 两份缓冲轮流用：一份在算，另一份在拷 |
| 事件（Event） | Stream 里的"完成标记"，另一条流可等它 |

**为什么需要它**：收益的上限在动手前就能想清楚：可隐藏量不可能超过同步路径的数据侧耗时。当数据成本远小于训练 Step 时，端到端收益自然有限——所以这一章的功夫一半在实现，一半在"量出数据侧到底占多少"。

**这一章怎么工作**：baseline 同步串行——主线程读 Shard、组批、阻塞式 H2D 到默认流，然后计算。variant 用一个 Worker 线程把未来 Batch 物化进 Pinned Memory（有界队列深度 2），并以错拍双缓冲调度：第 k 步计算前，主线程先在拷贝流上对第 k+1 步发起 `non_blocking=True` 的 H2D（源缓冲为 Pinned Memory）并记录事件，再让计算流 `wait_event` 第 k 步的拷贝事件。这样第 k+1 步的 H2D 与第 k 步的计算并行；计算流使用某 Batch 前必须等到它的拷贝事件完成，Pinned 源缓冲在事件完成前不会被复用。Worker 异常通过哨兵与 `check_error` 在主线程显式抛出，不静默吞掉。

**数字**（chapter_22 README）：Worker 数固定 1、队列深度 2；两条路径消费同一个第 21 期 Shard、同一 Batch 计划与同一训练步体；可隐藏上限 = 同步路径的数据侧耗时。

**边界与常见误区**：队列深度 2 是宿主内存占用与预取距离的折中；Worker 只开 1 个。[易错] 以为加了 `non_blocking=True` 就异步了——没有 Pinned 源内存和事件管理，拷贝仍会阻塞在默认流上，"伪异步"是这类流水线最常见的坑。

**产品对应**：DataLoader 的 `prefetch_factor` + `pin_memory=True` + `non_blocking` 传输 `[业界通识]`。

**方案视角**：数据侧耗时占比可以用分段计时量出来；占比小于一成时，预取不是优先级，先看计算段，判据是 CUDA Event 分段时间线。

**讲给客户的一句话**：数据预取是把"等数据"藏进"算梯度"的时间窗里，能藏多少取决于数据侧原本占多少。

**掌握目标**：能画出双缓冲的时序图（计算 k 与拷贝 k+1 重叠）；能判断一个数据流水线瓶颈在数据侧还是计算侧。

**深挖入口**：帖子「异步H2D」（10/3）· 仓库 chapter_22

### ch23 · BF16混合精度（训练篇 08）

**FP32 稳但慢——能不能让"算"用低精度、"存"仍是高精度，两头都占？**

FP32 的前向/反向由 GEMM（矩阵乘）主导，没吃到 Tensor Core 低精度算力的红利；但直接把参数全降精度又会动优化器的数值稳定性。混合精度的答案：算的部分降，存的主状态不降。

<!--fig:ch23-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| FP32 | 32 位浮点数，精度高、显存和带宽翻倍 |
| BF16 | 16 位脑浮点：指数位与 FP32 同多，动态范围一样大、尾数更少 |
| autocast | PyTorch 上下文管理器：进入区间后算子自动选低精度执行 |
| Tensor Core | GPU 上专做低精度矩阵乘的硬件单元，BF16 吞吐远高于 FP32 |
| 混合精度 | 算子用低精度、主参数/梯度/优化器状态保持 FP32 的折中方案 |
| GradScaler | FP16 专用的损失放大器，防小梯度下溢；BF16 不需要 |
| TF32 | 另一种自动降精度的 GEMM 模式，本章显式关闭以免混淆变量 |
| dtype 普查 | 逐模块/逐参数记录实际 dtype 的盘点手段 |

**为什么需要它**：不换精度，时间花在"每乘一个数都搬 4 字节"上；全换低精度，优化器的小步更新会被舍入吃掉。混合精度把两者分开：吃算力的部分（GEMM、激活）用 BF16，吃数值稳定性的部分（主权重、梯度、AdamW 动量）保持 FP32。

**这一章怎么工作**：baseline 全 FP32；variant 的 Forward 与 Loss 在 `torch.autocast(bfloat16)` 下执行，主参数、梯度、AdamW 状态仍 FP32——这就是"混合"的含义。BF16 的指数位与 FP32 相同，动态范围大，因此不需要 FP16 那套 GradScaler。效果预期：GEMM 走 Tensor Core 后前向/反向显著缩短；激活显存约减半，但 FP32 主状态不动，峰值显存只部分下降。代价：Loss 与梯度出现 BF16 舍入级分叉并随步数放大，用整 Epoch 轨迹误差量化；`mixed_precision.py` 用 forward hook 逐模块记录输出 dtype，并对参数/梯度/优化器状态做 dtype 直方图。TF32 显式关闭并记录，避免混入第三个精度变量。

**数字**（chapter_23 README）：512 条样本、批次计划与 ch19–22 相同；激活显存约减半、FP32 主状态不变；NaN/Inf 扫描与轨迹误差汇总是正确性的两只仪表。

**算一算**：0.6B 模型：激活显存约减半意味着前向激活部分省一半；但主权重 2.4 GB + 梯度 2.4 GB + AdamW 两份动量 4.8 GB 仍是 FP32 共约 9.6 GB——混合精度省的是"算"，不是"存"。

**边界与常见误区**：TF32 被固定关闭，防止它悄悄替你"开"了低精度 GEMM 污染归因。[易错] 把 FP16 的经验（必须 GradScaler、可能 NaN）直接套到 BF16 上——BF16 的动态范围与 FP32 相同，不需要 loss scale。

**产品对应**：`torch.autocast` 混合精度；GradScaler 是 FP16 方案 `[业界通识]`；训练配置里的 `bf16=True` 选项。

**方案视角**：显卡有 BF16 Tensor Core 时混合精度基本是标配；问供应商"精度策略是什么、优化器状态什么 dtype、有没有轨迹误差监控"，判据是 dtype 普查而不是口头承诺。

**讲给客户的一句话**：混合精度省的是计算和激活，不省优化器状态；数值分叉需要用轨迹误差监控来定价。

**掌握目标**：能画出一行参数在混合精度下的各段 dtype（前向激活 BF16 / 主参数与优化器 FP32）；能算出激活显存大约减半而状态显存不变的账。

**深挖入口**：帖子「BF16」（10/5）· 仓库 chapter_23

### ch24 · 梯度累积（训练篇 09）

**想要大 Batch 的稳定梯度，单卡显存却装不下——能不能把一步拆成 K 次小计算，效果还一样？**

单步能容纳的 Batch 受单卡显存约束。梯度累积的思路：一个 Optimizer Step 不由一次大 Forward/Backward 构成，而是由 K 个 Micro-batch（微批，小份样本组）的梯度累加而成。关键不是"累加"本身，而是累加时的加权规则——加错了，训练目标就和大 Batch 不是同一个。

<!--fig:ch24-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Micro-batch | 一次装得下的样本小组；若干微批凑成一个"有效大批" |
| 梯度累积 | 连续多个微批的 Backward 不更新参数，只把梯度加起来 |
| Optimizer Step | 用累积好的梯度更新一次参数的动作 |
| 有效 Batch | 累积后的等效批量：微批尺寸 × K |
| token_sum_i | 第 i 个微批的监督 Token 数 |
| total_supervised | 整个有效 Batch 的监督 Token 总数 |
| OOM | 显存不足报错（Out Of Memory） |
| 反例验证 | 故意用错误加权跑一遍，证明偏差真实存在 |

**为什么需要它**：想要大 Batch 的训练语义（更稳的梯度估计、更平滑的 Loss 曲面），单步显存却装不下。拆成 K 次，每次显存只按微批算——容量问题就变成了时间问题。

**这一章怎么工作**：baseline 一次大 Batch 完成 Forward/Backward/Step；variant 把同一批样本按同一 shuffle 分成 K 个微批，各自 Forward/Backward，最后一次裁剪 + Step。正确性全在一个加权规则：微批 i 的损失按 `token_sum_i / total_supervised` 缩放，累积出的梯度才与一次大 Batch 的 `token_mean` 严格同口径。如果每个微批的 Loss 取均值再简单平均，在监督 Token 数不均时会偏离联合目标——本章用反例显式验证这一点，而不是口头声明。FP32 下累积梯度与大 Batch 只差求和顺序噪声。代价是 K 次小 GEMM 的时间加总；收益是微批级显存上限带来的容量（大批 OOM 的配置可以跑）与更细的 Padding 控制。

**数字**（chapter_24 README）：Benchmark 四臂 `big4 / acc_2x2 / acc_4x1 / acc_4x4`（大批 4 / 微批 2×2 步 / 微批 1×4 步 / 微批 4×4 步）；大批递增做 OOM 扫描定位容量边界；`make_step_plan` 用同一 shuffle 按 16 分组再切片，保证四臂吃到完全相同的样本序列。

**算一算**：两个微批分别有 100 和 300 个监督 Token，总监督 400：第一个微批的 Loss 要乘 `100/400 = 0.25`，第二个乘 `300/400 = 0.75`——监督多的微批权重大，这才与"全部 400 个 Token 等权平均"同口径。

**边界与常见误区**：本期是单卡；分布式下"累积期间不同步梯度"（`no_sync`）是它的天然搭档，属后续章节变量。[易错] "每个微批 Loss 取均值再平均"——监督 Token 数不均时它不等于联合 `token_mean`，这不是理论洁癖，是可复现的偏差。

**产品对应**：训练配置的 `gradient_accumulation_steps` `[业界通识]`；几乎所有训练框架的标准选项。

**方案视角**：客户说"显存不够上大批"时，先看能不能用累积凑出等效 Batch；判据是目标有效 Batch 大小 = 微批尺寸 × K，以及可接受的步时间增加。

**讲给客户的一句话**：梯度累积用时间换显存容量，等价性的钥匙是每个微批按监督 Token 占比加权。

**掌握目标**：能算出给定两个微批监督 Token 数下各自应有的损失缩放系数；能判断一种累积实现是否与大 Batch 同口径。

**深挖入口**：帖子「梯度累积」（10/10）· 仓库 chapter_24

### ch25 · Activation Checkpointing（训练篇 10）

**显存里最贵的一批"租客"是中间激活——能不能只留房卡，退房时再重造？**

ch24 解决了大 Batch 的显存准入，但 autograd（PyTorch 的自动求导机制）为 Backward 保留的中间激活仍是显存大头：28 层 Transformer 每层的注意力分数、FFN 中间值都被扣着等反向用。Activation Checkpointing 的方案：激活不保存，只保存每层（段）的输入，Backward 时重算前向。

<!--fig:ch25-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 激活 | 前向算出的中间结果，Backward 求导时要复用 |
| autograd | PyTorch 自动求导：记录前向计算图，反向沿图算梯度 |
| Checkpointing | 只存段边界输入、反向时重算段内前向的策略 |
| 段（Segment） | 连续若干层打包成的一个"重算单位" |
| RNG 状态 | 随机数生成器的进度存档；dropout 每次丢哪些数由它决定 |
| 重算 | Backward 时把该段前向再算一遍，临时重建激活 |
| 粒度 | 段划得多细：越细越省显存、重算越多 |
| 非重入式 | `use_reentrant=False` 的官方 checkpoint 实现方式 |

**为什么需要它**：一个具体场景：28 层模型 batch 4、长度 512，28 层的激活全部扣着等反向——这是显存峰值的主要来源。若只存每段输入，段内激活用完即弃，峰值显存大幅下降；代价是 Backward 里多出约一次前向的重算。

**这一章怎么工作**：baseline 保存全部激活到 Backward；variant 只保存每段输入，Backward 时先恢复该段 RNG 状态（dropout 等随机算子依赖它），再 `enable_grad` 重算一次前向、然后正常反传。本章手写 `RecomputeCheckpoint`——一个 autograd Function：forward 存输入与 RNG，backward 恢复 RNG → 重算 → 传播并返回输入梯度——并与 `torch.utils.checkpoint` 的 `use_reentrant=False` 对照。dropout 正反例验证："不恢复 RNG 就逐位不一致"——RNG 状态是正确性的一部分，不只是工程细节。粒度扫描 {1, 2, 4, 7, 14, 28} 段呈现权衡曲线：段越细越省显存、重算越多。

**数字**（chapter_25 README）：模型共 28 层，粒度扫描 6 档 {1,2,4,7,14,28}；双 Epoch 轨迹与不 checkpoint 版本要求逐位一致；容量扫描观察最大 batch/序列边界右移。

**算一算**：28 层、段数为 4 时：每段 7 层，保存的从"28 层全部激活"降到"4 个段边界输入"；代价是每个段在 Backward 里多算一次 7 层前向——全模型合计恰好多约 28 层的一次前向，即"多付一次前向"。

**边界与常见误区**：只存层输入的重算方案，不涉及按算子的选择性 checkpoint。[易错] "checkpoint 就是无损免费"——时间代价恒在；以及忘存 RNG 导致含 dropout 的重算逐位漂移。

**产品对应**：gradient checkpointing，HF Trainer 的 `gradient_checkpointing=True`、FSDP 内建 `[业界通识]`。

**方案视角**：显存不够、又不想减 Batch 时第一顺位开关；判据是客户能接受的步时间增幅（约 +1 次前向）与段数设置的匹配。

**讲给客户的一句话**：激活显存可以换成约一次额外前向的时间，段数粒度就是这之间的定价表。

**掌握目标**：能画出"存边界→重算→反传"的时序；能算出 28 层模型在段数为 4 时保存与重算的相对量。

**深挖入口**：帖子「CKPT」（10/10）· 仓库 chapter_25

### ch26 · 训练版FlashAttention（训练篇 11）

**注意力显存随序列长度平方增长——能不能算注意力但不把 S×S 矩阵摆出来？**

Eager Attention 会物化完整的 S² score/probability 矩阵（S 是序列长度）：长度翻倍，这个矩阵占的显存翻四倍，序列一长显存先爆。FlashAttention 用分块计算解决它——但它在训练里能不能用，还跟精度绑定。

<!--fig:ch26-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Attention 矩阵 | 每对"查的 Token × 被看的 Token"一个分数，共 S×S 格 |
| 物化 | 把整个矩阵实实在在算出来放进显存 |
| SDPA | PyTorch 的 `scaled_dot_product_attention`：一个接口后接多种后端实现 |
| 后端 | SDPA 背后选用的具体 Kernel（eager/math/mem_efficient/flash） |
| Flash 后端 | 真正的 FlashAttention Kernel，要求 BF16/FP16 输入 |
| mem_efficient | 显存高效后端：分块计算、FP32 也可用 |
| 注入点 | 模型里预留的替换口：`attention_impl` 为 None 时走原路径 |
| 容量边界 | 固定显存下能跑的最大序列长度 |

**为什么需要它**：一个具体场景：S=4096 时 S² 矩阵是 1600 万格，batch 一大几 GB 就没了——而且这还只是"存分数"，不算算它们的时间。分块计算把"先算完再归一化"改成"一小块一小块地算、边算边累计"，S² 矩阵从头到尾不落地。

**这一章怎么工作**：baseline 的 Eager Attention 物化 S² 矩阵；variant 换成 SDPA 分块后端，不物化 S²。后端普查（`probe_backend_support`）给出可用性矩阵：**FP32 下 flash 后端不可用（要求 BF16/FP16），可用的是 mem_efficient 后端**；sdpa_math 作为数值对照。Attention 实现以注入点方式接入模型（`Qwen3Attention.attention_impl`），None 时走与之前逐位相同的内联路径——这意味着切换后端不碰模型其余部分。不同后端之间只有舍入级数值差异；训练里 Forward 和 Backward 都走分块 Kernel。

**数字**（chapter_26 README）：512 长度下端到端收益不大——机制价值在显存与序列容量边界，序列容量扫描显示能装下的最大长度右移；FP32 下三个可测臂（eager / sdpa_math / mem_efficient）做整 Epoch 对照。

**算一算**：S² 矩阵 FP32 存储约为 `S² × 4` 字节再乘 batch 与头数：S=512 时一"头"约 1 MB（512²×4B），S=4096 时约 67 MB——64 倍于前者，这就是"容量右移"的价值所在。

**边界与常见误区**：BF16 只出现在明确标注的跨精度对照臂，不与本机制混淆。[易错] 在 FP32 训练里期待 flash 后端——它要求半精度输入，FP32 下只有 mem_efficient 可用。

**产品对应**：FlashAttention 系列、PyTorch SDPA（`torch.nn.functional.scaled_dot_product_attention`）`[业界通识]`。

**方案视角**：客户序列长度超过 2–4K 时注意力显存成为瓶颈，分块后端是刚需；判据是目标序列长度下 S² 矩阵的显存账，而不是短序列 benchmark。

**讲给客户的一句话**：FlashAttention 训练版的入门门槛是"能不能装下长序列"，而且后端可用性跟精度绑定，要按 dtype 查支持矩阵。

**掌握目标**：能画出 Eager 物化 S² 与分块计算两路径的显存对比；能答出 FP32 与 BF16 下各自可用的 SDPA 后端。

**深挖入口**：帖子「训练版FA」（10/12）· 仓库 chapter_26

### ch27 · Fused AdamW（训练篇 12）

**优化器的数学一个字没改，只是"叫法"变了——为什么还能省出可观时间？**

0.6B 模型约有 310 个参数 Tensor。朴素 AdamW（loop 模式）要对每个 Tensor 逐个发射一串小 Kernel——GPU 上"启动一个 Kernel"有固定开销，几百个小 Tensor 乘几遍，Optimizer 段的时间就不好看了。本期数学完全不变，只改 Kernel 的组织方式。

<!--fig:ch27-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Kernel | GPU 上的一次最小执行单元，启动它有固定开销 |
| Kernel 发射 | CPU 命令 GPU 启动 Kernel 的动作，次数多了就是开销 |
| loop 优化器 | 逐 Tensor 循环更新，每个 Tensor 发射一串 Kernel |
| foreach | 多个 Tensor 打包成批一次性发射，减少启动次数 |
| fused | 整个 AdamW 融成单 Kernel：读梯度、更新、写回一趟完成 |
| 参数 Tensor | 权重按结构切成的存储单元；0.6B 约有 310 个 |
| 解耦权重衰减 | AdamW 的特征：权重衰减独立于梯度动量执行 |
| 偏置校正 | Adam 对早期步估计偏小的修正项 |

**为什么需要它**：一个具体场景：模型小、步时间短，而优化器一步要对 310 个 Tensor 各发一遍"读-算-写"小 Kernel——Kernel 启动的固定开销积少成多，Optimizer 段在分段时间线里占比显眼。这时"少发射"就是零数学风险的提速。

**这一章怎么工作**：三臂对照——loop（逐 Tensor 循环）、foreach（多 Tensor 打包发射）、fused（单 Kernel 完成整个更新），AdamW 公式（含解耦权重衰减 wd 0.1 与偏置校正）完全相同。`reference_adamw_step` 提供手写单 Tensor 参照，验证三臂数值一致；`count_step_kernels` 用 profiler 统计一次 step 的 CUDA Kernel 数与设备时间，把"省在哪"变成看得见的数字。`clip_grad_norm_` 三臂走同一路径并单独计时，作为 step 分段计时的锚点——梯度裁剪不进对照变量。

**数字**（chapter_27 README）：0.6B 模型约 310 个参数 Tensor；三臂对照步 32 步轨迹要求一致且确定性；Kernel 数与设备时间由 profiler 普查量化。

**算一算**：310 个 Tensor、loop 模式下每个 Tensor 若发约 5 个小 Kernel，一步就是 ~1550 次发射；fused 把它压到一次——发射次数差三个数量级，这是纯组织层面的收益。

**边界与常见误区**：不涉及优化器状态 dtype 与分片——那是 ZeRO-1（后续章节）的事，两者维度正交。[易错] 把 fused optimizer 和"更快的收敛"混为一谈——它不改任何数学，只改执行方式；模型大、步时间长时 Optimizer 份额本就小，收益有限。

**产品对应**：`torch.optim.AdamW(fused=True)`、apex fused optimizer `[业界通识]`；框架里 `optim="adamw_torch_fused"` 一类开关。

**方案视角**：参数 Tensor 多、模型小、步时间短的场景里 Optimizer 份额才显眼；判据是 profiler 分段里 optimizer 段占比，而不是无差别上 fused。

**讲给客户的一句话**：小 Tensor 一多，优化器也能吃掉可观的步时间——省 Kernel 发射次数是零数学风险的提速。

**掌握目标**：能区分 loop/foreach/fused 三种 Kernel 组织并画出它们的发射模式；能判断一个优化器加速 claim 是否涉及数学变化。

**深挖入口**：帖子「Fused AdamW」（10/13）· 仓库 chapter_27

### ch28 · torch.compile（训练篇 13）

**Eager 模式下每行 Python 都现算现发 Kernel——能不能让编译器先看完整张图再统筹？**

Eager（即时执行）下逐元素算子链（Norm、激活、残差相加这类"逐位置运算"）有大量 Kernel 启动和内存读写开销。`torch.compile` 让编译器看到整张计算图后做融合优化；但编译不是免费的，账要分三笔记。

<!--fig:ch28-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Eager | 逐行即时执行的模式，灵活但统筹少 |
| 计算图 | 一次前向涉及的算子依赖关系图 |
| Dynamo | torch.compile 的前端：把 Python 代码翻译成计算图 |
| Inductor | torch.compile 的后端编译器：生成优化后的 Kernel |
| 算子融合 | 把相邻多个小算子合并成一个 Kernel，省启动和中间读写 |
| Graph break | Dynamo 看不懂某段代码，被迫切断图、退回 Eager |
| 重编译 | 输入 shape 变化触发再次编译；变长训练的主要风险 |
| dynamic=True | 允许动态 shape 的编译模式，用灵活性换部分优化 |
| Break-even | 稳态每步节省累计到抵扣首次编译成本所需的步数 |

**为什么需要它**：一个具体场景：训练上千步，而第 1 步花了几十秒"编译"。这账划不划算，取决于"首次编译成本 ÷ 稳态每步节省 = 多少步回本"；更隐蔽的风险是变长序列——每来一个新 shape（张量形状）就可能触发一次重编译，训练被编译风暴拖垮。

**这一章怎么工作**：baseline 是 Eager 执行；variant 的 Forward/Backward 交给 `torch.compile`（Inductor 默认配置），Loss、梯度裁剪、优化器留在 Eager。三笔分开记的账：稳态收益——Inductor 只能融合逐元素链，收益受 GEMM 份额限制（GEMM 本身已高效，融合吃不到）；首次编译成本——`timed_first_step` 把首步与稳态分离；重编译——`dynamo_stats`/`RecompileTracker` 记录图数、重编译与 graph break 计数，`dynamic=False` 对照 `dynamic=True` 演示重编译风暴。变长序列经 `pad_batch_to` 固定到 512 形状（不触碰真实 Token 与 Labels），正是为规避重编译的工程手段。

**数字**（chapter_28 README）：三臂对照 `eager_natural / eager_fixed / compiled_fixed`；CPU 编译对照要求 Loss/梯度误差 <1e-5；固定 Shape 20 步对照与编译成本单独计量。

**算一算**：设首次编译花 60 秒、编译后每步省 10 毫秒：Break-even = 60s ÷ 10ms = 6000 步。训练 1 万步划算，训练 500 步血亏——编译收益必须用"步数"摊销。

**边界与常见误区**：编译优化器/裁剪、max-autotune、手动 CUDA Graph 均不混入本期。[易错] 拿"首个 Step"测速——那包含编译时间；稳态收益要在编译完成后测，变长训练的主要风险就是重编译风暴。

**产品对应**：`torch.compile`/Inductor `[业界通识]`；训练服务的 shape 策略（固定长度/桶化）是它的配套工程。

**方案视角**：训练 shape 固定、长跑的生产任务是编译的好场景；短任务或 shape 多变时先算 Break-even。判据是"编译耗时 ÷ 稳态每步节省 = 多少步回本"，以及输入 shape 的分布。

**讲给客户的一句话**：编译的账是"首次编译成本 vs 稳态收益"，变长输入会不断触发重编译——生产里要先固定 shape 策略。

**掌握目标**：能画出 Eager 与编译路径的首步/稳态时间对比；能判断一个训练任务是否会被重编译拖累。

**深挖入口**：帖子「torch.compile」（10/13）· 仓库 chapter_28

### ch29 · DDP（训练篇 14）

**单卡机制已压到边界，吞吐还要翻倍——打开第一个并行维度时，训练语义还能严格不变吗？**

从第 17 期起训练一直发生在单张 GPU 上。数据并行（Data Parallel, DP）的思路最朴素：每张卡持有一份完整模型，各处理全局批次的一个分片，各自算梯度，最后把梯度合起来。难的不是"多卡"，而是"加卡不悄悄改变训练语义"以及"通信别成为新瓶颈"。

<!--fig:ch29-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| DDP | DistributedDataParallel：PyTorch 的数据并行封装 |
| 副本 | 每张卡上那份完整的模型+梯度+优化器状态 |
| 分片 | 全局批次按 `rank::world_size` 切给每卡的一份 |
| rank / world_size | 当前卡的编号 / 总卡数 |
| AllReduce | 让所有卡上的同一份数据（这里是梯度）求和后人人拿到相同结果的通信原语 |
| AllReduce SUM | 上述"求和"版本，本章 Backward 后执行一次 |
| Ring AllReduce | 教学实现：沿环两阶段各 W-1 步完成，通信量 `2(W-1)/W·D` |
| Strong Scaling | 固定总问题规模加卡，看加速比 |
| NCCL | NVIDIA 的集合通信库，AllReduce 的生产实现 |
| 互联 | 卡间通信线路：PCIe 慢、NVLink 快 |

**为什么需要它**：一个具体场景：两卡各算半批样本，梯度不同——参数更新会分叉。解决办法只有把梯度"合并成同一个"再更新；而 0.6B 模型的 FP32 梯度约 2.4 GB，每步都要在卡间走一个来回，互联带宽直接决定这笔通信是"顺带"还是"路障"。

**这一章怎么工作**：baseline 单卡算全局 Batch；variant 两卡各持完整副本，各处理全局批次的一个分片。等价性核心在一条数学恒等式：本地损失按 `supervised_local / supervised_global`（本地监督 Token 占全局的比例）缩放后，Backward 结束做一次 AllReduce SUM，结果与单卡大 Batch 的 `token_mean` 严格等价——和 ch24 的累积加权是同一个思想在卡间的翻版。AllReduce 通信量公式 `2(W-1)/W·D`（W 是卡数、D 是梯度字节数）；两卡经 PCIe 无 NVLink 时，每步约 2.4 GB 的 FP32 梯度 AllReduce 完全暴露在关键路径上，Strong Scaling 效率显著低于 2 是预期内的。正确性对照在 AllReduce 之后、裁剪之前捕获梯度（`after_allreduce` 钩子）；比较时点错了会看到恒定偏差，那是比较错误不是梯度错误。

**数字**（chapter_29 README）：最低 2 张 GPU，参考覆盖 1/2/4 卡与双节点 4+4；每步约 2.4 GB FP32 梯度 AllReduce 走 PCIe PHB；NCCL 微基准覆盖 1–2360 MiB 测 algbw/busbw；对照要求 8 步轨迹与跨 Rank 逐位一致。

**算一算**：W=2 时 AllReduce 通信量系数 `2(W-1)/W = 1`，每卡发出 D ≈ 2.4 GB；若 PCIe 带宽 16 GB/s，理想也要 ~150 ms/步——和一步计算同量级，这就是"通信暴露"的具体含义。

**边界与常见误区**：本期不使用 Bucket 重叠（下一章）、`no_sync`（梯度累积配合）和状态分片（ZeRO）——通信严格排在 Backward 之后全额暴露，这正是后续章节要修的。[易错] "多卡自动加速一倍"——同步点暴露时远达不到，判据是 Strong Scaling 效率。

**产品对应**：`torch.nn.parallel.DistributedDataParallel`、NCCL `[业界通识]`；所有训练框架多卡模式的地基。

**方案视角**：数据并行适合模型单卡装得下、想扩吞吐的场景；问"通信和计算重叠吗、互联是什么（PCIe/NVLink/IB）"，判据是每步通信字节数除以互联带宽。

**讲给客户的一句话**：数据并行的语义等价性很好证明，但后置的全量梯度同步在慢互联上会全额暴露——通信时机是下一个优化变量。

**掌握目标**：能画出两卡 DDP 一步的时序（前向→反向→AllReduce→更新）；能算出 W=2 时 AllReduce 的通信量 `2(W-1)/W·D`。

**深挖入口**：帖子「DDP」（10/14）· 仓库 chapter_29
