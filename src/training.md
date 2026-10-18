# 训练篇（ch16–ch43）：从单卡 SFT 到三维并行

主线是 `Qwen/Qwen3-0.6B-Base` 全参数 SFT：一章只引入一个变量，每章都有 baseline/variant 两条路径、正确性检查与 Benchmark 入口。单卡章节与推理篇同一类 12GB 显卡，多卡覆盖单机 2/4 卡与双节点 4+4；多卡聚合显存不是统一显存，性能还受 PCIe、NUMA、NCCL 与节点间网络约束。仓库只提供机制实现与复现入口，不附带可外推的实验结论。

### ch16 · 手写Qwen3-0.6B全参数SFT（训练篇 01）

**一句话定位**：不用任何训练框架，手写从数据编码到可恢复 Checkpoint 的完整 SFT 训练系统。

**解决什么问题**：训练框架把 Forward、Backward、优化器、断点恢复都封装成黑盒，出了数值问题无从下手。要研究训练系统的每一笔开销，先得有一条每行代码都能打开看的完整链路，并且数学上可验证。

**原理图解**：`<!--fig:ch16-->`

**原理要点**：
- baseline 就是本章本身：单卡 FP32、Micro-batch 1、固定序列长度、Eager Attention、除 Padding 外完整对话 Token 参与 Loss；后续每章在它上面只换一个变量。
- 一步的构成：Forward → Shifted Cross Entropy（位置 t 预测 t+1 的下一 Token）→ Backward → 全局梯度裁剪（阈值 1）→ AdamW → Scheduler。
- 组件全部手写：直接解析 `tokenizer.json` 的 Byte-level BPE、RMSNorm/RoPE/GQA/SwiGLU 模型、Safetensors 权重加载，固定 `Qwen3-0.6B-Base@da87bfb6`。
- 正确性证据链：手算 Loss、小模型有限差分梯度对照、真实权重前反向、Checkpoint 恢复后复现同一个下一 Step。
- 训练数据是 32 条确定性样本，从 MiniMind `sft_t2t_mini.jsonl` 校验大小与 SHA256 后按固定行号选取。

**重点**：**任何优化结论都必须先有一条数学上可验证的等价基线。**

**边界与常见误区**：本章刻意不追求效率，所有 Benchmark 数字只是 baseline 入口，不构成可外推结论。`[易错]` 不能把"Loss 在降"当作实现正确的证据——正确性靠手算、有限差分和恢复复现三条证据支撑。

**产品对应**：相当于把 HF Trainer/accelerate 的内部拆开看一遍 `[业界通识]`；Checkpoint 模块对应框架的 resume-from-checkpoint。

**方案视角**：评估客户的训练方案，先问"你们的等价基线是什么、怎么验证的"。没有可验证基线的优化报告，数字再好也无法归因。

**讲给客户的一句话**：先跑通一条数学上可验证的最小训练链路，后面的每一次优化都有等价性对照可查。

**掌握目标**：能画出单步训练的完整数据流（编码→Forward→Loss→Backward→裁剪→AdamW→Checkpoint）；能区分哪些证据证明"梯度对"、哪些证明"断点恢复对"。

**深挖入口**：帖子「手写Qwen3-0.6B全参数SFT」（9/20）· 仓库 chapter_16

### ch17 · Assistant-only Loss（训练篇 02）

**一句话定位**：只改 `labels` 里的 `ignore_index=-100` 掩码，把监督目标从完整对话缩小到 Assistant 的回答。

**解决什么问题**：baseline 让 System、User、Assistant 都成了预测目标，模型在花梯度学"怎么提问"。SFT 真正想教的是"怎么回答"，所以要把监督信号精确地切到回答上。

**原理图解**：`<!--fig:ch17-->`

**原理要点**：
- baseline 监督除 Padding 外的完整对话；variant 只监督 Assistant 正文、它发出的 Tool Call 文本和该消息真实的 `<|im_end|>`。
- 被屏蔽的位置：`<|im_start|>assistant\n` 头部、System、User、Tool Response、消息间换行、Padding、截断补入的结束哨兵，全部置 `ignore_index=-100`。
- 模型、数据、`input_ids` 全部不动，唯一变量是 labels 里非 `-100` 的位置；因为监督 Token 集合缩小，Loss 数值、梯度方向和一次参数更新都会改变——这不是 bug，是目标变了。
- 模板边界来自固定模型 revision 的 `tokenizer_config.json`，Jinja2 上游渲染与手写渲染要求逐字符一致（模板 SHA256 固定）。

**重点**：**改的不是模型而是监督信号——mask 的边界定义直接决定模型学什么。**

**边界与常见误区**：模板边界依赖固定 revision 的 tokenizer 配置，无法无损表示的字段（如 `reasoning_content`）直接报错而不是静默吞掉。`[易错]` 常见误解是"掩码只是少算一些 Loss"——它改变的是梯度的方向与每次更新的内容。

**产品对应**：训练框架里的 loss mask / chat template 概念 `[业界通识]`；数据标注流程中"哪段参与训练"的字段设计。

**方案视角**：收到微调报价单时，问一句"你们监督 Assistant 段还是整段对话"。两种选择都合法，但效果预期和计费 Token 数完全不同，判据是客户要"模仿回答"还是"学会格式"。

**讲给客户的一句话**：同一份数据，监督掩码的设计直接决定模型学到"回答"还是"复述"。

**掌握目标**：能画出一轮对话的 Token 序列并标出两种掩码方案各自监督的 span；能判断一段给定文本在两种方案下是否参与 Loss。

**深挖入口**：帖子「Assistant-only Loss」（9/21）· 仓库 chapter_17

### ch18 · 固定Mini-batch（训练篇 03）

**一句话定位**：把逐样本累加梯度改成右 Padding 二维 Batch 一次算，同时保住训练目标。

**解决什么问题**：一条更新内逐个样本 Forward/Backward 太慢；但变长样本长度不一，合成二维 Batch 必须Padding，一不留神还会在合并时暗中改变训练目标。

**原理图解**：`<!--fig:ch18-->`

**原理要点**：
- baseline 是一组 `[1, T_i]` 调用逐样本累加梯度；variant 右 Padding 到组内最大长度后一次 `[B, max(T_i)]` 调用，两者用相同样本、Labels 和一次 AdamW Step。
- 归约定义：`token_mean` 对所有有效监督 Token 等权（`L = Σℓ / Σn_i`）；`sample_mean` 先逐样本平均再等权（`(1/B)Σ(1/n_i)Σℓ`）。变长下二者不是同一个目标。
- 逐样本路径要复现 `token_mean`，第 i 个样本的 mean Loss 必须乘 `n_i / Σn_i` 后再 Backward；两种权重不能混用。
- 物理槽位从 `ΣT_i` 变成 `B×max(T_i)`：Padding 槽位与激活显存都增加，吞吐要用有效监督 Token/s 而非物理 Token/s 衡量。

**重点**：**合 Batch 能否保持等价，全看 Loss 归约和加权方式是否同口径。**

**边界与常见误区**：批量计算更快但有 Padding 浪费，这正是 ch19 分桶的引子。`[易错]` "样本平均后再平均"看起来最自然，却把长短样本强行等权，与按 Token 等权是两个训练目标。

**产品对应**：DataLoader 的 collate 函数与 padding 策略 `[业界通识]`；训练配置里的 `group_by_length` 选项是它的后续。

**方案视角**：看供应商吞吐指标时，追问分母是物理 Token、非 Padding Token 还是监督 Token——三者可差数倍，判据是拿同一批数据的 supervised Token/s 对齐。

**讲给客户的一句话**：合 Batch 不是免费提速，是用 Padding 槽位换批量矩阵计算，账要算监督 Token 的吞吐。

**掌握目标**：能算出给定长度组 `T=[900, 200, 300]` 的物理槽位数与浪费比例；能区分 `token_mean` 与 `sample_mean` 并写出等价加权。

**深挖入口**：帖子「固定Mini-batch」（9/23）· 仓库 chapter_18

### ch19 · 按长度分桶（训练篇 04）

**一句话定位**：只改 Batch Sampler——在有限 Shuffle 窗口内按长度排序组批，减少 Padding 浪费。

**解决什么问题**：Padding Batch 里长短样本混在一组时，短样本要陪长样本填满整行槽位。想让 Batch 内长度接近，又不想要全排序那种完全确定的训练顺序。

**原理图解**：`<!--fig:ch19-->`

**原理要点**：
- baseline 用固定种子全局 Shuffle 后顺序切块；variant 用同一随机排列切成有限 Shuffle Window，窗口内按编码长度稳定排序再切块，最后打乱 Batch 顺序。
- 窗口越大，Batch 内长度越接近（`batch_length_spread_mean` 越小）、Padding 物理槽位越少；代价是 Batch 成员越来越由长度决定，用 `companion_repeat_ratio`（跨 Epoch 共现重复率）度量，全排序还可能带来可测的训练顺序损失。
- 窗口必须不小于且整除 Batch Size；`window == batch_size` 时每个窗口就是一个随机块，分桶成员与 baseline 完全一致——这个退化性质本身就是正确性对照。
- plan 是 `(lengths, seed, epoch, batch_size, window)` 的纯函数，结果 JSON 保存完整 plan，可从记录重放。

**重点**：**分桶省的是算力浪费，花的是样本共现随机性——窗口大小是这个交换的旋钮。**

**边界与常见误区**：窗口受"整除 Batch Size"约束，不能任意取值；收益上限由语料自然长度分布决定。`[易错]` "分桶破坏了随机性"只说对一半——全局排列仍是随机的，被牺牲的是跨窗口的成员共现多样性。

**产品对应**：HF Trainer 的 `group_by_length`、各框架的 length-grouped batching `[业界通识]`。

**方案视角**：长文本混合语料且按 Token 计费时，分桶几乎总是划算；问供应商窗口取多大、有没有报告 Padding 比例随窗口的变化曲线。

**讲给客户的一句话**：分桶用一点样本共现的随机性，换回被 Padding 吃掉的算力，窗口大小可调。

**掌握目标**：能画出 window=8、batch=4 时 16 条样本的分组过程；能判断一个窗口取值是否合法并解释退化性质。

**深挖入口**：帖子「按长度分桶」（9/28）· 仓库 chapter_19

### ch20 · Sequence Packing（训练篇 05）

**一句话定位**：把多个样本装进同一物理序列，用三条 Mask 约束锁住训练语义。

**解决什么问题**：分桶只能缩小长度差；只要一个样本独占一行，短样本仍以 Padding 填满自己那行。要让短样本互相拼满一行，必须保证它们在 Attention 里互相看不见。

**原理图解**：`<!--fig:ch20-->`

**原理要点**：
- baseline 沿用 Padding Batch（每样本一行）；variant 用贪心装包把不切分的样本装进固定容量的 Pack，三条正确性约束缺一不可。
- ① Position ID 从 0 重置：RoPE 是相对位置编码，不平移则后续样本全部位置错位。
- ② Block-diagonal 加性因果 Mask：跨样本与 Padding 位置以 `finfo(dtype).min` 屏蔽，Softmax 权重精确为 0，隔离可逐位验证；简单的 EOS 拼接只作负对照。
- ③ 边界 Loss Mask：Causal Shift 会让前一样本末 Token 预测下一样本首 Token，每个样本首个 Label 显式置 `ignore_index`。
- 代价：每步固定成本（Optimizer、Kernel 启动）随 Pack 数线性放大，单 Pack 每步可能仍慢于多行 Padding Batch。

**重点**：**Packing 把 Padding 换成有效 Token，但正确性靠位置重置、块对角 Mask、边界 Loss Mask 三条锁住。**

**边界与常见误区**：装包收益由长度分布决定，收益可能被每步固定开销吃掉；实现要求模型 forward 接受 `position_ids` 与 4D 加性 Mask。`[易错]` 只拼 Token 不做 Mask 是"看似省了槽位、实则训练语义已错"的最典型事故。

**产品对应**： packing + block-diagonal attention `[业界通识]`；如数据引擎的 sequence packing 选项与多种 Mask 实现。

**方案视角**：语料长度方差大时 Packing 收益最大；问供应商"packing 的 mask 方案是什么、有没有逐位隔离验证"，而不是只看吞吐提升。

**讲给客户的一句话**：Packing 把 Padding 槽位变成有效 Token，正确性靠三条 Mask 约束锁住，缺一条训练语义就悄悄错了。

**掌握目标**：能画出两个样本共居一个 Pack 的 Block-diagonal Mask 结构；能答出三条约束分别防止哪种错误。

**深挖入口**：帖子「Sequence Packing」（9/30）· 仓库 chapter_20

### ch21 · 离线Token Shard（训练篇 06）

**一句话定位**：把 Chat Template 渲染与 BPE 编码一次性离线做成带校验的二进制 Shard，训练只读不编。

**解决什么问题**：同一个数据集每个 Epoch 都被重复渲染和编码，这些确定性计算却占着训练数据路径。把编码搬到训练前，还得保证读回来的和在线编码逐 Token 一致、坏了能被发现。

**原理图解**：`<!--fig:ch21-->`

**原理要点**：
- baseline 在线读 JSONL、逐条渲染编码；variant 训练前一次性编码成二进制 Shard，训练时按索引随机读取并校验。
- 头部固定 100 字节：magic `MISFTSKD`、格式版本、40 字节模型 revision、40 字节数据 revision、最大长度、记录数；每条记录含 source_line、input_ids(u32)、labels(i32) 和 FNV-1a 64 校验和；尾部自描述索引表支持 O(1) 随机读。
- 读取端强制校验：magic/版本/revision 不符、校验和不匹配、索引错位、文件截断一律抛错，绝不静默回退。
- FNV-1a 不是密码学哈希：职责是检测意外损坏，不防篡改；`OnlineCorpus` 与 `ShardCorpus` 暴露同一 `materialize(index)` 接口，逐 Token 一致性可全量比对。

**重点**：**数据格式自带版本头与校验和，训练热路径只做读取——"数据可复现"落到字节。**

**边界与常见误区**：收益大小取决于在线编码成本占训练 Step 的比例，需要实测而非默认可观。`[易错]` 把 FNV-1a 当成完整性保证的全部——它不防恶意篡改，revision 绑定和大小/SHA256 校验是另一层。

**产品对应**：预 tokenize 数据格式，如 tokenized datasets / 预编码数据缓存 `[业界通识]`。

**方案视角**：多 Epoch 长训练或数据集跨机分发时，离线 Shard 几乎稳赚；一次性小实验则要权衡构建成本，判据是"在线编码时间 ÷ Step 时间"。

**讲给客户的一句话**：数据文件自带版本与校验，训练只做读取和验证——复现实验从数据格式这一层就有据可查。

**掌握目标**：能画出 Shard 的头部/记录/索引三段布局；能区分"校验和防什么"与"revision 绑定防什么"。

**深挖入口**：帖子「离线Token Shard」（10/1）· 仓库 chapter_21

### ch22 · DataLoader预取与异步H2D（训练篇 07）

**一句话定位**：Worker 线程 + Pinned 双缓冲 + 独立 CUDA Stream，把"等数据"藏进"算梯度"。

**解决什么问题**：Shard 读取、校验、组批和 Host-to-Device 拷贝仍发生在每步关键路径上，GPU 算完就闲着等下一批数据。

**原理图解**：`<!--fig:ch22-->`

**原理要点**：
- baseline 同步串行：主线程读 Shard、组批、阻塞式 H2D 到默认流，然后计算；variant 用一个 Worker 线程把未来 Batch 物化进 Pinned Memory，有界队列深度 2。
- Pinned Memory 是锁页内存，可被 DMA 直接异步拷贝；非锁页内存的 `non_blocking` 拷贝实际会退化为同步。
- 双缓冲错拍：第 k 步计算期间，在独立 CUDA Stream 上对第 k+1 步发起 `non_blocking` H2D 并记录事件；计算流 `wait_event` 等待对应拷贝事件，Pinned 源缓冲在事件完成前不复用。
- 可隐藏的上限就是同步路径的数据侧耗时——数据成本远小于训练 Step 时，端到端收益自然有限。
- Worker 异常通过哨兵在主线程显式抛出，不静默吞掉。

**重点**：**预取不是让数据变快，而是把数据侧耗时藏进计算时间窗，上限就是同步路径的数据耗时。**

**边界与常见误区**：Worker 固定 1 个，队列深度 2 限制宿主内存占用。`[易错]` 以为加了 `non_blocking=True` 就异步了——没有 Pinned 源内存和事件管理，拷贝仍阻塞在默认流上。

**产品对应**：DataLoader 的 `prefetch_factor` + `pin_memory=True` + `non_blocking` 传输 `[业界通识]`。

**方案视角**：数据侧耗时占比可以用分段计时量出来；占比小于一成时，预取不是优先级，先看计算段，判据是 CUDA Event 分段时间线。

**讲给客户的一句话**：数据预取是把"等数据"藏进"算梯度"的时间窗里，能藏多少取决于数据侧原本占多少。

**掌握目标**：能画出双缓冲的时序图（计算 k 与拷贝 k+1 重叠）；能判断一个数据流水线瓶颈在数据侧还是计算侧。

**深挖入口**：帖子「异步H2D」（10/3）· 仓库 chapter_22

### ch23 · BF16混合精度（训练篇 08）

**一句话定位**：Forward/Loss 进 `autocast(bfloat16)`，主参数、梯度与 AdamW 状态保持 FP32。

**解决什么问题**：FP32 的前向/反向由 GEMM 主导，没吃到 Tensor Core 低精度算力的红利；但直接全低精度又会动优化器的数值稳定性。

**原理图解**：`<!--fig:ch23-->`

**原理要点**：
- baseline 全 FP32；variant 的 Forward 与 Loss 在 `torch.autocast(bfloat16)` 下执行，主参数、梯度、AdamW 状态仍 FP32——这就是"混合"的含义。
- BF16 比 FP16 动态范围大，不需要 loss scale 的 GradScaler。
- 效果预期：前向/反向显著缩短（GEMM 走 BF16 Tensor Core），激活显存减半但 FP32 主状态不动，峰值显存只部分下降。
- 代价：Loss 与梯度出现 BF16 舍入级分叉并随步数累积，用轨迹误差和 dtype 普查（forward hook 逐模块记录输出 dtype、参数/梯度/优化器状态直方图）量化。
- TF32 显式关闭并记录，避免混入第三个精度变量。

**重点**：**混合精度省的是计算和激活，不省优化器状态；数值分叉是精度代价，要用轨迹误差盯住。**

**边界与常见误区**：TF32 被固定关闭，防止它悄悄替你"开"了低精度 GEMM。`[易错]` 把 FP16 的经验（必须 GradScaler、可能 NaN）直接套到 BF16 上——BF16 的动态范围与 FP32 相同，不需要 loss scale。

**产品对应**：`torch.autocast` 混合精度；GradScaler 是 FP16 方案 `[业界通识]`；训练配置里的 `bf16=True` 选项。

**方案视角**：显卡有 BF16 Tensor Core 时混合精度基本是标配；问供应商"精度策略是什么、优化器状态什么 dtype、有没有轨迹误差监控"，判据是 dtype 普查而不是口头承诺。

**讲给客户的一句话**：混合精度省的是计算和激活，不省优化器状态；数值分叉需要用轨迹误差监控来定价。

**掌握目标**：能画出一行参数在混合精度下的各段 dtype（前向激活 BF16 / 主参数与优化器 FP32）；能算出激活显存大约减半而状态显存不变的账。

**深挖入口**：帖子「BF16混合精度」（10/5）· 仓库 chapter_23

### ch24 · 梯度累积（训练篇 09）

**一句话定位**：一个 Optimizer Step 由 K 个微批的梯度累加构成，用时间换显存容量。

**解决什么问题**：想要大 Batch 的训练语义（更稳的梯度估计），单步显存却装不下那么大的批。能不能拆成 K 次小 Forward/Backward，效果还和一次大批一样？

**原理图解**：`<!--fig:ch24-->`

**原理要点**：
- baseline 一次大 Batch 完成 Forward/Backward/Step；variant 把同一批样本切成 K 个微批，各自 Forward/Backward 累加梯度，最后一次裁剪 + Step。
- 正确性全在加权规则：微批 i 的损失按 `token_sum_i / total_supervised` 缩放，累积出的梯度才与一次大 Batch 的 `token_mean` 严格同口径。
- 微批均值直接平均在监督 Token 数不均时会偏离联合目标——本章用反例显式验证这一点。
- 代价是 K 次小 GEMM 的时间加总；收益是微批级显存上限带来的容量（大批 OOM 的配置可以跑）和更少 Padding。
- FP32 下累积梯度与大 Batch 只差求和顺序噪声。

**重点**：**等价性的钥匙是每个微批按监督 Token 占比 `token_sum_i / total_supervised` 加权。**

**边界与常见误区**：与 ch31 的 `no_sync` 是分布式版的天然搭档（本期是单卡）。`[易错]` "每个微批 Loss 取均值再平均"——监督 Token 数不均时它不等于联合 `token_mean`，这不是理论洁癖，是可复现的偏差。

**产品对应**：训练配置的 `gradient_accumulation_steps` `[业界通识]`；几乎所有训练框架的标准选项。

**方案视角**：客户说"显存不够上大批"时，先看能不能用累积凑出等效 Batch；判据是目标有效 Batch 大小 = 微批尺寸 × K，以及可接受的步时间增加。

**讲给客户的一句话**：梯度累积用时间换显存容量，等价性的钥匙是每个微批按监督 Token 占比加权。

**掌握目标**：能算出给定两个微批监督 Token 数下各自应有的损失缩放系数；能判断一种累积实现是否与大 Batch 同口径。

**深挖入口**：帖子「梯度累积」（10/10）· 仓库 chapter_24

### ch25 · Activation Checkpointing（训练篇 10）

**一句话定位**：不存全部中间激活，只存每段层输入和 RNG 状态，Backward 时重算前向。

**解决什么问题**：ch24 解决了大 Batch 准入，但 autograd 为 Backward 保留的中间激活仍是显存大头——28 层 Transformer 每层的注意力分数、FFN 中间值都被扣着等反向用。

**原理图解**：`<!--fig:ch25-->`

**原理要点**：
- baseline 保存全部激活到 Backward；variant 只保存每层（段）的输入，Backward 时先恢复该段 RNG 状态（dropout 等随机算子依赖它），再 `enable_grad` 重算一次前向、然后正常反传。
- 本章手写 `RecomputeCheckpoint`（autograd Function：forward 存输入与 RNG，backward 恢复 RNG → 重算 → 传播梯度），与 `torch.utils.checkpoint` 的 `use_reentrant=False` 对照。
- dropout 正反例验证："不恢复 RNG 就逐位不一致"——RNG 状态是正确性的一部分，不只是工程细节。
- 粒度扫描（存 1/2/4/7/14/28 段）呈现权衡曲线：段越细越省显存、重算越多，代价约为 Backward 内多一次前向。

**重点**：**激活显存可以换成约一次额外前向的时间，段数粒度就是这之间的定价表。**

**边界与常见误区**：只存层输入的重算方案，不涉及按算子的选择性 checkpoint。`[易错]` "checkpoint 就是无损免费"——时间代价恒在；以及忘存 RNG 导致含 dropout 的重算逐位漂移。

**产品对应**：gradient checkpointing，HF Trainer 的 `gradient_checkpointing=True`、FSDP 内建 `[业界通识]`。

**方案视角**：显存不够、又不想减 Batch 时第一顺位开关；判据是客户能接受的步时间增幅（约 +1 次前向）与段数设置的匹配。

**讲给客户的一句话**：激活显存可以换成约一次额外前向的时间，段数粒度就是这之间的定价表。

**掌握目标**：能画出"存边界→重算→反传"的时序；能算出 28 层模型在段数为 4 时保存与重算的相对量。

**深挖入口**：帖子「Activation Checkpointing」（10/10）· 仓库 chapter_25

### ch26 · 训练版FlashAttention（训练篇 11）

**一句话定位**：用 SDPA 分块后端替代物化 S² 矩阵的 Eager Attention，把注意力显存从平方降下来。

**解决什么问题**：Eager Attention 会物化完整的 S² score/probability 矩阵，序列一长显存先爆；FlashAttention 是答案，但它能不能用还跟精度绑定。

**原理图解**：`<!--fig:ch26-->`

**原理要点**：
- baseline 的 Eager Attention 物化 S² 矩阵；variant 换成 `scaled_dot_product_attention` 的分块计算后端，不物化 S²。
- 后端普查（`probe_backend_support`）：FP32 下 flash 后端不可用（要求 BF16/FP16），可用的是 mem_efficient 后端；sdpa_math 作为数值对照。
- 512 长度下端到端收益不大，机制价值在显存与序列容量边界——序列容量扫描显示能装下的最大长度右移。
- 数值差异对照：不同后端有舍入级差异；Attention 实现以注入点方式接入模型，None 时走与之前逐位相同的内联路径。

**重点**：**FlashAttention 的训练收益首先是"能不能装下长序列"，其次才是速度——且后端可用性与精度绑定。**

**边界与常见误区**：BF16 只出现在明确标注的跨精度对照臂，不与本机制混淆。`[易错]` 在 FP32 训练里期待 flash 后端——它要求半精度输入，FP32 下只有 mem_efficient 可用。

**产品对应**：FlashAttention 系列、PyTorch SDPA（`torch.nn.functional.scaled_dot_product_attention`）`[业界通识]`。

**方案视角**：客户序列长度超过 2–4K 时注意力显存成为瓶颈，分块后端是刚需；判据是目标序列长度下 S² 矩阵的显存账，而不是短序列 benchmark。

**讲给客户的一句话**：FlashAttention 训练版的入门门槛是"能不能装下长序列"，而且后端可用性跟精度绑定，要按 dtype 查支持矩阵。

**掌握目标**：能画出 Eager 物化 S² 与分块计算两路径的显存对比；能答出 FP32 与 BF16 下各自可用的 SDPA 后端。

**深挖入口**：帖子「训练版FlashAttention」（10/12）· 仓库 chapter_26

### ch27 · Fused AdamW（训练篇 12）

**一句话定位**：AdamW 数学不变，只把参数更新的 Kernel 从逐 Tensor 循环换成 foreach/fused，省发射次数。

**解决什么问题**：0.6B 模型有约 310 个参数 Tensor，优化器一步要为每个 Tensor 发射一串小 Kernel，逐元素算子的 Kernel 启动开销积少成多，Optimizer 段的时间不好看。

**原理图解**：`<!--fig:ch27-->`

**原理要点**：
- 三种 Kernel 组织对照：loop（逐 Tensor 循环）、foreach（多 Tensor 打包发射）、fused（单 Kernel 完成整个更新）——数学公式完全相同。
- 用 profiler 统计一次 step 的 CUDA Kernel 数和设备时间，配合手写参照实现（含解耦权重衰减与偏置校正）验证三臂数值一致。
- 方法论核心：算子不变，只改 Kernel 组织，开销差异就能被干净归因到"发射次数"。
- `clip_grad_norm_` 三臂走同一路径并单独计时，作为 step 分段计时的锚点。

**重点**：**算子数学不变，只改 Kernel 组织，Optimizer 段的开销就能被干净归因和压缩。**

**边界与常见误区**：不涉及优化器状态 dtype 与分片——那是 ZeRO-1（ch32）的事，两者维度正交。`[易错]` 把 fused optimizer 和"更快的收敛"混为一谈——它不改任何数学，只改执行方式。

**产品对应**：`torch.optim.AdamW(fused=True)`、apex fused optimizer `[业界通识]`；框架里 `optim="adamw_torch_fused"` 一类开关。

**方案视角**：参数 Tensor 多、模型小、步时间短的场景里 Optimizer 份额才显眼；判据是 profiler 分段里 optimizer 段占比，而不是无差别上 fused。

**讲给客户的一句话**：小 Tensor 一多，优化器也能吃掉可观的步时间——省 Kernel 发射次数是零数学风险的提速。

**掌握目标**：能区分 loop/foreach/fused 三种 Kernel 组织并画出它们的发射模式；能判断一个优化器加速 claim 是否涉及数学变化。

**深挖入口**：帖子「Fused AdamW」（10/13）· 仓库 chapter_27

### ch28 · torch.compile（训练篇 13）

**一句话定位**：把 Forward/Backward 交给 Inductor 编译融合，用首次编译成本换稳态 Kernel 收益。

**解决什么问题**：Eager 执行下逐元素算子链有大量 Kernel 启动和内存读写开销，编译能融合它们；但编译不是免费的，账要分三笔记。

**原理图解**：`<!--fig:ch28-->`

**原理要点**：
- baseline 是 Eager 执行；variant 的 Forward/Backward 交给 `torch.compile`（Inductor 默认配置），Loss、梯度裁剪、优化器留在 Eager。
- 三笔必须分开记的账：稳态收益（Inductor 只能融合逐元素链，收益受 GEMM 份额限制）、首次编译成本（Break-even 摊销）、变长序列的重编译——每来一个新 shape 就触发一次编译。
- `dynamic=True` 用动态 shape 消化重编译，但可能牺牲部分优化；用 dynamo 统计图数、重编译次数与 graph break。
- 变长序列经 `pad_batch_to` 固定到 512 形状（不触碰真实 Token 与 Labels），正是为规避重编译风暴的工程手段。

**重点**：**编译的账是"首次编译成本 vs 稳态收益"，变长输入会不断触发重编译。**

**边界与常见误区**：编译优化器/裁剪、max-autotune、CUDA Graph 均不混入本期。`[易错]` 拿"首个 Step"测速——那包含编译时间；稳态收益要在编译完成后测，变长训练的主要风险就是重编译风暴。

**产品对应**：`torch.compile`/Inductor `[业界通识]`；训练服务的 shape 策略（固定长度/桶化）是它的配套工程。

**方案视角**：训练 shape 固定、长跑的生产任务是编译的好场景；短任务或 shape 多变时先算 Break-even。判据是"编译耗时 ÷ 稳态每步节省 = 多少步回本"，以及输入 shape 的分布。

**讲给客户的一句话**：编译的账是"首次编译成本 vs 稳态收益"，变长输入会不断触发重编译——生产里要先固定 shape 策略。

**掌握目标**：能画出 Eager 与编译路径的首步/稳态时间对比；能判断一个训练任务是否会被重编译拖累。

**深挖入口**：帖子「torch.compile」（10/13）· 仓库 chapter_28

### ch29 · DDP（训练篇 14）

**一句话定位**：两卡各持全量模型，各算各的分片，Backward 后一次 AllReduce SUM 合成全局梯度。

**解决什么问题**：单卡机制已被前几章压到边界，吞吐还要翻倍就得打开第一个并行维度——数据并行。但加卡不能悄悄改变训练语义，通信也会成为新瓶颈。

**原理图解**：`<!--fig:ch29-->`

**原理要点**：
- baseline 单卡算全局 Batch；variant 两张卡各持一份完整模型、完整梯度和完整优化器状态，各处理全局批次的一个分片（成员按 `rank::world_size` 切片）。
- 等价性核心：本地损失按 `supervised_local / supervised_global` 缩放后，Backward 结束做一次 AllReduce SUM（各卡梯度求和并广播一致），与单卡大 Batch 的 `token_mean` 数学等价。
- AllReduce 通信量为 `2(W-1)/W·D`；W=2 时每张卡每步发出约等量的 FP32 梯度数据。
- 两卡走 PCIe 无 NVLink 时，每步约 2.4 GB 的 FP32 梯度 AllReduce 完全暴露在关键路径上，Strong Scaling 效率显著低于 2 是预期内的。
- 正确性对照在 AllReduce 之后、裁剪之前捕获梯度（`after_allreduce` 钩子）；比较时点错了会看到恒定偏差，那是比较错误不是梯度错误。

**重点**：**本地损失按监督 Token 占比缩放 + Backward 后一次 AllReduce SUM，就是数据并行的全部语义等价性。**

**边界与常见误区**：本期不使用 Bucket 重叠（ch30）、`no_sync`（ch31）和状态分片（ch32/33）——通信严格排在 Backward 之后全额暴露，这正是后三章要修的。`[易错]` "多卡自动加速一倍"——同步点暴露时远达不到，判据是 Strong Scaling 效率。

**产品对应**：`torch.nn.parallel.DistributedDataParallel`、NCCL `[业界通识]`；所有训练框架多卡模式的地基。

**方案视角**：数据并行适合模型单卡装得下、想扩吞吐的场景；问"通信和计算重叠吗、互联是什么（PCIe/NVLink/IB）"，判据是每步通信字节数除以互联带宽。

**讲给客户的一句话**：数据并行的语义等价性很好证明，但后置的全量梯度同步在慢互联上会全额暴露——通信时机是下一个优化变量。

**掌握目标**：能画出两卡 DDP 一步的时序（前向→反向→AllReduce→更新）；能算出 W=2 时 AllReduce 的通信量 `2(W-1)/W·D`。

**深挖入口**：帖子「DDP」（10/14）· 仓库 chapter_29

