# 训练篇（ch16–ch43）：从单卡 SFT 到三维并行

主线是 `Qwen/Qwen3-0.6B-Base` 全参数 SFT：一章只引入一个变量，每章都有 baseline/variant 两条路径、正确性检查与 Benchmark 入口。单卡章节与推理篇同一类 12GB 显卡，多卡覆盖单机 2/4 卡与双节点 4+4；多卡聚合显存不是统一显存，性能还受 PCIe、NUMA、NCCL 与节点间网络约束。仓库只提供机制实现与复现入口，不附带可外推的实验结论。

### ch16 · 手写Qwen3-0.6B全参数SFT（训练篇 01）

**解决什么问题**：不借助 Transformers/Trainer，能否把一次全参数 SFT 从数据一直跑到可恢复的 Checkpoint？这是后续所有实验的原始 baseline。

**机制通俗版**：Python + PyTorch 手写 Byte-level BPE Tokenizer（直接解析 `tokenizer.json`）、RMSNorm/RoPE/GQA/SwiGLU 模型、Causal LM Loss（Shifted Cross Entropy，位置 t 预测下一 Token）和训练循环：Forward → Backward → 梯度裁剪 → AdamW → Scheduler，外加完整 Checkpoint 保存/恢复。baseline 配置刻意保守：单卡、FP32、Micro-batch 1、固定序列长度、Eager Attention，且除 Padding 外的完整对话 Token 都参与 Loss。

**架构上的位置**：训练篇的地基，ch17 起的每一章都在它上面换一个变量。正确性证据链包括手算 Loss、小模型有限差分、恢复后复现同一个下一 Step。边界：不追求效率，任何优化结论都必须先有这条等价基线。

**业界对应**：相当于把 HF Trainer/accelerate 内部拆开看一遍 `[业界通识]`。

**讲给客户的一句话**：先跑通一条数学上可验证的最小训练链路，后面的每一次优化都有等价性对照可查。

**深挖入口**：帖子「手写Qwen3-0.6B全参数SFT」（9/20）

### ch17 · Assistant-only Loss（训练篇 02）

**解决什么问题**：baseline 让 System、User、Assistant 都成了预测目标；但 SFT 真正想教模型的是"怎么回答"，不是"怎么提问"。

**机制通俗版**：模型、数据、输入全部不动，唯一变量是 `labels` 里哪些位置设为 `ignore_index=-100`。baseline 监督除 Padding 外的完整对话；variant 只监督 Assistant 正文、Tool Call 文本和真实的 `<|im_end|>`，`<|im_start|>assistant\n` 头部、System/User/Tool Response、消息间换行全部屏蔽。因为监督 Token 集合缩小，Loss 数值、梯度方向和一次参数更新都会改变——这不是 bug，是目标变了。

**架构上的位置**：依赖 ch16 的 Loss 定义。关键边界：模板边界来自固定模型 revision 的 `tokenizer_config.json`，无法无损表示的字段（如 `reasoning_content`）直接报错而不是静默吞掉。

**业界对应**：训练框架里的 loss mask / chat template 概念 `[业界通识]`。

**讲给客户的一句话**：改的不是模型而是监督信号——同一份数据，mask 设计直接决定模型学到"回答"还是"复述"。

**深挖入口**：帖子「Assistant-only Loss」（9/21）

### ch18 · 固定Mini-batch（训练篇 03）

**解决什么问题**：ch17 前一条更新内是逐样本 Forward/Backward 累加梯度；把多个变长样本合成一个二维 Batch 一次算，能否保持训练目标又提高吞吐？

**机制通俗版**：baseline 是多个 `[1, T_i]` 调用；variant 右 Padding 到组内最大长度后一次 `[B, max(T_i)]` 调用。正确性关键在 Loss 归约：`token_mean`（所有有效 Token 等权）与 `sample_mean`（先逐样本平均再等权）在变长下不是同一个目标，逐样本路径必须按 `n_i / sum(n_i)` 加权 backward，两种权重不能混用，否则 Batch 化前后暗中改了训练目标。代价：物理 Token 从 `sum(T_i)` 变成 `B×max(T_i)`，Padding 槽位和激活显存都增加。

**架构上的位置**：依赖 ch17 的 Assistant-only Labels。边界：批量计算更快但有 Padding 浪费——这正是 ch19 的引子。

**业界对应**：DataLoader 的 collate/padding `[业界通识]`。

**讲给客户的一句话**：合 Batch 不是免费提速，是用 Padding 槽位换批量矩阵计算，账要算监督 Token 的吞吐。

**深挖入口**：帖子「固定Mini-batch」（9/23）

### ch19 · 按长度分桶（训练篇 04）

**解决什么问题**：ch18 的 Padding Batch 里，长短样本混在一组时短样本浪费大量槽位。

**机制通俗版**：唯一变量是 Batch Sampler。baseline 用固定种子全局 Shuffle 后顺序切块；variant 用同一随机排列切成有限 Shuffle Window，窗口内按编码长度稳定排序再切块，最后打乱 Batch 顺序。窗口越大 Batch 内长度越接近、Padding 物理槽位越少，但 Batch 成员越来越由长度决定——用 `companion_repeat_ratio`（跨 Epoch 共现重复率）度量，全排序还可能带来训练顺序损失。这是"减少浪费 vs 打乱随机性"的显式交换。

**架构上的位置**：依赖 ch18 的 Padding 组批与 `token_mean`。关键边界：窗口必须不小于且整除 Batch Size；`window == batch_size` 时退化为随机切块，这个退化性质本身就是正确性对照。plan 是纯函数，可从 JSON 重放。

**业界对应**：length-grouped batching `[业界通识]`。

**讲给客户的一句话**：分桶省的是算力浪费，花的是样本共现的随机性——窗口大小是这个交换的旋钮。

**深挖入口**：帖子「按长度分桶」（9/28）

### ch20 · Sequence Packing（训练篇 05）

**解决什么问题**：分桶只能缩小长度差；只要一个样本独占一行，短样本仍以 Padding 填满自己那行。

**机制通俗版**：variant 把多个不切分的样本装进一个固定容量的物理序列（Pack），必须同时满足三件事：① Position ID 从 0 重置（RoPE 是相对位置，不平移则错位）；② Block-diagonal 加性因果 Mask——用 `finfo(dtype).min` 屏蔽跨样本与 Padding 位置，使 Softmax 权重精确为 0，隔离可逐位验证；③ 边界 Loss Mask——Causal Shift 会让前一样本末 Token 预测下一样本首 Token，样本首 Label 必须显式置 `ignore_index`。EOS 拼接但不做 Mask 只作为负对照。代价：每步固定成本（Optimizer、Kernel 启动）随 Pack 数线性放大。

**架构上的位置**：依赖 ch19；实现要求模型 forward 接受可选 `position_ids` 与 4D 加性 Mask。边界：装包收益由长度分布决定，收益可能被固定开销吃掉。

**业界对应**：packing + block-diagonal attention `[业界通识]`。

**讲给客户的一句话**：Packing 把 Padding 的槽位变成有效 Token，但正确性靠三条 Mask 约束锁住，缺一训练语义就悄悄错了。

**深挖入口**：帖子「Sequence Packing」（9/30）

### ch21 · 离线Token Shard（训练篇 06）

**解决什么问题**：Chat Template 渲染和 BPE 编码还发生在训练数据路径上，每步都在重复算同一份结果。

**机制通俗版**：唯一变量是 Tokenize 的执行阶段。baseline 在线读 JSONL、逐条渲染编码；variant 训练前一次性编码成二进制 Shard：头部固定 100 字节（magic、格式版本、40 字节模型 revision、40 字节数据 revision、最大长度、记录数），每条记录带 source_line、input_ids、labels 和 FNV-1a 64 校验和，尾部自描述索引表支持 O(1) 随机读。读取端强制校验：magic/版本/revision 不符、校验和不匹配、文件截断一律抛错，绝不静默回退。FNV-1a 不是密码学哈希，只防意外损坏不防篡改。

**架构上的位置**：依赖 ch20 的数据流。边界：收益大小取决于在线编码成本占训练 Step 的比例，需要实测。

**业界对应**：预 tokenize 数据格式，如 tokenized datasets `[业界通识]`。

**讲给客户的一句话**：数据格式自带版本与校验和，训练热路径只做读取——这是"数据可复现"落到字节的工程。

**深挖入口**：帖子「离线Token Shard」（10/1）

### ch22 · DataLoader预取与异步H2D（训练篇 07）

**解决什么问题**：Shard 读取、校验、组批和 Host-to-Device 拷贝仍在每步关键路径上，GPU 在等数据。

**机制通俗版**：baseline 同步串行；variant 用一个 Worker 线程把未来 Batch 物化进 Pinned Memory（锁页内存，可被 DMA 直接异步拷贝），有界队列深度 2，双缓冲错拍：第 k 步计算期间，在独立 CUDA Stream 上对第 k+1 步发起 `non_blocking` H2D 并记录事件，计算流 `wait_event` 等待对应拷贝事件，Pinned 源缓冲在事件完成前不复用。可隐藏的上限就是同步路径的数据侧耗时——数据成本远小于训练 Step 时收益自然有限。

**架构上的位置**：依赖 ch21 的 Shard。边界：Worker 固定 1 个，队列深度限制宿主内存；Worker 异常经哨兵在主线程显式抛出。

**业界对应**：DataLoader prefetch + pin_memory + non_blocking `to(device)` `[业界通识]`。

**讲给客户的一句话**：数据预取不是让数据变快，而是把"等数据"藏进"算梯度"的时间窗里。

**深挖入口**：帖子「异步H2D」（10/3）

### ch23 · BF16混合精度（训练篇 08）

**解决什么问题**：FP32 的前向/反向由 GEMM 主导，没吃到 Tensor Core 低精度算力的红利。

**机制通俗版**：唯一变量是 Forward 与 Loss 放进 `torch.autocast(bfloat16)` 执行；主参数、梯度、AdamW 状态仍保持 FP32——这就是"混合"的含义。BF16 比 FP16 动态范围大，不需要 loss scale 的 GradScaler。效果预期：前向/反向显著缩短，激活显存减半但 FP32 主状态不动，峰值显存只部分下降；Loss 与梯度出现 BF16 舍入级分叉并随步数累积，用轨迹误差和 dtype 普查（forward hook 逐模块记录输出 dtype）量化。

**架构上的位置**：依赖 ch22 的稳定数据流水。边界：TF32 显式关闭并记录，避免混入第三个精度变量。

**业界对应**：autocast 混合精度；GradScaler 是 FP16 方案 `[业界通识]`。

**讲给客户的一句话**：混合精度省的是计算和激活，不省优化器状态；数值分叉是精度代价，要用轨迹误差盯住。

**深挖入口**：帖子「BF16混合精度」（10/5）

### ch24 · 梯度累积（训练篇 09）

**解决什么问题**：单步能容纳的 Batch 受显存约束，想要大 Batch 的训练语义却没有大 Batch 的显存。

**机制通俗版**：一个 Optimizer Step 由 K 个 micro-batch（一次 Forward/Backward 的最小批）的梯度累加构成。正确性全在一个加权规则：微批 i 的损失按 `token_sum_i / total_supervised` 缩放，累积出的梯度才与一次大 Batch 的 `token_mean` 严格同口径——微批均值直接平均在监督 Token 数不均时会偏离联合目标，本章用反例显式验证这一点。代价是 K 次小 GEMM 的时间加总，收益是微批级显存上限带来的容量和更少 Padding。

**架构上的位置**：依赖 ch23。边界：FP32 下与大 Batch 只差求和顺序噪声；与 ch31 的 no_sync 是分布式版的天然搭档。

**业界对应**：gradient accumulation `[业界通识]`。

**讲给客户的一句话**：梯度累积用时间换显存容量，等价性的钥匙是每个微批按监督 Token 占比加权。

**深挖入口**：帖子「梯度累积」（10/10）

### ch25 · Activation Checkpointing（训练篇 10）

**解决什么问题**：ch24 解决了大 Batch 准入，但 autograd 为 Backward 保留的中间激活仍是显存大头。

**机制通俗版**：不再保存全部激活，只保存每层（段）的输入，Backward 时先恢复该段的 RNG 状态（dropout 等随机算子依赖它），再 `enable_grad` 重算一次前向、然后正常反传。本章手写 `RecomputeCheckpoint`（autograd Function），与 `torch.utils.checkpoint` 的 `use_reentrant=False` 对照；dropout 正反例验证"不恢复 RNG 就逐位不一致"。粒度扫描（存 1/2/4/7/14/28 段）呈现权衡曲线：段越细越省显存、重算越多，代价约为 Backward 内多一次前向。

**架构上的位置**：依赖 ch24。边界：只存层输入的重算方案，不涉及按算子的选择性 checkpoint。

**业界对应**：gradient checkpointing，FSDP 内建 `[业界通识]`。

**讲给客户的一句话**：激活显存可以换成约一次额外前向的时间，段数粒度就是这之间的定价表。

**深挖入口**：帖子「Activation Checkpointing」（10/10）

### ch26 · 训练版FlashAttention（训练篇 11）

**解决什么问题**：Eager Attention 会物化完整的 S² score/probability 矩阵，序列一长显存先爆。

**机制通俗版**：换成 `scaled_dot_product_attention` 的分块计算后端，不物化 S²。关键认知：FP32 下 flash 后端不可用（要求 BF16/FP16），可用的是 mem_efficient 后端；512 长度下端到端收益不大，机制价值在显存与序列边界。本章做后端普查（probe_backend_support）、数值差异对照和序列容量扫描，Attention 实现以注入点方式接入模型。

**架构上的位置**：依赖 ch25。边界：BF16 只出现在明确标注的跨精度对照臂，不与本机制混淆。

**业界对应**：FlashAttention 系列、PyTorch SDPA `[业界通识]`。

**讲给客户的一句话**：FlashAttention 的训练收益首先是"能不能装下长序列"，其次才是速度——而且后端可用性跟精度绑定。

**深挖入口**：帖子「训练版FlashAttention」（10/12）

### ch27 · Fused AdamW（训练篇 12）

**解决什么问题**：AdamW 数学公式没变，但 0.6B 模型有约 310 个参数 Tensor，逐 Tensor 循环更新的 Kernel 发射次数成了 Optimizer 段的主要开销。

**机制通俗版**：三种 Kernel 组织对照：loop（逐 Tensor）、foreach（多 Tensor 打包发射）、fused（单 Kernel 完成整个更新）。用 profiler 统计一次 step 的 CUDA Kernel 数和设备时间，配合手写参照实现（含解耦权重衰减与偏置校正）验证三臂数值一致。核心方法论：算子不变，只改 Kernel 组织，开销差异就能被干净地归因。

**架构上的位置**：依赖 ch26 的稳定步体；`clip_grad_norm_` 三臂同路径并单独计时作锚点。边界：不涉及优化器状态 dtype 与分片（那是 ZeRO-1 的事）。

**业界对应**：`torch.optim.AdamW(fused=True)`、apex fused optimizer `[业界通识]`。

**讲给客户的一句话**：小 Tensor 一多，优化器也能吃掉可观的步时间——省 Kernel 发射次数是零数学风险的提速。

**深挖入口**：帖子「Fused AdamW」（10/13）

### ch28 · torch.compile（训练篇 13）

**解决什么问题**：Eager 执行下逐元素算子链有大量 Kernel 启动和内存读写开销，编译能融合，但代价要先算清。

**机制通俗版**：唯一变量是 Forward/Backward 交给 `torch.compile`（Inductor 默认配置），Loss、裁剪、优化器留在 Eager。三个必须分开记的账：稳态收益（Inductor 只能融合逐元素链，受 GEMM 份额限制）、首次编译成本（Break-even 摊销）、变长序列的重编译——每来一个新 shape 就触发一次编译，`dynamic=True` 用动态 shape 消化重编译但可能牺牲部分优化。用 dynamo 统计图数、重编译与 graph break。

**架构上的位置**：依赖 ch27。边界：变长训练的主要风险就是重编译风暴；编译优化器/裁剪、max-autotune 均不混入。

**业界对应**：`torch.compile`/Inductor `[业界通识]`。

**讲给客户的一句话**：编译的账是"首次编译成本 vs 稳态收益"，变长输入会不断触发重编译——生产里要先固定 shape 策略。

**深挖入口**：帖子「torch.compile」（10/13）

### ch29 · DDP（训练篇 14）

**解决什么问题**：单卡机制已被压到边界，训练的第一个并行维度——数据并行——还没打开。

**机制通俗版**：DDP（Distributed Data Parallel，数据并行）让每张卡持一份完整模型、完整梯度和完整优化器状态，各处理全局批次的一个分片。等价性核心：本地损失按 `supervised_local / supervised_global` 缩放后，Backward 结束做一次 AllReduce SUM（把各卡梯度求和广播为一致）就与单卡大 Batch 的 token_mean 数学等价。AllReduce（全归约）的通信量公式为 `2(W-1)/W·D`。两卡走 PCIe 无 NVLink 时，每步约 2.4 GB 的 FP32 梯度 AllReduce 完全暴露在关键路径上。

**架构上的位置**：最低 2 卡。这是分布式篇章的起点，ch30–33 分别修它的通信时机与状态冗余。边界：Strong Scaling 效率显著低于 2 是预期内的。

**业界对应**：`torch.nn.parallel.DistributedDataParallel`、NCCL `[业界通识]`。

**讲给客户的一句话**：数据并行的语义等价性很好证明，但后置的全量梯度同步在慢互联上会全额暴露——通信时机是下一章的变量。

**深挖入口**：帖子「DDP」（10/14）

### ch30 · DDP Gradient Bucket与通信计算重叠（训练篇 15）

**解决什么问题**：ch29 的 AllReduce 严格排在 Backward 结束后，整段通信时间都暴露；而反向传播是逐层完成的，早算完的层梯度本可以早发。

**机制通俗版**：把参数按反向就绪顺序（参数逆序）装成若干 Bucket（桶），用 hook 监听：某桶梯度一齐就绪就立刻 `async_op=True` 发起 AllReduce，与后续 Backward 计算重叠，尾部再 `wait` 把梯度以视图挂回参数。通信数据总量与 ch29 完全相同，变的只是启动时机。桶是双刃剑：桶太小则固定延迟占比上升反而变慢，桶太大退化为全暴露形态——用 1/16/128/1024 MiB 四档实测。

**架构上的位置**：最低 2 卡，依赖 ch29。边界：PCIe PHB 无 NVLink、通信计算共享带宽，收益可能有限。

**业界对应**：真实 DDP 的 bucket 机制、`gradient_as_bucket_view` `[业界通识]`。

**讲给客户的一句话**：把一次大同步拆成若干小桶提前发，通信就藏进了计算——桶大小决定藏得多少。

**深挖入口**：帖子未发，读仓库 README

### ch31 · 梯度累积期间延迟同步（训练篇 16）

**解决什么问题**：ch30 重叠了单次 Backward 内的通信；当一步由 K 个微步累积时，每个微步都全量同步一次是纯浪费——中间结果反正不用于更新。

**机制通俗版**：前 K−1 个微步 `no_sync`（纯本地累积梯度），最后一个微步结束后统一 AllReduce SUM。损失按监督 Token 占比缩放的前提下，两种模式最终梯度数学等价，通信量降为 1/K。关键陷阱："每微步同步"的正确实现必须用独立累加器——对已归约的累积缓冲再次 AllReduce 会把先前微步的贡献放大 World Size 倍。

**架构上的位置**：最低 2 卡，依赖 ch24 的加权规则；本期刻意不用 Bucket（那是 ch30 的变量），AllReduce 仍是 Backward 后平铺 SUM。边界：K=2 时省一次全量 AllReduce。

**业界对应**：DDP 的 `no_sync()` 上下文 `[业界通识]`。

**讲给客户的一句话**：梯度累积和延迟同步是天然搭档——K 个微步只同步一次，通信量直接除以 K，数学一点不变。

**深挖入口**：帖子未发，读仓库 README

### ch32 · ZeRO-1切分优化器状态（训练篇 17）

**解决什么问题**：DDP 每卡冗余持有完整模型、梯度、优化器状态；0.6B 在 12GB 上，AdamW 的 FP32 状态已是容量压力的大头。

**机制通俗版**：ZeRO-1（Zero Redundancy Optimizer 第一级，DeepSpeed 提出 `[业界通识]`）把 AdamW 动量（exp_avg/exp_avg_sq）在 Data Parallel Rank 间连续分片，参数和梯度仍是完整副本。AdamW 是逐元素更新，"本 Rank 的状态分片 × 本 Rank 的参数分片 × 本 Rank 的梯度分片"与全量 AdamW 逐位一致；更新只发生在参数分片所有者上，更新后用分块 AllGather（128MB 块缓冲）把新参数同步给其他 Rank。

**架构上的位置**：最低 2 卡，依赖 ch29 的梯度路径。省什么省多少：**省的是优化器状态显存，W 卡时从每卡 14N 字节降到 14N/W，量级是"状态除以卡数"；代价是每步新增约 1× 参数字节的 AllGather，DDP 本来没有这笔通信。** 边界：准入边界应比复制式 DDP 更晚 OOM。

**业界对应**：DeepSpeed ZeRO stage 1 `[业界通识]`。

**讲给客户的一句话**：先把最占显存的优化器状态按卡数切开，多付一笔参数同步的通信——这是显存换通信的第一步。

**深挖入口**：帖子未发，读仓库 README

### ch33 · ZeRO-2切分梯度（训练篇 18）

**解决什么问题**：ch32 切了状态，但每卡仍留完整梯度——Backward 结束那一刻它必须存在，之后只有本 Rank 的那一份会被用到。

**机制通俗版**：把通信算子从 AllReduce 换成 Reduce-Scatter（归约并分散：各卡得到求和结果的一个分片），完成后立即释放完整梯度，只留本 Rank 的分片。W=2 时 Reduce-Scatter 与 AllReduce 的每链接数据量同为 `2(W-1)/W·D`，通信时间应接近；梯度显存再降为 1/W。连带改造：梯度分片后 `clip_grad_norm_` 不可用，全局范数须由"各 Rank 分片局部范数平方和 → 一次标量 AllReduce"合成再缩放本地分片。

**架构上的位置**：最低 2 卡，依赖 ch32（状态分片沿用 ZeRO-1，参数仍是完整副本，所以是 ZeRO-2 不是 ZeRO-3）。省什么省多少：**在 ZeRO-1 基础上再省梯度显存，从每卡 2N 降到约 2N/W。** 边界：本期逐参数发起 Reduce-Scatter，真实 ZeRO 用扁平桶，代价如实测量不掩盖。

**业界对应**：DeepSpeed ZeRO stage 2 `[业界通识]`。

**讲给客户的一句话**：AllReduce 换 Reduce-Scatter，通信几乎不涨、梯度显存再除以卡数——但梯度裁剪要自己重建全局范数。

**深挖入口**：帖子未发，读仓库 README

### ch34 · ZeRO-3/FSDP（训练篇 19）

**解决什么问题**：状态和梯度都切了，参数仍整份常驻——它只在 Forward/Backward 里被读取，却在两个 Step 之间占着显存。

**机制通俗版**：ZeRO-3/FSDP（Fully Sharded Data Parallel，全分片数据并行）让参数也只以 Shard（分片）形式常驻，于是参数有了完整生命周期：分片常驻 → Forward 前 AllGather（聚合出全量）→ 计算 → Reshard（释放全量存储）→ Backward 前按需再 AllGather → 梯度 Reduce-Scatter → 分片 Optimizer Step。下一个 Forward 的 AllGather 本身就是参数刷新，ZeRO-1/2 那次 post-step AllGather 被取消。通信从"一步一次"搬进了 Forward/Backward 热路径。

**架构上的位置**：最低 2 卡，参考 4 卡与双节点 4+4。省什么省多少：**每卡常驻模型状态从 ZeRO-2 的 `2N+14N/W` 降到 `16N/W` 字节（FP32 账本）——小 World Size 下参数冗余是最后一块大头。** 代价：AllGather 进热路径，0.6B 上很可能 Step Time 变长，负优化如实保留。

**业界对应**：`torch.distributed.fsdp` FULL_SHARD、DeepSpeed ZeRO-3、Megatron-LM 均为此形态 `[业界通识]`。

**讲给客户的一句话**：参数也切开后显存账本最好看，但通信进了热路径——模型越大越划算，小模型是负优化。

**深挖入口**：帖子未发，读仓库 README

### ch35 · MLP Tensor Parallel（训练篇 20）

**解决什么问题**：ZeRO/FSDP 切的是"层与层之间"的状态，单层内部的矩阵乘和权重还是每卡完整——模型宽到单层放不下时需要层内切分。

**机制通俗版**：Tensor Parallel 把 SwiGLU MLP 的权重切开：Gate/Up 沿输出维切（Column Parallel，输入复制输出分片），Down 沿输入维切（Row Parallel，输入分片、输出 AllReduce 求和）。梯度所有权分三类：分片权重梯度纯本地不通信；残差流输入梯度必须每层一次 backward AllReduce（Megatron 的 f 算子），漏掉会让"部分梯度"静默流进 Attention 路径；复制参数（Attention/LN/Embedding）梯度在 TP 组内天然一致，再 AllReduce 是放大 tp_size 倍的错误不是浪费。

**架构上的位置**：最低 2 卡，TP 组内数据复制，每 Rank 处理相同 Batch。边界：0.6B 单卡本可完整常驻，TP 是预期内的可能负优化——课程如实保留。

**业界对应**：Megatron-LM 的 Column/Row Parallel `[业界通识]`。

**讲给客户的一句话**：层内切分是"模型大到单层放不下"的工具，代价是每层两次强制同步的通信。

**深挖入口**：帖子未发，读仓库 README

### ch36 · Attention Tensor Parallel（训练篇 21）

**解决什么问题**：MLP 切了，Attention 投影与 Head 仍每卡完整。

**机制通俗版**：把 Q/KV Head 按 TP Rank 整数切分。Qwen3-0.6B 是 GQA（分组查询注意力：32 个 Q Head 共享 8 个 KV Head，4:1），切分约束是每 Rank 拿到整数个完整 Head：TP=2 各得 16 Q/4 KV，TP=4 各得 8 Q/2 KV，局部 GQA 组数不变（4）；`head_dim` 永不切。RoPE 是 Head 内固定旋转，局部重放与全局逐位等价。陷阱：q_norm/k_norm 权重是复制的，但每个 Rank 只累积自己 Head 的梯度项——部分和各 Rank 不同，必须 hook 一次 backward AllReduce，否则副本从第一步起静默分叉。

**架构上的位置**：最低 2 卡，在 ch35 的 MLP TP 之上叠加。边界：0.6B 的 Attention 占比小，预期收益有限甚至为负。

**业界对应**：Megatron-LM 的 Attention TP、GQA 分组切分 `[业界通识]`。

**讲给客户的一句话**：按 Head 切 Attention 的前提是整除约束，GQA 下 Q 和 KV 头要成组切；归一化权重的部分梯度同步是最容易踩的坑。

**深挖入口**：帖子未发，读仓库 README

### ch37 · Vocab Parallel Embedding与分布式CE（训练篇 22）

**解决什么问题**：权重都切了，最后一个大头是词表：完整 Logits 是 TP 模型里最大的单项激活——batch 2 × seq 512 时约 622 MiB。

**机制通俗版**：Embedding/LM Head 沿词表维切（词表 151936，TP=2 每 Rank 75968 行，tie 共享同一分片 Parameter），配合不物化完整 Logits 的分布式 Cross Entropy：log-sum-exp 可按词表分块——全局 max 用 AllReduce MAX、全局 exp 和用 AllReduce SUM、目标 token 的 logit 按所属 Rank 用一次 SUM 取回；backward（softmax−onehot）只需要本 Rank 分片和三个全局标量，CE 本身 backward 零通信。注意 LM Head 输入梯度 `dlogits_r @ W_r` 是部分和，仍需一次 backward AllReduce（f 算子）。

**架构上的位置**：最低 2 卡，在 ch36 布局之上。边界：tp=2 下词表切分收益可能仍为负，如实保留。

**业界对应**：Megatron-LM 的 Vocab Parallel Embedding 与 fused distributed CE `[业界通识]`。

**讲给客户的一句话**：词表切分省下的不是权重而是那 600 多 MiB 的 Logits 激活，方法是让 CE 只用分块归约的三个标量完成。

**深挖入口**：帖子未发，读仓库 README

### ch38 · Sequence Parallel（训练篇 23）

**解决什么问题**：TP 区域之间的激活仍各 Rank 重复保存：残差流与 LayerNorm 的激活是完整 `[batch, seq, hidden]`，与权重切不切无关。

**机制通俗版**：利用 RMSNorm 逐 Token 独立（均值只在 hidden 维）、残差逐元素的性质，把这些非 TP 区域的激活沿 Sequence 维切分：进入 TP 区域前 AllGather（g 算子）聚合出全序列，离开时 Reduce-Scatter（ḡ 算子）收回分片；Attention 是唯一序列耦合算子，仍在 TP 区域内跑全序列。代价是每层多一对 AllGather/Reduce-Scatter 边界通信；收益首先出现在长序列容量。连锁变化：LayerNorm 等复制参数的梯度从"天然一致"变成"部分和"，须 hook 同步。

**架构上的位置**：最低 2 卡，权重布局与 ch36 完全一致。边界：明确排除 Context Parallel/Ring Attention；Megatron 把 f/g 融合为一次 Reduce-Scatter 省一半反向字节，本章不做。

**业界对应**：Megatron-LM Sequence Parallel `[业界通识]`。

**讲给客户的一句话**：序列并行切的不是权重是激活——长序列下那才是显存大头，代价是每层一次往返的边界通信。

**深挖入口**：帖子未发，读仓库 README

### ch39 · GPipe Fill-Drain（训练篇 24）

**解决什么问题**：TP/SP 切的是"层内"；Pipeline Parallel 把 28 层按层跨卡切（PP=2 每 Stage 14 层）。但只切层不够——一个 micro-batch 从头走到尾时，任一时刻只有一张卡在算。

**机制通俗版**：GPipe 的 Fill-Drain 调度：先让所有 micro-batch（micro-batch 即流水线上流动的小批次）连续做完 Forward，再逆序做 Backward，流水填满后排空。Bubble（流水气泡，即 Stage 空转）上界为 `(S-1)/(M+S-1)`，micro 越多气泡越小；代价是每个 Stage 从 Fill 起要同时驻留 M 份激活图。Backward 靠 P2P 传"输出激活的输入梯度"驱动，与单卡语义一致。PP 特有的工程问题：tied Embedding/LM Head 分居首末 Stage，反向后要一对 P2P 交换两份部分梯度再更新。

**架构上的位置**：最低 2 卡，与 serial（逐个走完全流水）对照。边界：0.6B 太小、P2P 走 PCIe，端到端可能为负。

**业界对应**：GPipe 论文、Megatron-LM pipeline `[业界通识]`。

**讲给客户的一句话**：流水线用 micro-batch 排队填满各 Stage，气泡随批数下降，但显存要同时握住所有批的激活。

**深挖入口**：帖子未发，读仓库 README

### ch40 · 1F1B（训练篇 25）

**解决什么问题**：GPipe 的 Fill-Drain 要求每个 Stage 从头握到尾 M 份激活图，显存随 M 线性上涨。

**机制通俗版**：非交错 1F1B（一次 Forward 一次 Backward 交替）：Stage s 先做 `min(S−1−s, M)` 个 Warm-up Forward，进入 Steady State 后每轮先收后发地交替一次 Forward、一次 Backward，最后 Cool-down 排空 Backward。in-flight 激活（同时在手的激活图）从 M 份降到 `min(S−s, M)` 份——这是 1F1B 的主要价值，Bubble 上界仍服从同一填充-排空模型。细节：Steady 一轮必须"所有阻塞 recv 先于 send 挂出"，否则严格阻塞的传输（gloo）会互相等待成死锁；梯度按 micro 序累加，与 GPipe 逐位一致。

**架构上的位置**：最低 2 卡，参考 4 卡，与 ch39 共享 Stage 划分与 P2P 协议。边界：价值首先是峰值激活显存下降，不是吞吐大幅上升。

**业界对应**：PipeDream/Megatron-LM 的 1F1B schedule `[业界通识]`。

**讲给客户的一句话**：1F1B 不减少气泡，它把每个 Stage 手里攥的激活从 M 份压到与 Stage 位置相关的小数目——用调度换显存。

**深挖入口**：帖子未发，读仓库 README

### ch41 · Interleaved 1F1B（训练篇 26）

**解决什么问题**：1F1B 的流水级数 S 被物理卡数锁死，每卡只持一个连续 Stage，气泡压不下去。

**机制通俗版**：每张卡保存多个非连续的虚拟 Stage（模型块），total stage = P × vpp：虚拟 Stage s 放在物理 Rank `s % P`，28 层 / P=4 / vpp=2 时 8 个虚拟 Stage 依次 [4,4,4,4,3,3,3,3] 层，每卡合计仍是 7 层。调度上 micro 按 P 分组，组内按块扫 Forward、按块逆序扫 Backward；流水级数从 P 增到 2P，气泡缩短，但同一载荷要飞过 2P−1 条 Stage 边，P2P 次数约翻倍、Warm-up 数与 in-flight 显存上升。依赖在块边界闭合成环，阻塞 send 会死锁，必须异步 `isend` 投递 + 固定 FIFO 顺序。

**架构上的位置**：最低 2 卡，参考 4 卡；与每 Rank 一个连续 Stage 的 1F1B 对照，两臂每 Rank 层数相同。边界：micro 必须能被 P 整除；0.6B + PCIe 下端到端可能为负。

**业界对应**：Megatron-LM 的 interleaved schedule（virtual pipeline）`[业界通识]`。

**讲给客户的一句话**：虚拟 Stage 用"每卡多拿几段模型"把气泡压小，付的是通信次数翻倍和更长的等待链。

**深挖入口**：帖子未发，读仓库 README

### ch42 · 混合并行与双节点4+4（训练篇 27）

**解决什么问题**：DP、TP、PP 已各自验证，真实的训练集群需要三者同时启用——组合会不会破坏语义？跨节点该让哪个维度跨出去？

**机制通俗版**：不发明新原语，按 rank 分解公式建三维拓扑（TP 组节点内、PP 组节点内、DP 组跨节点），每节点一份完整 TP2×PP2 模型副本，训练主路径只有 DP 同步跨节点。数据只按 DP 切，TP 切权重不切数据，PP 按 micro 流过。三组单变量差分（只加 PP / 只加 DP）加一个综合配置保证受控性；任意两维布局之间的横向差异不写成单变量因果结论。易错点集中：`dist.new_group` 是全局集合操作必须全 Rank 同序创建；TP 组内再 AllReduce 复制参数梯度会放大 tp_size 倍；精确全局范数要防 (dp,pp) 单元被重复计数。

**架构上的位置**：最低 4 卡，完整参考为双节点 4+4。边界：跨节点通信口径以 NCCL Transport 记录解释，不做分段数字；边界激活在 TP 组内冗余传输 tp_size 倍是本布局的真实代价。

**业界对应**：Megatron-LM/DeepSpeed 的 3D 并行 `[业界通识]`。

**讲给客户的一句话**：三维并行的工程本质是一张 rank 分配表加三类组内通信——把 TP/PP 留在节点内、DP 跨节点，是对互联带宽的尊重。

**深挖入口**：帖子未发，读仓库 README

### ch43 · 分布式Checkpoint与恢复（训练篇 28）

**解决什么问题**：三维并行跑通了，但训练状态只存在于各 Rank 内存——崩溃即从头再来。checkpoint（检查点）要做的不只是存参数：进度、Sampler、调度器、每 Rank 的 RNG 都得进清单。

**机制通俗版**：唯一变量是保存方式：centralized（rank 0 聚合写单文件，其他 Rank 在 barrier 等待）对照 sharded（每 Rank 并行写自足分片 + rank 0 写 manifest 清单），`torch.distributed.checkpoint` 作官方对照。DDP 下各 Rank 状态逐位相同，所以"分片"是可互换的完整副本，4 卡保存 → 2 卡加载的重分片退化为"副本选择 + 校验"。恢复陷阱三连：LR Scheduler 的 state dict 不回写 optimizer 的 param_groups，必须用 `_last_lr` 手动回填；RNG 是每 Rank 状态，各取自己槽位；全部写盘走 `*.tmp` + `os.replace` 原子替换，被 Kill 的进程不会留下会被信任的半写文件。

**架构上的位置**：最低 2 卡，训练配置固定为 ch29 的 DDP 4 卡。边界：DDP 分片不省字节（sharded 体积约等于 centralized，收益只在并行 I/O 时间）；ZeRO/FSDP 的真字节级重分片属 ch34 形态。

**业界对应**：`torch.distributed.checkpoint`、各训练框架的 reshardable checkpoint `[业界通识]`。

**讲给客户的一句话**：checkpoint 的价值在"恢复后逐位接上"——学习率、随机数、进度任何一项漏存，续训就在静默漂移。

**深挖入口**：帖子未发，读仓库 README
