# 训练篇 B：从数据并行的通信优化到三维并行与容错（ch30–ch43）

### ch30 · DDP 梯度分桶与通信计算重叠

**先回答一个问题：为什么第 29 期的 DDP"慢"在哪个环节？**

DDP（Distributed Data Parallel，数据并行：每张卡持有完整模型、各吃一份数据，最后把大家的梯度加起来取平均）。它的梯度汇总靠 AllReduce（全体互相交换并求和，最后每张卡都拿到同一个总和的集合通信）。第 29 期的做法是：Backward 全部算完，再一次性对整份梯度做 AllReduce——通信排队在计算之后，整段通信时间都暴露在关键路径上（所谓关键路径，就是决定总时长的那条串行链条）。真实的 DDP 不这样工作：参数被分成若干桶（Bucket，把相邻参数打包成的通信单位），某个桶的梯度一齐就绪时就立刻发起异步 AllReduce，与后续 Backward 计算重叠。本期只改变一件事：AllReduce 的启动时机。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 梯度 | "往哪边调参数能让 loss 变小"的方向值，Backward 逐层算出 |
| Bucket | 把相邻参数的梯度打包成一个桶，攒齐一桶发一桶 |
| 异步 AllReduce | 发起通信后不干等，继续算 Backward，晚点再取结果 |
| 关键路径 | 决定整步耗时的串行链条；藏进链条里的时间等于白省 |
| 尾部等待 | 最后一个桶算完还得等通信收尾的那段时间 |

**为什么需要它**：想象你在食堂打饭——要么所有菜都点完再一起结账（第 29 期），要么点一个菜结一次、结账的同时继续点（本期）。当某层梯度算好时，它的通信根本不必等后面的层。

**怎么工作**：`build_buckets` 按参数逆序装桶（反向传播是从最后一层往回算的，逆序装桶让"先就绪的梯度"落在同一个桶里）；`register_post_accumulate_grad_hook` 钩子在梯度就绪时触发：拷入桶缓冲→计数→桶满即以 `async_op=True` 发 AllReduce；`wait_and_attach` 在尾部等所有通信完成、把结果以视图挂回参数。桶太小则固定延迟占比上升反而变慢；桶太大退化为第 29 期的全暴露形态。

**数字**（README ch30）：固定项为 2 卡、全局 batch 4、FP32、每步约 2.4 GB 梯度；benchmark 扫 1/16/128/1024 MiB 四档桶，3 次重复。假设前提：PCIe PHB、无 NVLink，通信与计算共享带宽，收益可能有限。

**边界与易错**：收益依赖通信带宽与计算能否真的重叠；桶大小是权衡不是越大越好。[易错] "分桶一定更快"——桶过小时每桶的启动延迟加起来比省下的暴露时间还多。

**产品对应**：PyTorch DDP 内置的 gradient bucket（默认 bucket 化）`[业界通识]`。

**方案视角**：现场先问"卡间走的是什么互联？"——NVLink 与 PCIe 的重叠收益差一个量级；再看单步梯度量，决定桶档位。

**讲给客户的一句话**：数据并行的通信不必干等——按桶流水发出去，让通信和计算同时进行。

**掌握目标**：能画出"Backward 中按桶异步 AllReduce"的时序图并与一次性 AllReduce 对比；能答出桶过大、过小分别退化成什么。

**深挖入口**：帖子未发，读仓库 chapter_30

<!--fig:ch30-->

### ch31 · 梯度累积期间延迟同步（no_sync）

**先回答一个问题：当一个 Optimizer Step 由 K 个 Micro-step 组成时，前 K−1 次同步是浪费吗？**

是。Micro-step（微步）指把一个大批次切成 K 份、逐份算梯度但不更新参数——这叫梯度累积，显存装不下大批次时的标准做法。每微步都做一次全量 AllReduce 是极大的浪费：前 K−1 次可以纯本地累积（no_sync 模式），最后一次统一同步，梯度数学不变、通信量降为 1/K。本期只改变一件事：微步间同步的时机（每微步同步 对照 仅最后一次同步）；Bucket 不用，AllReduce 仍是 Backward 结束后的平铺 SUM。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 梯度累积 | 切 K 份小批次逐个算梯度、攒够再更新一次参数 |
| no_sync | 累积期间不做跨卡同步，各卡只攒自己的梯度 |
| 独立累加器 | 存"每微步 AllReduce 结果之和"的缓冲区 |
| 监督 Token | 真正参与 loss 计算的 Token（不含 padding） |
| token_mean | loss 按全局监督 Token 数取平均的口径 |

**为什么需要它**：K=2 时每微步同步要做 2 次全量 AllReduce，而数学上只需要 1 次——一半的通信是白花的。

**怎么工作**：`every` 模式：每微步 backward→AllReduce SUM→写入独立累加器；`last` 模式：全部微步本地累积后一次 AllReduce。每微步的 loss 按该微步监督 Token 占比缩放后，两种模式都等于全局 token_mean 梯度。[易错] 不能把每微步的 AllReduce 结果直接加进"已被归约的累积缓冲"再反复 AllReduce——那会把先前微步的贡献放大 World Size 倍；必须用独立累加器。

**数字**（README ch31）：主对照 K=2（全局 batch 4），容量对照 K=4（全局 batch 8，每卡每微步 1 样本）；K=2 时 no_sync 省一次全量 AllReduce，即每步通信量 1/K。

**边界与易错**：省的是通信不是计算，backward 计算量不变；K 越大每微步 batch 越小、算力利用率越低。[易错] "延迟同步会改变梯度"——正确实现下两种模式梯度逐位（FP32 结合律容差内）一致。

**产品对应**：PyTorch DDP 的 `model.no_sync()` 上下文、HF Trainer 的 `gradient_accumulation_steps` `[业界通识]`。

**方案视角**：现场先问"显存允许的单步 batch 多大？"——不够就得累积，累积就该配 no_sync；顺带确认框架是否真的关掉了微步间同步。

**讲给客户的一句话**：小卡训大 batch，靠梯度累积；累积期间关掉跨卡同步，通信量直接除以累积份数。

**掌握目标**：能算出 K 个微步下 no_sync 的通信量节省；能答出"对已归约缓冲反复 AllReduce"错在哪里。

**深挖入口**：帖子未发，读仓库 chapter_31

<!--fig:ch31-->

### ch32 · ZeRO-1：切分优化器状态

**先回答一个问题：DDP 每张卡上，最占显存但"最不该人手一份"的是什么？**

优化器状态。AdamW 给每个参数存两个动量（一阶矩、二阶矩，可理解为"最近梯度的滑动平均"和"滑动平方平均"），FP32 下是参数字节数的 2 倍。DDP 每卡持有完整模型、完整梯度和完整优化器状态——0.6B 模型在 12GB 卡上已足以形成容量压力。本期只改变一件事：FP32 优化器状态是否在 Data Parallel Rank 间切分（ZeRO-1；Shard 指每卡只留 1/W 的分片）。参数和梯度仍每卡完整。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 优化器状态 | AdamW 每参数两份动量，更新参数用的"记忆" |
| ZeRO | 零冗余优化器：把训练状态切到各卡，消除人手一份的冗余 |
| Shard | 一份数据被切成 W 片，每卡只持一片 |
| AllGather | 各卡把各自分片拼回完整数据、人人拿全量的集合通信 |
| Rank | 分布式训练里每张卡的编号 |

**为什么需要它**：显存账（README 相对关系口径）：0.6B FP32 下参数 2.4GB、AdamW 状态 4.8GB（见 ch43 账本）——优化器状态是参数的 2 倍。复制式 DDP 里这 4.8GB 每卡一份、W 份全同；切分后每卡只留 1/W。0.6B 尚且如此，几十 B 的模型下这笔账就是"训得动 vs 训不动"。

**怎么工作**：AdamW 是逐元素更新，"本 Rank 的状态分片 × 参数分片 × 梯度分片"与全量 AdamW 逐位一致。代价：每步更新完成后用分块 AllGather（块缓冲 128MB）把新参数同步给其他 Rank——这是 DDP 原本没有的约 1×参数字节的额外通信；收益是状态显存降为 1/W。

**算一算**：W=2 时，0.6B 的 AdamW 状态从每卡 4.8GB 降到 2.4GB，省 2.4GB；付出的通信是每步约 2.4GB（参数字节）的 AllGather——用便宜的通信换昂贵的显存。

**边界与易错**：多付通信，小模型上可能不划算；参数、梯度仍全量，更大的模型还得继续切（ch33/34）。[易错] "ZeRO-1 改变了数学"——分片更新与全量 AdamW 逐位一致，语义不变。

**产品对应**：DeepSpeed ZeRO Stage 1、FSDP 的 optimizer state sharding `[业界通识]`。

**方案视角**：现场先问三个数：卡数、单卡显存、卡间带宽。显存是第一瓶颈且带宽尚可时先上 ZeRO-1；用 capacity_probe 式探针找到 Batch/Length 准入边界再定服务形态。

**讲给客户的一句话**：优化器状态是训练显存的大头，把它按卡数切分，同样的卡能装下更大的 batch 或更长的序列。

**掌握目标**：能算出 W 张卡下 ZeRO-1 的状态显存与新增 AllGather 字节；能判断给定显存下该不该上 ZeRO-1。

**深挖入口**：帖子未发，读仓库 chapter_32

<!--fig:ch32-->

### ch33 · ZeRO-2：再切分梯度

**先回答一个问题：backward 结束那一刻必须存在的完整梯度，之后都用得上吗？**

用不上。ZeRO-1 之后每卡仍保留完整梯度，但本卡的优化器只会用到"属于本卡参数分片"的那一份。本期只改变一件事：梯度是否切分——完整 AllReduce 对照 Reduce-Scatter（归约求和的同时顺手分发：每卡只拿到总和里属于自己的那一片）。优化器状态沿用 ZeRO-1 切分，参数仍每卡完整——所以是 ZeRO-2 而不是 ZeRO-3。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Reduce-Scatter | 各卡数据求和后按编号分发：人人拿到总和的 1/W |
| 梯度分片 | 本卡只保留"本卡参数分片"对应的梯度片 |
| 全局范数 | 所有参数梯度平方和开根号，梯度裁剪用 |
| 裁剪 | 梯度范数超阈值就按比例缩小，防训练爆炸 |
| 容量准入边界 | 给定显存下能跑的 Batch/长度上限 |

**为什么需要它**：显存账：完整梯度 FP32 下 0.6B 是 2.4GB；切分后每卡 2.4/W GB。连同 ch32 的状态分片，每卡省下的显存直接换成更大的 batch——准入边界外移。

**怎么工作**：backward 逐参数算出梯度后立刻 Reduce-Scatter（整除时直接以梯度视图为输入，完成后释放完整梯度；非整除按零填充，填充位求和仍为零）。Reduce-Scatter 与 AllReduce 的每链接数据量同为 2(W−1)/W·D，W=2 时通信时间接近——同样的通信字节数，换来的是"只留分片"。[易错] 梯度切分后 `clip_grad_norm_` 直接失效：各卡只有局部范数，必须"各卡局部平方和 → 一次标量 AllReduce → 合成全局范数 → 缩放本地分片"。

**数字**（README ch33）：固定项沿用 2 卡配置；capacity_probe 扫 ddp/zero1/zero2 × Micro-batch {1,2,4} × Length {128,256,512} 的准入网格；假设 ZeRO-2 应比 ZeRO-1 更晚 OOM。

**边界与易错**：真实 ZeRO 的扁平桶化 Reduce-Scatter 本期不做，逐参数发起的通信代价如实测量；参数仍完整常驻。[易错] "Reduce-Scatter 是新的通信开销"——W=2 时它与 AllReduce 数据量相同，不是额外开销而是等价替换。

**产品对应**：DeepSpeed ZeRO Stage 2、FSDP 的 grad sharding `[业界通识]`。

**方案视角**：现场先问"卡数/显存/网络"三件套；ZeRO 级别每升一级，显存多省一截、通信结构多一点复杂——用准入边界实测值而不是参数量拍脑袋选 Stage。

**讲给客户的一句话**：梯度算完只有 1/W 有用——Reduce-Scatter 让每卡只留自己那份，同样的卡再挤出一份显存。

**掌握目标**：能算出 W=2/4 时每卡的梯度显存节省；能答出分片后全局范数怎么合成。

**深挖入口**：帖子未发，读仓库 chapter_33

<!--fig:ch33-->

### ch34 · ZeRO-3 / FSDP：再切分参数

**先回答一个问题：参数只在 forward/backward 里被读，凭什么在两步之间整份占着显存？**

不该占。ZeRO-3/FSDP（Fully Sharded Data Parallel，全分片数据并行）让训练参数也只以 Shard 形式常驻每个 Rank，用的时候再临时拼全量。至此每卡常驻模型状态从 ZeRO-2 的 `2N+14N/W` 降到 `16N/W` 字节（README 的 FP32 账本口径，N 为参数量、W 为卡数）。本期只改变一件事：参数是否分片，对照 ZeRO-2。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| AllGather 参数 | forward 前把各卡参数分片临时拼成全量 |
| Reshard | 用完全量立刻释放全量存储，退回分片状态 |
| 热路径 | forward/backward 主计算流程，通信进去就要排队 |
| no-grad 区 | autograd 不记录的代码段，纯搬运用 |
| tied 权重 | Embedding 与 LM Head 共用同一个参数 |

**为什么需要它**：显存账（README 口径）：ZeRO-2 每卡 `2N+14N/W`，ZeRO-3 `16N/W`——W 越大差距越大。0.6B 小模型上差异不明显；参数量上去后这是"单机装不下 → 必须多机"还是"单机多卡即可"的分界。

**怎么工作**：参数从此有了完整生命周期：分片常驻 → Forward 前 AllGather → 计算 → Reshard → Backward 前按需 AllGather → 梯度 Reduce-Scatter → 分片 Optimizer Step。关键变化是 AllGather 从"每步一次"搬进了热路径；下一个 forward 的 AllGather 本身就是参数刷新，ZeRO-1/2 那次 post-step AllGather 被取消。代价：通信从每步约一次参数量级变成 forward/backward 各一次——0.6B 上很可能 Step Time 变长，README 明确保留这个负优化结果。

**数字**（README ch34）：参考配置 4 卡、覆盖双节点 4+4（W=8）；benchmark 扫 W ∈ {2,4,8}；显存报告与理论账本（`2N+14N/W` vs `16N/W`）同列；FSDP 的 Reduce-Scatter 取平均，本地 loss 乘 W 校正后与全局 token_mean 一致。

**边界与易错**：通信进热路径，带宽差时纯负优化；激活重算与 CPU Offload 本期排除。[易错] "ZeRO-3 一定比 ZeRO-2 快"——它是容量方案不是速度方案，小模型上大概率更慢。

**产品对应**：PyTorch FSDP（`FULL_SHARD`、`use_orig_params`、`BackwardPrefetch`）、DeepSpeed ZeRO Stage 3、MiCS/HF Accelerate 的分片封装 `[业界通识]`。

**方案视角**：现场先问"卡数/显存/网络"：模型单卡放不下才谈 ZeRO-3；卡间带宽弱时先问"能不能用 ZeRO-2 + 更小 batch 顶住"。

**讲给客户的一句话**：ZeRO-3 是"装得下"的方案——参数也按卡切片，用的时候再临时拼，代价是热路径里多两次通信。

**掌握目标**：能写出 ZeRO-2 与 ZeRO-3 的每卡常驻字节公式并算出 W=4 的差值；能画出参数"分片→AllGather→Reshard"生命周期图。

**深挖入口**：帖子未发，读仓库 chapter_34

<!--fig:ch34-->

### ch35 · MLP Tensor Parallel

**先回答一个问题：ZeRO 切的是"训练状态"，单层内部的矩阵运算还能切吗？**

能。Tensor Parallel（TP，张量并行：把一层内部的权重矩阵切开，各卡算自己那块，再拼起来）。本期只切 SwiGLU MLP：Gate/Up 权重沿输出维切（Column Parallel，输入复制、输出分片），Down 权重沿输入维切（Row Parallel，输入分片、输出 AllReduce 求和）。Attention、Embedding、LayerNorm 保持每卡完整，单独测清 MLP TP 这一个机制。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Column Parallel | 权重按列切：各卡算不同的输出通道 |
| Row Parallel | 权重按行切：各卡吃不同的输入切片，输出求和 |
| f 算子 | forward 恒等、backward 做 AllReduce 的通信占位符 |
| g 算子 | forward 做 AllReduce、backward 恒等的通信占位符 |
| SwiGLU | Qwen3 的 MLP 结构：silu(Gate)×Up 再过 Down |

**为什么需要它**：单卡显存装不下"单层的权重+激活"时，就得把层切开。0.6B 单卡本可完整常驻，所以 TP 在本实验里更可能是负优化——这是课程预期内的合法结果，负优化如实保留。

**怎么工作**：每个 Rank 拿相同输入 x，算自己分片的 `G_r=x@W_gate_r^T` 等局部结果，forward 末尾一次 AllReduce SUM 拼出完整输出 y。梯度分三类（本期最重要的一课）：① MLP 分片权重的梯度只在本地，不需要也不应该做 AllReduce——各卡分片梯度拼起来就是完整梯度；② MLP 输入激活的残差流梯度是各卡部分和，必须做一次 backward AllReduce（f 算子）——漏掉它，"部分梯度"会沿残差流悄悄流进 Attention 路径，训练语义被静默改坏；③ 复制参数的梯度各卡天然相同，再做 AllReduce SUM 会恰好放大 tp_size 倍——是错误，不是浪费。

**数字**（README ch35）：每层新增 2 次 TP AllReduce（forward down_proj 输出、backward MLP 输入梯度），全部暴露在关键路径上；对照臂 tp1/tp2/tp4，正确性红线 loss < 1e-5。

**边界与易错**：TP 通信暴露不重叠，收益依赖高速互联；每层两次 AllReduce 随层数线性累积。[易错] "所有梯度都要 AllReduce"——分片权重不需要，复制参数更不能做，三类梯度三种处理。

**产品对应**：Megatron-LM 的 MLP 并行、vLLM/SGLang 的 TP 推理同源机制 `[业界通识]`。

**方案视角**：现场先问"卡间互联带宽够不够 TP"——TP 的 AllReduce 每层两次且不能重叠，NVLink 机器和 PCIe 机器结论完全不同。

**讲给客户的一句话**：TP 把一层内部的计算切开分给多卡，但每层要付两次"全员对账"的通信——只有高速互联才付得起。

**掌握目标**：能画出 MLP TP 的 f/g 算子位置；能区分三类梯度的通信要求并答出"复制参数再 AllReduce 会放大几倍"。

**深挖入口**：帖子未发，读仓库 chapter_35

<!--fig:ch35-->

### ch36 · Attention Tensor Parallel

**先回答一个问题：MLP 切完了，Attention 的"头"怎么切？**

按 Head 切。Qwen3-0.6B 用 GQA（Grouped-Query Attention：多个 Q 头共享一组 KV 头，省 KV 显存的注意力结构）：32 个 Q 头、8 个 KV 头（4:1）、head_dim 128。TP=2 时每 Rank 拿 16 个 Q 头 + 4 个 KV 头，TP=4 时 8+2——切分必须满足整除约束，每 Rank 拿整数个完整 Head；head_dim 永不切。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Q/KV Head | 注意力的"视角"：Q 头提问，KV 头提供检索内容 |
| GQA | 多个 Q 头共享一组 KV 头，Q:KV = 4:1 |
| RoPE | 给 Q/K 加位置信息的固定旋转，Head 内操作 |
| q/k norm | Qwen3 对每个 Head 内部做 RMSNorm |
| repeat_kv | 把 KV 头复制 4 份对齐 Q 头的操作 |

**为什么需要它**：0.6B 的 Attention 参数占比小，预期收益有限甚至为负——但机制必须掌握：这是大模型 TP 的标准组成。

**怎么工作**：在 MLP TP 基础上，q/k/v 投影按 Head 行切、o_proj 按列切。RoPE 与 `repeat_kv` 都在局部 Head 集合内重放，与全局逐位等价。每层通信从 mlp_tp 的 1 次前向 + 1 次反向变成 2 次前向（o_proj 输出、down_proj 输出）+ 4 次反向（两个块输入梯度 + q/k norm 同步）。本期特有的陷阱：q_norm/k_norm 权重每卡复制，但各卡只累积自己 Head 的梯度项——原始梯度是部分和，不处理后各卡对同一参数施加不同更新，副本从第一步起静默分叉。实现用 post-accumulate-grad 钩子对每个 norm 权重做一次 backward AllReduce SUM。

**数字**（README ch36）：32 Q/8 KV Head，TP=2 → 16/4，TP=4 → 8/2，局部 GQA 组数恒为 4；attn_tp 臂每层 6 次 AllReduce 对 mlp_tp 的 2 次。

**边界与易错**：Head 数与 tp_size 的整除约束（32/8 都能被 2、4 整除）；0.6B 上收益小。[易错] "RoPE 要在全局上做"——它是 Head 内固定旋转，局部重放逐位等价。

**产品对应**：Megatron-LM 与 vLLM/SGLang 的 Attention TP、GQA 布局 `[业界通识]`。

**方案视角**：现场先问"卡间互联带宽够不够 TP"，再看模型 Head 数能否被 TP 度整除——不整除的模型配不出来的 TP 度直接排除。

**讲给客户的一句话**：Attention 按"头"切是多卡并行的天然切法，但每层通信次数翻倍，对互联带宽的要求比只切 MLP 更高。

**掌握目标**：能算出 TP=2/4 下每 Rank 的 Q/KV 头数；能答出 q/k norm 的梯度为什么是部分和、怎么修。

**深挖入口**：帖子未发，读仓库 chapter_36

<!--fig:ch36-->

### ch37 · 词表并行 Embedding 与分布式 Cross Entropy

**先回答一个问题：TP 模型里最大的单项激活是什么？**

不是注意力，是 Logits。Qwen3-0.6B 词表 151936；batch 2 × seq 512 的完整 Logits 是 151936 × 1024 × 4B ≈ 622 MiB——TP 模型里最大的单项激活。本期把 Embedding/LM Head 沿词表维切分（TP=2 每 Rank 75968 行），并用不物化完整 Logits 的分布式 Cross Entropy 闭合训练。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Logits | 对全词表每个 Token 打的分数，CE 的输入 |
| Cross Entropy（CE） | 把分数变成"下一个词"的 loss 的标准算法 |
| log-sum-exp | CE 里的归一化分母，可分块计算 |
| Vocab Parallel | 词表按行切开，每 Rank 持有一段 |
| tie | Embedding 与 LM Head 共享同一个权重矩阵 |

**为什么需要它**：622MiB 的完整 Logits 在长序列/大 batch 下线性膨胀，常常是 OOM 的元凶；分片后每 Rank 只物化自己的 1/tp_size 段。

**怎么工作**：CE 的 log-sum-exp 可按词表任意分块归约——全局 max 用 AllReduce MAX、全局 exp 和用 AllReduce SUM、目标 token 的 logit 按所属 Rank 用一次 AllReduce SUM 取回；这三个 Collective 的尺寸是 token 数而不是词表数。CE 的 backward（softmax − onehot）只需要本 Rank 分片和三个全局标量——**backward 0 通信**。隐藏状态 h 是复制的，LM Head 分片的输入梯度 `dlogits_r @ W_r` 是部分和，需要一次 backward AllReduce（f 算子）——漏掉它，LayerNorm 梯度跨 Rank 不一致、tied Embedding 梯度对照失败。tie 的两条路径（lookup + logits）梯度自动累进同一个分片参数。

**算一算**：完整 Logits ≈ 622 MiB；TP=2 下 vocab_tp 臂物化 ≈ 311 MiB，`local_logits_bytes/full_logits_bytes = 1/tp_size`。

**数字**（README ch37）：词表 151936、TP=2 每 Rank 75968 行、batch 2×seq 512 完整 Logits ≈ 622 MiB；vocab_tp 臂 forward 每层 2 次块输出 + embedding 1 次 + CE 3 次。

**边界与易错**：0.6B、tp=2 下收益可能仍为负——负优化如实保留；词表必须被 tp_size 整除。[易错] "分布式 CE 的 backward 很贵"——恰相反，它 backward 0 通信；固定代价是 LM Head 输入梯度那一次 AllReduce。

**产品对应**：Megatron-LM 的 VocabParallelEmbedding 与分布式 CE、vLLM 的并行词表 `[业界通识]`。

**方案视角**：现场先问"卡间互联带宽够不够 TP"，再看序列长度 × batch × 词表的 Logits 账——长序列场景它是第一显存大项。

**讲给客户的一句话**：词表 15 万行的 Logits 是显存杀手——把它按卡切开、loss 各算各的再合，622MiB 直接变 311MiB。

**掌握目标**：能算出给定 batch/seq/词表下的完整 Logits 字节；能答出分布式 CE 的 3 次 forward Collective 和 backward 为何 0 通信。

**深挖入口**：帖子未发，读仓库 chapter_37

<!--fig:ch37-->

### ch38 · Sequence Parallel

**先回答一个问题：权重都切完了，还有什么在每卡重复？**

残差流和 LayerNorm 的激活。它们是完整的 `[batch, seq, hidden]`，与权重是否分片无关。本期在 Attention+MLP 权重分片不变的前提下，把这些非 TP 区域的激活沿 Sequence 维切分（Sequence Parallel，SP：激活按序列切段、每卡只存一段）。Attention 内部仍跑全序列；Context Parallel / Ring Attention 不在本期。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 激活 | forward 中间结果，backward 要用所以驻留显存 |
| SP 区域 | TP 块之间做 LayerNorm/残差的部分 |
| g（SP 版） | 进 TP 块前 AllGather 拼全序列 |
| ḡ | 出 TP 块后 Reduce-Scatter 切回序列段 |
| saved-for-backward | 为反向保存的激活字节数 |

**为什么需要它**：长序列训练时激活是显存第一大头；权重切得再细，激活不切照样 OOM。

**怎么工作**：RMSNorm 逐 Token 独立（均值只在 hidden 维）、残差逐元素，所以在 `[b, s/N, h]` 上算等于全序列算完再切片。流程：序列分片 → RMSNorm → AllGather 进 TP 块 → 注意力部分和（不归约）→ Reduce-Scatter 出块 → 残差仍在分片上。本期最深的易错点：非 TP 权重（LayerNorm、最终 norm、Embedding/LM Head）的梯度从"天然一致"变成"部分和"——每个 Rank 只看到自己的 Token，必须用钩子逐权重做 backward AllReduce SUM；漏掉任何一个，副本从第一次 optimizer step 起静默分叉。另外 g 的 backward 是本地切片不是 Reduce-Scatter（f 已求和出一致完整梯度，切片即可；对一致副本再做 Reduce-Scatter 会放大 tp_size 倍）。

**数字**（README ch38）：主实验 W=2、L=512；代价是每并行块多一次 AllGather + 一次 Reduce-Scatter；0.6B 激活占比小，端到端吞吐可能为负，收益预期首先出现在长序列容量（capacity_probe 扫 L 到 2048）。

**边界与易错**：短序列下边界通信吃掉收益；Megatron 把 f 与 g 融合成一次 Reduce-Scatter 省一半反向字节，本期不做。[易错] "SP 是新并行维度"——它只切 TP 区域之间的激活，序列维切分发生在 SP 区域。

**产品对应**：Megatron-LM Sequence Parallel、Megatron Context Parallel（扩展）`[业界通识]`。

**方案视角**：现场先问"卡间互联带宽够不够 TP"，再看训练序列长度——L 上千时激活账主导，SP 才值得付边界通信。

**讲给客户的一句话**：长序列训练的显存瓶颈在激活——把 LayerNorm 之间的激活按序列切开，权重布局不变。

**掌握目标**：能画出每层 SP 区域的 g/ḡ 位置；能答出"非 TP 权重梯度为什么从天然一致变成部分和"。

**深挖入口**：帖子未发，读仓库 chapter_38

<!--fig:ch38-->

### ch39 · GPipe Fill-Drain 流水

**先回答一个问题：模型按层切成两半放两卡，为什么反而更慢？**

因为一个 micro-batch 走流水时，任一时刻只有一张卡在算——另一张在等。Pipeline Parallel（PP，流水线并行：模型按层跨卡切成 Stage，数据像流水线一样流过）。本期把 28 层连续均分：PP=2 每 Stage 14 层、PP=4 每 Stage 7 层（0.6B 恰好整除）；唯一变量是多 micro-batch 的调度——逐个走完全流水（serial）对照先连续 Forward 再连续 Backward 的 GPipe Fill-Drain。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Stage | 流水线上的一站：连续若干层+边界件 |
| Micro-batch | 把 batch 切小后逐个喂进流水线的单位 |
| Fill/Drain | 流水的填充与排空阶段，必然有空转 |
| Bubble | 流水空转的时间占比 |
| P2P | 相邻卡点对点直传激活与梯度 |

**为什么需要它**：PP 把模型放下了，但不调度就白切——serial 模式下 S 张卡的利用率只有 1/S。

**怎么工作**：GPipe 调度是"先 F…F 再 B…B"：所有 micro 依次 Forward 到最后 Stage，再依次 Backward 流回来。填充-排空模型给出 Bubble 上界 (S−1)/(M+S−1)（S 为 Stage 数、M 为 micro 数）——M 越大 Bubble 越小。代价：峰值激活显存从 serial 的每 Stage 1 份激活图涨到 M 份。Backward 靠 P2P 梯度种子驱动：末 Stage 从 loss 反传，其余 Stage 收到输出激活的输入梯度后 `autograd.backward(out, g)` 启动本段反传。tied Embedding/LM Head 被 PP 拆在首末两个 Stage，各自累积不同部分梯度——`exchange_tied_gradients` 在反向结束后做一对 P2P 交换（每步 2 × vocab×hidden 字节），两侧都拿 `g_embed + g_lm_head` 更新。

**算一算**：PP=2、M=8 时 Bubble 上界 = (2−1)/(8+2−1) ≈ 9%；M=2 时 = 1/3——micro 数直接决定流水效率。

**数字**（README ch39）：28 层 PP=2 → [14,14]、PP=4 → [7,7,7,7]；micro 扫 {2,4,8}；serial 与 gpipe 峰值显存差（1 份 vs M 份激活图）是机制性结论。

**边界与易错**：0.6B 太小、P2P 走 PCIe，端到端加速可能被通信吃掉甚至为负。[易错] "PP=2 就快一倍"——没有流水调度是 1/S 的利用率，有调度还有 Bubble 与激活显存代价。

**产品对应**：Megatron-LM 的 GPipe schedule、torchgpipe `[业界通识]`。

**方案视角**：现场先问"模型层数和卡数怎么配 Stage"——层数要能被 Stage 数基本整除；再问 micro 数能不能开大，micro 太少流水白搭。

**讲给客户的一句话**：把模型按层切给多卡接力，再用多个微批把流水填满——卡越多切得越细，但"填不满"的空转要用微批数压下去。

**掌握目标**：能算出给定 S、M 的 Bubble 上界；能画出 Fill-Drain 时序并标出空转段。

**深挖入口**：帖子未发，读仓库 chapter_39

<!--fig:ch39-->

### ch40 · 1F1B 流水调度

**先回答一个问题：GPipe 的代价是"所有 micro 的激活图都攒在手里"，能不能边走边放？**

能。1F1B（one-forward-one-backward：稳态时每做一次 Forward 就做一次 Backward）把激活生命周期从"整个 Step"缩短到"一个 micro"。Fill-Drain 下每 Stage 从 Fill 开始驻留 M 份激活图；1F1B 下每 Stage 的 in-flight 激活峰值降到 **min(S−s, M)** 份（s 为本 Stage 编号、S 为 Stage 总数、M 为 micro 数）。本期唯一变量：Fill-Drain 对照非交错 1F1B，Stage 划分、micro 序列、P2P 协议完全相同。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Warm-up | 开局连做若干次 Forward，凑出稳态需要的在飞深度 |
| Steady State | 一次 Forward + 一次 Backward 交替的主循环 |
| Cool-down | 结尾排空剩余 Backward |
| in-flight 激活 | 已 Forward 还没 Backward、必须驻留的激活图 |
| 1F1B | 稳态 F/B 交替的流水调度 |

**为什么需要它**：M 大时 GPipe 的 M 份激活图是显存灾难；1F1B 用"更早开始 Backward"换显存，Bubble 上界不变，仍服从 (S−1)/(M+S−1)。

**怎么工作**：Stage s 的 Warm-up 数 = min(S−1−s, M)——越靠前的 Stage 第一个 Backward 到达越晚，能提前做的 Forward 越多；M 不够时全部算 Warm-up，退化成 Fill-Drain 形状。稳态每轮"收激活→Forward→收梯度→Backward→发梯度→发激活"：所有阻塞 recv 先于本轮 send 挂出，打断 gloo 严格阻塞传输下的死锁环（NCCL 异步入队两种顺序都成立，所以 CPU 与 GPU 同一条代码路径）。梯度语义不变：各参数梯度按 micro 序累加，与 gpipe 逐位一致。

**算一算**：S=4、末 Stage（s=3）的 Warm-up = min(0, M) = 0——最后 Stage 不需要 Warm-up，直接开始 1F1B；首 Stage（s=0）= min(3, M)，M=4 时 3 份在飞。

**数字**（README ch40）：in-flight 峰值 min(S−s, M) 对照 gpipe 的 M；benchmark 扫 micro {2,4,8}，W=2 主实验、W=4 对照；主要价值假设是峰值激活显存下降而非吞吐大增。

**边界与易错**：Bubble 不比 GPipe 小，收益全在显存；更多次收发等待在 PCIe 小模型上可能吃掉收益。[易错] "1F1B 更快"——它先解决"放得下"，其次才是速度。

**产品对应**：Megatron-LM 的 non-interleaved 1F1B schedule、DeepSpeed Pipeline Engine `[业界通识]`。

**方案视角**：现场先问"模型层数和卡数怎么配 Stage"，再看显存瓶颈在参数还是激活——激活主导时 1F1B 优先于加大 micro。

**讲给客户的一句话**：1F1B 让每张卡"算一个就还一个"激活，流水照样满，但显存占用从微批总数降到流水深度量级。

**掌握目标**：能算出给定 S、s、M 的 Warm-up 数与 in-flight 峰值；能画出 1F1B 时序并对照 Fill-Drain 标出显存差。

**深挖入口**：帖子未发，读仓库 chapter_40

<!--fig:ch40-->

### ch41 · Interleaved 1F1B（虚拟 Stage）

**先回答一个问题：流水级数 S 被物理卡数锁死，还能再"切细"吗？**

能——一个物理卡放多个非连续的虚拟 Stage（模型块）。虚拟 Stage s 位于物理 Rank `s % P`，total = P × vpp（P 为物理卡数、vpp 为每卡块数）。W=4、vpp=2 时 28 层切成 8 个虚拟 Stage [4,4,4,4,3,3,3,3]，每 Rank 合计仍 7 层、权重切分互补不重叠。流水级数从 P 增到 2P，Bubble 更短；代价是同一载荷要飞过 2P−1 条 Stage 边而非 P−1 条，P2P 次数约翻倍。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 虚拟 Stage | 一张卡上的一个模型块，占流水线中的一级 |
| vpp | 每物理卡放的虚拟 Stage 数 |
| Placement 表 | 虚拟 Stage → 物理 Rank 的映射，唯一真相来源 |
| wrap 边 | 块边界从 P−1 绕回 Rank 0 的跨卡传递 |
| isend | 异步发送：投递后不等对方，先干别的 |

**为什么需要它**：Bubble 上界 (S−1)/(M+S−1) 里 S 越大 Bubble 越小；物理卡不够时，虚拟 Stage 是唯一能"加级数"的办法。

**怎么工作**：micro 按 P 分组，组内按块扫 Forward、按块逆序扫 Backward；`num_micro % P ≠ 0` 直接报错。块内边在相邻 Rank，块边界是 wrap 对（不再是相邻 Rank）；所有收发查 `stage_owner` 表。死锁与正确性的双重前提是异步发送：Interleaved 的依赖在块边界闭合成环（Rank A 下一块的 Forward 依赖邻居 B 当前块的 Backward，而 B 的这次 Backward 又依赖 A 上一块的产物——开发期 CPU smoke 上阻塞式 send 实测死锁），实现统一用 `dist.isend` 立即投递、步末统一 wait：所有阻塞等待都是 recv，链条必终于无 recv 的端点，可证明无环。

**算一算**：P=4、M=8 时 P2P 激活消息数：interleaved (2×4−1)×8=60 条 vs 非交错 (4−1)×8=24 条——消息更小更密，次数与等待时间要分开读。

**数字**（README ch41）：total = P × vpp；W=4/vpp=2 下 8 块 [4,4,4,4,3,3,3,3]、每 Rank 7 层；micro 扫 {4,8}；warm-up 公式 min(2(P−1−r) + (vpp−1)P, vpp·M)。

**边界与易错**：0.6B + PCIe 上端到端可能为负；vpp>2 与自动负载均衡排除。[易错] "虚拟 Stage 免费"——它用约翻倍的 P2P 次数和更大的 in-flight 显存换 Bubble。

**产品对应**：Megatron-LM 的 interleaved 1F1B schedule（`--num-layers-per-virtual-pipeline-stage`）`[业界通识]`。

**方案视角**：现场先问"模型层数和卡数怎么配 Stage"——层数 ÷ (卡数 × vpp) 要基本整除；Bubble 仍是痛点时再评估 vpp=2 的通信增量。

**讲给客户的一句话**：给每张卡发两段不相邻的模型块，流水级数翻倍、空转更短——代价是卡间点对点传输次数约翻倍。

**掌握目标**：能算出给定 P、vpp、层数的块分布与 total Stage 数；能答出 Interleaved 为什么必须用异步发送。

**深挖入口**：帖子未发，读仓库 chapter_41

<!--fig:ch41-->

### ch42 · 混合并行与双节点 4+4

**先回答一个问题：DP、TP、PP 各自学完了，生产环境怎么组合？**

三维并行（TP×PP×DP 同时启用）。本期不发明新原语，把三者拼起来并在双节点 4+4 上验证。拓扑由唯一的 rank 分解公式给出：`rank = dp_rank × (pp_size × tp_size) + pp_rank × tp_size + tp_rank`。TP=2、PP=2、DP=2（8 rank）时：TP 组 {0,1}{2,3}{4,5}{6,7}、PP 组 {0,2}{1,3}{4,6}{5,7} 都在节点内，DP 组 {0,4}{1,5}{2,6}{3,7} 跨节点——每个节点一份完整 TP2×PP2 模型副本，训练主路径只有 DP 的梯度 AllReduce 跨节点。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| DP/TP/PP | 数据/张量/流水三种并行的组合维度 |
| Process Group | 一次集合通信的"参与人名单" |
| 逻辑 Rank | 一张卡在 (dp,pp,tp) 三维坐标系里的坐标 |
| Strong Scaling | 同一全局工作量下比较不同并行布局 |
| LOCAL_RANK | 一张卡在本节点内的插槽编号 |

**为什么需要它**：单维并行各有硬边界：TP 受限于节点内带宽、PP 受限于层数与 Bubble、DP 受限于显存；生产级训练（如 Llama 系列公开配置 `[业界通识]`）都是三维组合。

**怎么工作**：三族组表由 rank 公式一次性导出并断言"恰好划分 world"；`dist.new_group` 是全局集合操作，每个 rank 必须按相同顺序创建所有组——只建"自己的组"会让其他 rank 永久等待。数据只按 DP 切；梯度所有权三类不变（TP shard 梯度本地、残差流 f 算子已归约、复制参数 TP 组内一致），DP 维度所有参数只需一次梯度 AllReduce。精确全局范数要防重复计数：先 TP 组归约 shard 部分，再全 world 归约后除以 tp_size。PP=1 时 tied 权重就地绑定单参数（首末两端同 Stage 的特例）。

**数字**（README ch42）：实验 D 在双节点 4+4 上跑 TP2/PP2/DP2；Stage 连续均分 28 层 / PP=2 → [14,14]；每条 Stage 边界消息按 tp_rank 配对飞 tp_size 次（字节冗余 tp_size 倍，计入 p2p_bytes 不隐藏）。

**边界与易错**：本期的调度是第 39 期 GPipe，无 1F1B/Interleaved；0.6B 上组合可能是负优化。[易错] "任意两维布局的性能差能归因到某一维"——README 明确：同时改多个维度的横向差异不写成单变量因果结论，只有单变量差分（A1→A2、B1→B2、C1→C2）才有因果意义。

**产品对应**：Megatron-LM / DeepSpeed 的三维并行配置、NCCL 进程组管理 `[业界通识]`。

**方案视角**：现场先问"卡间互联带宽够不够 TP"并把 TP/PP 锁在节点内、DP 跨节点——这是机柜内 NVLink、机柜间以太网时代的标准布局；用 rank 公式核对客户集群的拓扑映射。

**讲给客户的一句话**：单维并行各有天花板——TP 藏进节点内、PP 接力层、DP 只在节点间同步梯度，三维各司其职才能同时吃满显存、算力和网络。

**掌握目标**：能由 rank 公式推出给定 (dp,pp,tp) 的组表；能判断"TP/PP 留节点内、DP 跨节点"布局下跨节点通信的口径。

**深挖入口**：帖子未发，读仓库 chapter_42

<!--fig:ch42-->

### ch43 · 分布式 Checkpoint 与恢复

**先回答一个问题：训练跑到第 N 步断电，前面烧的卡时怎么办？**

靠 Checkpoint（训练状态快照：参数、优化器状态、进度、随机数种子等一起存盘，崩溃后从快照续跑）。第 42 期之后训练状态只在各 Rank 内存里——崩溃或停机即从头再来。本期唯一变量：完整状态集中保存为单文件（rank 0 聚合写出，其他 Rank 在 barrier 等待），还是各 Rank 并行保存 Shard（分片 + manifest 清单）；`torch.distributed.checkpoint`（DCP）作为官方对照。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Checkpoint | 可恢复续训的完整训练状态快照 |
| manifest | 分片清单：文件名、Rank、校验和、字节 |
| 校验和 | 内容指纹，用来核对两边数据逐位一致 |
| 重分片 | W1 张卡存的快照被 W2 张卡加载 |
| 原子替换 | `*.tmp` 写完再 `os.replace`，绝不留半写文件 |
| RTO | 灾难恢复时间目标：断多久能接着训 |

**为什么需要它**：快照不是只存参数——0.6B FP32 每副本约 7.2GB（参数 2.4GB + AdamW 状态 4.8GB），还要 LR 调度器、全局 Step、Sampler 游标、每 Rank 一份的 RNG 状态、拓扑与软件版本。漏任何一样，续训就"接不上"。

**怎么工作**：DDP 各 Rank 在 AllReduce 后持有逐位相同的副本，所以 DDP 下的重分片退化为"副本选择 + 校验"：新 Rank r 读 `shard-{r % 4}`，rank 0 先核对各分片校验和一致。两个著名陷阱：① LR 调度器 `load_state_dict` 不回写 optimizer 的 param_groups——新建调度器把 LR 重置为 Warm-up 初值，恢复后必须用 `_last_lr` 回填；② RNG 是每 Rank 状态，同 World Size 各取槽位，重分片时新 Rank r 取 `r % W_old`。全部写盘走 `*.tmp` + `os.replace` 原子替换——被 Kill 的进程不可能留下会被恢复信任的半写文件。恢复即验证：loss、LR、下一 batch 预览、RNG 指纹、最终参数校验和五样对照，同构恢复要求逐位相等。

**数字**（README ch43）：0.6B FP32 Checkpoint ≈ 7.2GB/副本；sharded 不省字节（总体积约等于 centralized），收益只在不并行就得不到的 I/O 时间；4 卡保存 → 2 卡加载重分片可跑。

**边界与易错**：异步/影子 Checkpoint、远端对象存储、ZeRO/FSDP 的真重分片（字节级实切）都排除。[易错] "存个参数就能续训"——LR、Sampler、RNG、拓扑缺一个，续跑轨迹就分叉；平坦 Warm-up 恰好会掩盖 LR 回填错误。

**产品对应**：`torch.distributed.checkpoint`（DCP）、DeepSpeed 的 save_checkpoint、HF Trainer 的 resumable checkpoint `[业界通识]`。

**方案视角**：现场先问"恢复时间目标（RTO）是多少"——它决定保存频率（保存时间占训练时间的比例）和模式：吞吐敏感用 sharded 并行写，网络出口受限再谈集中式。

**讲给客户的一句话**：Checkpoint 保的不是权重一个文件，而是"断在哪一步、学到什么状态、随机数走到哪"的全套现场——目标只有一个：从最近快照逐位接上，一刻不从头再来。

**掌握目标**：能列出 Checkpoint 保存清单的七类内容；能答出"4 卡存的 Checkpoint 能否被 2 卡加载"及 DDP 下它为什么合法。

**深挖入口**：帖子未发，读仓库 chapter_43

<!--fig:ch43-->
