# 训练篇 B：从数据并行的通信优化到三维并行与容错（ch30–ch43）

### ch30 · DDP 梯度分桶与通信计算重叠

**先回答一个问题：为什么第 29 期的 DDP"慢"在哪个环节？**

DDP（Distributed Data Parallel，数据并行：每张卡持有完整模型、各吃一份数据，最后把大家的梯度加起来取平均）。它的梯度汇总靠 AllReduce（全体互相交换并求和，最后每张卡都拿到同一个总和的集合通信）。第 29 期的做法是：Backward 全部算完，再一次性对整份梯度做 AllReduce——通信排队在计算之后，整段通信时间都暴露在关键路径上（所谓关键路径，就是决定总时长的那条串行链条）。真实的 DDP 不这样工作：参数被分成若干桶（Bucket，把相邻参数打包成的通信单位），某个桶的梯度一齐就绪时就立刻发起异步 AllReduce，与后续 Backward 计算重叠。本期只改变一件事：AllReduce 的启动时机。

<!--fig:ch30-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 梯度 | "往哪边调参数能让 loss 变小"的方向值，Backward 逐层算出 |
| Bucket | 把相邻参数的梯度打包成一个桶，攒齐一桶发一桶 |
| 异步 AllReduce | 发起通信后不干等，继续算 Backward，晚点再取结果 |
| 关键路径 | 决定整步耗时的串行链条；藏进链条里的时间等于白省 |
| 尾部等待 | 最后一个桶算完还得等通信收尾的那段时间 |

**为什么需要它**：想象你在食堂打饭——要么所有菜都点完再一起结账（第 29 期），要么点一个菜结一次、结账的同时继续点（本期）。当某层梯度算好时，它的通信根本不必等后面的层。

**怎么工作（拆到计算步骤级）**：整个机制只让"通信别排在计算后面"，拆成三步：
1. **装桶**：`build_buckets` 按**参数逆序**装桶——backward 从最后一层往回算，逆序装桶让"先就绪的梯度"恰好聚在同一个桶里；参数粒度打包，不跨桶拆分。
2. **桶满即发**：`register_post_accumulate_grad_hook` 在每个参数梯度算好的瞬间触发——拷入桶缓冲、计数器加一；桶内梯度一齐就绪，立刻以 `async_op=True` 发起 AllReduce，**不等**还没算完的层。
3. **重叠与收尾**：后续 backward 计算与已发出的 AllReduce 并行跑；`wait_and_attach` 在尾部等所有通信完成，把结果以视图挂回参数。baseline（第 29 期）是把第 2、3 步整体排在 backward 之后，通信零重叠。

桶的大小决定重叠质量——太小则每桶的固定启动延迟加起来反超省下的暴露时间；太大退化为第 29 期的全暴露形态。

**算一算**：每步约 2.4GB 梯度要过 AllReduce。PCIe 3.0 x16 的有效带宽大约 12–16 GB/s `[业界通识]`，全暴露时单步通信 ≈150–200ms 排在关键路径上；重叠成功时这一百多毫秒的大头藏进 backward 计算——这就是分桶的赌注。

**显存账**：底数是每步 2.4GB FP32 梯度；分桶额外付桶缓冲与"已发未收"桶的临时显存，桶越大临时显存越高（README 把临时显存列为桶档位的权衡项之一）。

**数字**（README ch30）：benchmark 扫 1/16/128/1024 MiB 四档桶，3 次重复；假设前提：PCIe PHB、无 NVLink，通信与计算共享带宽，收益可能有限。

**边界与易错**：收益依赖通信带宽与计算能否真的重叠；桶大小是权衡不是越大越好。[易错] "分桶一定更快"——桶过小时每桶的启动延迟加起来比省下的暴露时间还多。

**产品对应**：PyTorch DDP 内置的 gradient bucket（默认 bucket 化）`[业界通识]`。

**方案视角**：现场先问"卡间走的是什么互联？"——NVLink 与 PCIe 的重叠收益差一个量级；再看单步梯度量，决定桶档位。

**讲给客户的一句话**：数据并行的通信不必干等——按桶流水发出去，让通信和计算同时进行。

**掌握目标**：能画出"Backward 中按桶异步 AllReduce"的时序图并与一次性 AllReduce 对比；能答出桶过大、过小分别退化成什么。

**深挖入口**：帖子未发，读仓库 chapter_30 README

### ch31 · 梯度累积期间延迟同步（no_sync）

**先回答一个问题：当一个 Optimizer Step 由 K 个 Micro-step 组成时，前 K−1 次同步是浪费吗？**

是。Micro-step（微步）指把一个大批次切成 K 份、逐份算梯度但不更新参数——这叫梯度累积，显存装不下大批次时的标准做法。每微步都做一次全量 AllReduce 是极大的浪费：前 K−1 次可以纯本地累积（no_sync 模式），最后一次统一同步，梯度数学不变、通信量降为 1/K。本期只改变一件事：微步间同步的时机（每微步同步 对照 仅最后一次同步）；Bucket 不用，AllReduce 仍是 Backward 结束后的平铺 SUM。

<!--fig:ch31-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 梯度累积 | 切 K 份小批次逐个算梯度、攒够再更新一次参数 |
| no_sync | 累积期间不做跨卡同步，各卡只攒自己的梯度 |
| 独立累加器 | 存"每微步 AllReduce 结果之和"的缓冲区 |
| 监督 Token | 真正参与 loss 计算的 Token（不含 padding） |
| token_mean | loss 按全局监督 Token 数取平均的口径 |

**为什么需要它**：K=2 时每微步同步要做 2 次全量 AllReduce，而数学上只需要 1 次——一半的通信是白花的。

**怎么工作（拆到计算步骤级）**：把两种模式的执行序并排走一遍。
- **baseline `every`**：①微步 k 本地 backward → ②全量 AllReduce SUM → ③结果写入**独立累加器**；K 个微步重复 K 轮，共 K 次全量 AllReduce。
- **variant `last`**：①微步 k 本地 backward，梯度只加在本卡累积缓冲，no_sync 期间零跨卡通信；②K 个微步全部走完后，一次 AllReduce SUM 统一同步。
- 两种模式每微步的 loss 都按该微步监督 Token 占比缩放，求和等于全局 token_mean 梯度——**数学不变，通信量降为 1/K**。[易错] 把每微步的 AllReduce 结果直接加进已被归约的累积缓冲再反复 AllReduce，会把先前微步的贡献放大 World Size 倍——必须用独立累加器。

**显存账**：每卡梯度仍是 2.4GB 全量驻留——no_sync 不省显存，省的全是通信（全量 AllReduce 从每步 K 次降到 1 次）。

**算一算**：每步全量 AllReduce 的字节数约 2.4GB（FP32 0.6B 梯度），K 个微步下节省 = (K−1)/K × 2.4GB——K=2 省 1.2GB、K=4 省 1.8GB 的卡间传输。

**数字**（README ch31）：主对照 K=2（全局 batch 4），容量对照 K=4（全局 batch 8，每卡每微步 1 样本）；K=2 时 no_sync 省一次全量 AllReduce，即每步通信量 1/K。

**边界与易错**：省的是通信不是计算，backward 计算量不变；K 越大每微步 batch 越小、算力利用率越低。[易错] "延迟同步会改变梯度"——正确实现下两种模式梯度逐位（FP32 结合律容差内）一致。

**产品对应**：PyTorch DDP 的 `model.no_sync()` 上下文、HF Trainer 的 `gradient_accumulation_steps` `[业界通识]`。

**方案视角**：现场先问"显存允许的单步 batch 多大？"——不够就得累积，累积就该配 no_sync；顺带确认框架是否真的关掉了微步间同步。

**讲给客户的一句话**：小卡训大 batch，靠梯度累积；累积期间关掉跨卡同步，通信量直接除以累积份数。

**掌握目标**：能算出 K 个微步下 no_sync 的通信量节省；能答出"对已归约缓冲反复 AllReduce"错在哪里。

**深挖入口**：帖子未发，读仓库 chapter_31 README

### ch32 · ZeRO-1：切分优化器状态

**先回答一个问题：DDP 每张卡上，最占显存但"最不该人手一份"的是什么？**

优化器状态。AdamW 给每个参数存两个动量（一阶矩、二阶矩，可理解为"最近梯度的滑动平均"和"滑动平方平均"），FP32 下是参数字节数的 2 倍。DDP 每卡持有完整模型、完整梯度和完整优化器状态——0.6B 模型在 12GB 卡上已足以形成容量压力。本期只改变一件事：FP32 优化器状态是否在 Data Parallel Rank 间切分（ZeRO-1；Shard 指每卡只留 1/W 的分片）。参数和梯度仍每卡完整。

<!--fig:ch32-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 优化器状态 | AdamW 每参数两份动量，更新参数用的"记忆" |
| ZeRO | 零冗余优化器：把训练状态切到各卡，消除人手一份的冗余 |
| Shard | 一份数据被切成 W 片，每卡只持一片 |
| AllGather | 各卡把各自分片拼回完整数据、人人拿全量的集合通信 |
| Rank | 分布式训练里每张卡的编号 |

**为什么需要它**：显存账（README 相对关系口径）：0.6B FP32 下参数 2.4GB、AdamW 状态 4.8GB（见 ch43 账本）——优化器状态是参数的 2 倍。复制式 DDP 里这 4.8GB 每卡一份、W 份全同；切分后每卡只留 1/W。0.6B 尚且如此，几十 B 的模型下这笔账就是"训得动 vs 训不动"。

**怎么工作（拆到计算步骤级）**：AdamW 是逐元素更新，派生出三步机制：
1. **切分**：`shard_boundaries` 把参数连续切块、余数给前几个 Rank；"本 Rank 状态分片 × 参数分片 × 梯度分片"与全量 AdamW 逐位一致——切了不等于改了数学。
2. **只为分片配状态**：`ShardedStateAdamW` 仅给本地分片分配 exp_avg/exp_avg_sq，`step()` 逐操作镜像 torch AdamW 公式，只更新本卡拥有的那段参数。
3. **分块 AllGather 同步新参数**：`all_gather_parameters`（块缓冲 128MB、非整除按 padded 长度打包）把更新后的参数同步给其他 Rank——DDP 原本没有的约 1×参数字节的额外通信。

**显存账**：0.6B FP32 下参数 2.4GB、梯度 2.4GB、AdamW 状态 4.8GB（见 ch43 账本）——复制式 DDP 每样全量；切分后状态降为 4.8/W GB，参数与梯度不动。

**算一算**：W=2 时 AdamW 状态从每卡 4.8GB → 2.4GB，省 2.4GB；付出每步约 2.4GB（参数字节）的 AllGather——用便宜的通信换昂贵的显存。

**边界与易错**：多付通信，小模型上可能不划算；参数、梯度仍全量，更大的模型还得继续切（ch33/34）。[易错] "ZeRO-1 改变了数学"——分片更新与全量 AdamW 逐位一致，语义不变。

**产品对应**：DeepSpeed ZeRO Stage 1、FSDP 的 optimizer state sharding `[业界通识]`。

**方案视角**：现场先问三个数：卡数、单卡显存、卡间带宽。显存是第一瓶颈且带宽尚可时先上 ZeRO-1；用 capacity_probe 式探针找到 Batch/Length 准入边界再定服务形态。

**讲给客户的一句话**：优化器状态是训练显存的大头，把它按卡数切分，同样的卡能装下更大的 batch 或更长的序列。

**掌握目标**：能算出 W 张卡下 ZeRO-1 的状态显存与新增 AllGather 字节；能判断给定显存下该不该上 ZeRO-1。

**延伸阅读**：[ZeRO (arXiv 1910.02054)](https://arxiv.org/abs/1910.02054)——优化器状态冗余的显存分析与 Stage 划分的原始出处

**深挖入口**：帖子未发，读仓库 chapter_32 README

### ch33 · ZeRO-2：再切分梯度

**先回答一个问题：backward 结束那一刻必须存在的完整梯度，之后都用得上吗？**

用不上。ZeRO-1 之后每卡仍保留完整梯度，但本卡的优化器只会用到"属于本卡参数分片"的那一份。本期只改变一件事：梯度是否切分——完整 AllReduce 对照 Reduce-Scatter（归约求和的同时顺手分发：每卡只拿到总和里属于自己的那一片）。优化器状态沿用 ZeRO-1 切分，参数仍每卡完整——所以是 ZeRO-2 而不是 ZeRO-3。

<!--fig:ch33-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Reduce-Scatter | 各卡数据求和后按编号分发：人人拿到总和的 1/W |
| 梯度分片 | 本卡只保留"本卡参数分片"对应的梯度片 |
| 全局范数 | 所有参数梯度平方和开根号，梯度裁剪用 |
| 裁剪 | 梯度范数超阈值就按比例缩小，防训练爆炸 |
| 容量准入边界 | 给定显存下能跑的 Batch/长度上限 |

**为什么需要它**：backward 结束那一刻必须存在的完整梯度，之后只有本卡那一份会被用到——省下就是白赚。连同 ch32 的状态分片，每卡省下的显存直接换成更大的 batch——准入边界外移。

**怎么工作（拆到计算步骤级）**：梯度从此"按所有权分发"，四步：
1. **backward 产梯度**：autograd 逐参数算出本卡完整梯度（这一步不变）。
2. **Reduce-Scatter 只留本 rank 分片**：`reduce_scatter_gradients` 逐参数发起——整除时直接以梯度视图为输入，完成后立即释放完整梯度；非整除按零填充，填充位求和仍为零，不污染结果。
3. **本地 optimizer step + 分块 AllGather 同步参数**：与 ZeRO-1 完全相同的 `ShardedStateAdamW` 分片更新与参数刷新。
4. **全局范数裁剪**：梯度切分后 `clip_grad_norm_` 直接失效——各卡只有局部范数，必须"各卡局部平方和 → 一次标量 AllReduce → 合成全局范数 → 按比例缩放本地分片"。

关键等式：Reduce-Scatter 与 AllReduce 的每链接数据量同为 2(W−1)/W·D，W=2 时通信时间接近——**同样的通信字节数，免费换来"只留 1/W 梯度"**（等价替换，不是新开销）。

**显存账**：完整梯度 FP32 下 0.6B 是 2.4GB；Reduce-Scatter 后每卡只留 2.4/W GB。

**算一算**：W=2 每卡省 1.2GB 梯度显存、W=4 省 1.8GB（2.4×(1−1/W)）；通信侧不亏——Reduce-Scatter 与 AllReduce 的每链接数据量同为 2(W−1)/W·D，W 越大两者同步下降，省显存不减通信时间。

**数字**（README ch33）：固定项沿用 2 卡配置；capacity_probe 扫 ddp/zero1/zero2 × Micro-batch {1,2,4} × Length {128,256,512} 的准入网格；假设 ZeRO-2 应比 ZeRO-1 更晚 OOM。

**边界与易错**：真实 ZeRO 的扁平桶化 Reduce-Scatter 本期不做，逐参数发起的通信代价如实测量；参数仍完整常驻。[易错] "Reduce-Scatter 是新的通信开销"——W=2 时它与 AllReduce 数据量相同，不是额外开销而是等价替换。

**产品对应**：DeepSpeed ZeRO Stage 2、FSDP 的 grad sharding `[业界通识]`。

**方案视角**：现场先问"卡数/显存/网络"三件套；ZeRO 级别每升一级，显存多省一截、通信结构多一点复杂——用准入边界实测值而不是参数量拍脑袋选 Stage。

**讲给客户的一句话**：梯度算完只有 1/W 有用——Reduce-Scatter 让每卡只留自己那份，同样的卡再挤出一份显存。

**掌握目标**：能算出 W=2/4 时每卡的梯度显存节省；能答出分片后全局范数怎么合成。

**延伸阅读**：[ZeRO (arXiv 1910.02054)](https://arxiv.org/abs/1910.02054)——Stage 2 的通信量分析：Reduce-Scatter 与 AllReduce 同量级

**深挖入口**：帖子未发，读仓库 chapter_33 README

### ch34 · ZeRO-3 / FSDP：再切分参数

**先回答一个问题：参数只在 forward/backward 里被读，凭什么在两步之间整份占着显存？**

不该占。ZeRO-3/FSDP（Fully Sharded Data Parallel，全分片数据并行）让训练参数也只以 Shard 形式常驻每个 Rank，用的时候再临时拼全量。至此每卡常驻模型状态从 ZeRO-2 的 `2N+14N/W` 降到 `16N/W` 字节（README 的 FP32 账本口径，N 为参数量、W 为卡数）。本期只改变一件事：参数是否分片，对照 ZeRO-2。

<!--fig:ch34-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| AllGather 参数 | forward 前把各卡参数分片临时拼成全量 |
| Reshard | 用完全量立刻释放全量存储，退回分片状态 |
| 热路径 | forward/backward 主计算流程，通信进去就要排队 |
| no-grad 区 | autograd 不记录的代码段，纯搬运用 |
| tied 权重 | Embedding 与 LM Head 共用同一个参数 |

**为什么需要它**：显存账（README 口径）：ZeRO-2 每卡 `2N+14N/W`，ZeRO-3 `16N/W`——W 越大差距越大。0.6B 小模型上差异不明显；参数量上去后这是"单机装不下 → 必须多机"还是"单机多卡即可"的分界。

**怎么工作（拆到计算步骤级）**：参数从此有了完整生命周期，四步一个循环：
1. **分片常驻**：训练参数平时只以 Shard 形式躺在每 Rank；forward 需要参数时，在 autograd 不可见的 no-grad 区里 AllGather 拼出全量权重。
2. **算完即 Reshard**：本单元的前向计算结束后真正释放全量存储、退回分片——不是"标记可回收"。
3. **backward 前再 AllGather**：权重梯度要用全量权重，按需再拼一次；算出的权重梯度当场 Reduce-Scatter，只有本 Rank 分片落进 `.grad`。
4. **分片 Optimizer Step**：与 ZeRO-1/2 相同的分片 AdamW；**下一个 forward 的 AllGather 本身就是参数刷新**——ZeRO-1/2 那次 post-step AllGather 被取消。
代价：AllGather 从"每步一次"搬进了热路径；0.6B 上 Step Time 大概率变长，README 明确保留这个负优化结果。
**显存账**：W=4、N=0.6B（FP32）时：ZeRO-2 每卡 2N+14N/W = 1.2+2.1≈3.3GB 常驻模型状态，ZeRO-3 = 16N/4 = 2.4GB——参数、梯度、状态全按 1/W 等分。

**算一算**：按公式代值（N 取 0.6GB 的口径）：W=8 时 ZeRO-2 每卡 2N+14N/8 ≈ 2.25GB，ZeRO-3 = 16N/8 = 1.2GB；W=4 时 3.3GB vs 2.4GB——差距随 W 线性放大，0.6B 上不算悬殊，几十 B 模型上就是"单机多卡 vs 必须多机"。

**数字**（README ch34）：参考配置 4 卡、覆盖双节点 4+4（W=8）；benchmark 扫 W ∈ {2,4,8}；显存报告与理论账本（`2N+14N/W` vs `16N/W`）同列；FSDP 的 Reduce-Scatter 取平均，本地 loss 乘 W 校正后与全局 token_mean 一致。

**边界与易错**：通信进热路径，带宽差时纯负优化；激活重算与 CPU Offload 本期排除。[易错] "ZeRO-3 一定比 ZeRO-2 快"——它是容量方案不是速度方案，小模型上大概率更慢。

**产品对应**：PyTorch FSDP（`FULL_SHARD`、`use_orig_params`、`BackwardPrefetch`）、DeepSpeed ZeRO Stage 3、MiCS/HF Accelerate 的分片封装 `[业界通识]`。

**方案视角**：现场先问"卡数/显存/网络"：模型单卡放不下才谈 ZeRO-3；卡间带宽弱时先问"能不能用 ZeRO-2 + 更小 batch 顶住"。

**讲给客户的一句话**：ZeRO-3 是"装得下"的方案——参数也按卡切片，用的时候再临时拼，代价是热路径里多两次通信。

**掌握目标**：能写出 ZeRO-2 与 ZeRO-3 的每卡常驻字节公式并算出 W=4 的差值；能画出参数"分片→AllGather→Reshard"生命周期图。

**延伸阅读**：[ZeRO (arXiv 1910.02054)](https://arxiv.org/abs/1910.02054)——Stage 3 与 16N/W 账本的出处

**深挖入口**：帖子未发，读仓库 chapter_34 README

### ch35 · MLP Tensor Parallel

**先回答一个问题：ZeRO 切的是"训练状态"，单层内部的矩阵运算还能切吗？**

能。Tensor Parallel（TP，张量并行：把一层内部的权重矩阵切开，各卡算自己那块，再拼起来）。本期只切 SwiGLU MLP：Gate/Up 权重沿输出维切（Column Parallel，输入复制、输出分片），Down 权重沿输入维切（Row Parallel，输入分片、输出 AllReduce 求和）。Attention、Embedding、LayerNorm 保持每卡完整，单独测清 MLP TP 这一个机制。

<!--fig:ch35-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Column Parallel | 权重按列切：各卡算不同的输出通道 |
| Row Parallel | 权重按行切：各卡吃不同的输入切片，输出求和 |
| f 算子 | forward 恒等、backward 做 AllReduce 的通信占位符 |
| g 算子 | forward 做 AllReduce、backward 恒等的通信占位符 |
| SwiGLU | Qwen3 的 MLP 结构：silu(Gate)×Up 再过 Down |

**为什么需要它**：单卡显存装不下"单层的权重+激活"时，就得把层切开。0.6B 单卡本可完整常驻，所以 TP 在本实验里更可能是负优化——这是课程预期内的合法结果，负优化如实保留。

**怎么工作（拆到计算步骤级）**：一个 forward 一个 backward，四步：
1. **输入复制、切权重算局部**：每 Rank 拿**相同**的输入 x，算自己分片的 `G_r = x@W_gate_r^T`、`U_r = x@W_up_r^T`、`H_r = silu(G_r)·U_r`、`P_r = H_r@W_down_r^T`——Gate/Up 沿输出维切（Column Parallel），Down 沿输入维切（Row Parallel）。
2. **forward 一次 AllReduce（g 算子）**：各自的部分和 `P_r` 求和成完整输出 y——每层前向的通信。
3. **backward：分片权重梯度全留本地**：`dW_down_r = dY^T@H_r`、`dW_gate_r`、`dW_up_r` 只依赖本 Rank 分片前向和复制的 x；各 Rank 分片梯度按顺序拼接就是完整梯度——**不需要也不应该 AllReduce**。
4. **backward：残差流输入梯度一次 AllReduce（f 算子）**：真实梯度是 `Σ_r dX_r`；down_proj 前向 AllReduce 的 backward 是恒等、补不上这一步——漏掉它，"部分梯度"沿残差流流进 Attention 路径，训练语义被静默改坏。

第三类陷阱：复制参数（Attention、LayerNorm）的梯度各卡**天然相同**（forward AllReduce 后激活逐位一致），再做 AllReduce SUM 会恰好放大 tp_size 倍——是错误不是浪费。三类梯度三种处理，是 Megatron 张量并行的核心纪律 `[业界通识]`。

**显存账**：MLP 的 Gate/Up/Down 权重与其 AdamW 状态按 1/tp_size 与其他卡分摊，Attention/Embedding/LayerNorm 仍全量复制；README 的所有权账本按"复制参数/分片参数/状态"三类同列。

**算一算**：tp=2 时 MLP 参数与其 AdamW 状态每卡各减半（省 MLP 参数 × 6 字节 FP32 + 动量的一半）；Attention/Embedding 逐字节不变——所以 0.6B 上挤出不多，模型越大 MLP 占比越高收益越大。tp=4 再减半，但每层 2 次 AllReduce 也在暴露的链路上排队。

**数字**（README ch35）：每层新增 2 次 TP AllReduce（forward down_proj 输出、backward MLP 输入梯度），全部暴露在关键路径上；对照臂 tp1/tp2/tp4，正确性红线 loss < 1e-5。

**边界与易错**：TP 通信暴露不重叠，收益依赖高速互联；每层两次 AllReduce 随层数线性累积。[易错] "所有梯度都要 AllReduce"——分片权重不需要，复制参数更不能做，三类梯度三种处理。

**产品对应**：Megatron-LM 的 MLP 并行、vLLM/SGLang 的 TP 推理同源机制 `[业界通识]`。

**方案视角**：现场先问"卡间互联带宽够不够 TP"——TP 的 AllReduce 每层两次且不能重叠，NVLink 机器和 PCIe 机器结论完全不同。

**讲给客户的一句话**：TP 把一层内部的计算切开分给多卡，但每层要付两次"全员对账"的通信——只有高速互联才付得起。

**掌握目标**：能画出 MLP TP 的 f/g 算子位置；能区分三类梯度的通信要求并答出"复制参数再 AllReduce 会放大几倍"。

**延伸阅读**：[Megatron-LM (arXiv 1909.08053)](https://arxiv.org/abs/1909.08053)——f/g 通信算子与 Column/Row Parallel 的原始出处

**深挖入口**：帖子未发，读仓库 chapter_35 README

### ch36 · Attention Tensor Parallel

**先回答一个问题：MLP 切完了，Attention 的"头"怎么切？**

按 Head 切。Qwen3-0.6B 用 GQA（Grouped-Query Attention：多个 Q 头共享一组 KV 头，省 KV 显存的注意力结构）：32 个 Q 头、8 个 KV 头（4:1）、head_dim 128。TP=2 时每 Rank 拿 16 个 Q 头 + 4 个 KV 头，TP=4 时 8+2——切分必须满足整除约束，每 Rank 拿整数个完整 Head；head_dim 永不切。

<!--fig:ch36-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Q/KV Head | 注意力的"视角"：Q 头提问，KV 头提供检索内容 |
| GQA | 多个 Q 头共享一组 KV 头，Q:KV = 4:1 |
| RoPE | 给 Q/K 加位置信息的固定旋转，Head 内操作 |
| q/k norm | Qwen3 对每个 Head 内部做 RMSNorm |
| repeat_kv | 把 KV 头复制 4 份对齐 Q 头的操作 |

**为什么需要它**：0.6B 的 Attention 参数占比小，预期收益有限甚至为负——但机制必须掌握：这是大模型 TP 的标准组成。

**怎么工作（拆到计算步骤级）**：在 MLP TP 基础上，Attention 补做四步：
1. **按 Head 切投影**：q/k/v 投影按 Head 行切（本 Rank 只算自己的 16+4 份投影）、o_proj 切对面的列；RoPE 是 Head 内的固定旋转、`repeat_kv` 只涉及本 Rank 的 Q/KV 对——都在**局部 Head 集合内重放**，与全局逐位等价。
2. **Attention 部分和不归约**：本 Rank 的注意力输出就是部分和，直接交给 MLP 段——归约只发生在块边界，块内不对账。
3. **每层通信账**：forward 2 次（o_proj 输出、down_proj 输出）+ backward 4 次（两个块输入梯度 + q/k norm 同步）——从 mlp_tp 的 1+1 变成 2+4。
4. **本期特有的陷阱——q/k norm**：这两个 norm 权重每卡**复制**，但各卡只累积**自己那几个 Head** 的梯度项——原始梯度是部分和，各卡不同；不处理则各卡对同一参数施加不同更新，副本从第一次 optimizer step 起静默分叉。实现用 post-accumulate-grad 钩子对每个 norm 权重做一次 backward AllReduce SUM。

**显存账**：Attention 的 q/k/v/o 权重及其 AdamW 状态按 Head 数等分成 1/tp_size（TP=2 时 Q/KV 都减半，局部 GQA 组数恒为 4），LayerNorm/Embedding 仍复制。

**算一算**：TP=2 → 每 Rank 32/2=16 个 Q 头、8/2=4 个 KV 头；TP=4 → 8+2。无论切几份，每 Rank 的"局部 GQA 组数"都是 4 个 Q 头共享 1 组 KV——组结构在切分下守恒。

**数字**（README ch36）：32 Q/8 KV Head，TP=2 → 16/4，TP=4 → 8/2，局部 GQA 组数恒为 4；attn_tp 臂每层 6 次 AllReduce 对 mlp_tp 的 2 次。


**边界与易错**：Head 数与 tp_size 的整除约束（32/8 都能被 2、4 整除）；0.6B 上收益小。[易错] "RoPE 要在全局上做"——它是 Head 内固定旋转，局部重放逐位等价。

**产品对应**：Megatron-LM 与 vLLM/SGLang 的 Attention TP、GQA 布局 `[业界通识]`。

**方案视角**：现场先问"卡间互联带宽够不够 TP"，再看模型 Head 数能否被 TP 度整除——不整除的模型配不出来的 TP 度直接排除。

**讲给客户的一句话**：Attention 按"头"切是多卡并行的天然切法，但每层通信次数翻倍，对互联带宽的要求比只切 MLP 更高。

**掌握目标**：能算出 TP=2/4 下每 Rank 的 Q/KV 头数；能答出 q/k norm 的梯度为什么是部分和、怎么修。

**延伸阅读**：[Megatron-LM (arXiv 1909.08053)](https://arxiv.org/abs/1909.08053)——按 Head 切分 Attention 的自注意力并行化推导

**深挖入口**：帖子未发，读仓库 chapter_36 README

### ch37 · 词表并行 Embedding 与分布式 Cross Entropy

**先回答一个问题：TP 模型里最大的单项激活是什么？**

不是注意力，是 Logits。Qwen3-0.6B 词表 151936；batch 2 × seq 512 的完整 Logits 是 151936 × 1024 × 4B ≈ 622 MiB——TP 模型里最大的单项激活。本期把 Embedding/LM Head 沿词表维切分（TP=2 每 Rank 75968 行），并用不物化完整 Logits 的分布式 Cross Entropy 闭合训练。

<!--fig:ch37-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Logits | 对全词表每个 Token 打的分数，CE 的输入 |
| Cross Entropy（CE） | 把分数变成"下一个词"的 loss 的标准算法 |
| log-sum-exp | CE 里的归一化分母，可分块计算 |
| Vocab Parallel | 词表按行切开，每 Rank 持有一段 |
| tie | Embedding 与 LM Head 共享同一个权重矩阵 |

**为什么需要它**：622MiB 的完整 Logits 在长序列/大 batch 下线性膨胀，常常是 OOM 的元凶；分片后每 Rank 只物化自己的 1/tp_size 段。

**怎么工作（拆到计算步骤级）**：CE 的两个数学性质让"不物化全 Logits"成为可能：log-sum-exp 可按词表任意分块归约；softmax−onehot 的 backward 只需要本 Rank 分片和三个全局标量。逐步走：
1. **词表切行**：Embedding/LM Head 沿词表维每个 Rank 持 75968 行（tie 共享同一分片参数）；Embedding 查表时越界 token mask 成 0、输出一次 AllReduce 求和，人人拿到完整隐状态。
2. **分片 logits**：LM Head 只算本 Rank 分片的 logits，完整 Logits 从 622MiB 降到 311MiB。
3. **分布式 CE 的 3 次 forward Collective**：全局 max 用 AllReduce **MAX**、全局 exp 和用 AllReduce **SUM**、目标 token 的 logit 按所属 Rank 用一次 AllReduce **SUM** 取回——三次 Collective 的尺寸都是 token 数而不是词表数，几乎不占带宽。
4. **backward 0 通信**：`(softmax − onehot)` 只用本 Rank 分片和三个全局标量即可算出——CE 本身零反向通信。真正要补的是 LM Head 输入梯度 `dlogits_r @ W_r` 是部分和，需要一次 backward AllReduce（f 算子）——漏掉它，LayerNorm 梯度跨 Rank 不一致、tied Embedding 梯度对照失败。tie 的 lookup+logits 两条路径梯度自动累进同一个分片参数。

**显存账**：批 2 × seq 512 下的完整 Logits ≈ 622MiB 是 TP 模型最大的单项激活，长序列/大 batch 下线性膨胀、常是 OOM 元凶；再加上 Embedding/LM Head 权重与其状态按 1/tp_size 分摊。

**算一算**：152.0K 词表 × (批 2 × 512 tokens = 1024 个位置) × 4B ≈ 622 MiB；TP=2 只物化一半 ≈ 311MiB，比值恒为 1/tp_size。

**数字**（README ch37）：词表 151936、TP=2 每 Rank 75968 行、batch 2×seq 512 完整 Logits ≈ 622 MiB；vocab_tp 臂 forward 每层 2 次块输出 + embedding 1 次 + CE 3 次。

**边界与易错**：0.6B、tp=2 下收益可能仍为负——负优化如实保留；词表必须被 tp_size 整除。[易错] "分布式 CE 的 backward 很贵"——恰相反，它 backward 0 通信；固定代价是 LM Head 输入梯度那一次 AllReduce。

**产品对应**：Megatron-LM 的 VocabParallelEmbedding 与分布式 CE、vLLM 的并行词表 `[业界通识]`。

**方案视角**：现场先问"卡间互联带宽够不够 TP"，再看序列长度 × batch × 词表的 Logits 账——长序列场景它是第一显存大项。

**讲给客户的一句话**：词表 15 万行的 Logits 是显存杀手——把它按卡切开、loss 各算各的再合，622MiB 直接变 311MiB。

**掌握目标**：能算出给定 batch/seq/词表下的完整 Logits 字节；能答出分布式 CE 的 3 次 forward Collective 和 backward 为何 0 通信。

**延伸阅读**：[Megatron-LM (arXiv 1909.08053)](https://arxiv.org/abs/1909.08053)——词表并行 Embedding 与分布式 softmax loss 的原文

**深挖入口**：帖子未发，读仓库 chapter_37 README

### ch38 · Sequence Parallel

**先回答一个问题：权重都切完了，还有什么在每卡重复？**

残差流和 LayerNorm 的激活。它们是完整的 `[batch, seq, hidden]`，与权重是否分片无关。本期在 Attention+MLP 权重分片不变的前提下，把这些非 TP 区域的激活沿 Sequence 维切分（Sequence Parallel，SP：激活按序列切段、每卡只存一段）。Attention 内部仍跑全序列；Context Parallel / Ring Attention 不在本期。

<!--fig:ch38-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 激活 | forward 中间结果，backward 要用所以驻留显存 |
| SP 区域 | TP 块之间做 LayerNorm/残差的部分 |
| g（SP 版） | 进 TP 块前 AllGather 拼全序列 |
| ḡ | 出 TP 块后 Reduce-Scatter 切回序列段 |
| saved-for-backward | 为反向保存的激活字节数 |

**为什么需要它**：长序列训练时激活是显存第一大头；权重切得再细，激活不切照样 OOM。

**怎么工作（拆到计算步骤级）**：一个 SP 区的进出是四次换手，一层重复一遍：
1. **分片上算 norm**：`x_r [b, s/N, h]` 上直接做 RMSNorm——它逐 Token 独立（均值只在 hidden 维）、残差逐元素，分片算等于全局算完再切片。
2. **AllGather 进 TP 块（g 算子）**：把 `s/N` 段拼成全序列交给 Attention/MLP——Attention 是唯一的序列耦合算子，仍在 TP 区域内部跑全序列。
3. **Reduce-Scatter 出 TP 块（ḡ 算子）**：Attention/MLP 输出的部分和切回每卡一段，残差加法仍在分片上——`x_r += out_r`。
4. **非 TP 权重梯度 hook 同步**：序列切分后每个 Rank 只看到自己的 Token，LayerNorm、最终 norm、Embedding/LM Head 的原始梯度从"天然一致"变成"部分和"——必须用钩子逐权重做 backward AllReduce SUM；漏掉任何一个，副本从第一次 optimizer step 起静默分叉。

两个反向细节：g 的 backward 是**本地切片**不是 Reduce-Scatter（f 已求和出一致的完整梯度，切片即可；对一致副本再 Reduce-Scatter 会放大 tp_size 倍）；ḡ 的 forward 是 Reduce-Scatter、backward 是 AllGather——两种"分散"语义不同，不能互换。Megatron 把 f 与 g 融合成一次 Reduce-Scatter 省一半反向字节，本期保留第 36 期形态、不做该融合。

**显存账**：saved-for-backward 的激活字节 SP < 纯 TP（README 用 `ActivationLedger` 实测两条线：峰值 allocated 与 saved-for-backward 字节，同 batch 两臂差值即 SP 收益）；权重布局与 ch36 完全一致不变。

**算一算**：SP 区域的激活（LayerNorm 输出、残差流等）从 `[b,s,h]` 缩到 `[b,s/N,h]`——相对纯 TP，这部分激活字节约省 1−1/N（主实验 N=2 时约省一半的 SP 区激活）；TP 块内部激活不动。0.6B、L=512 下余额小，L=2048 时才值得付边界通信。

**数字**（README ch38）：主实验 W=2、L=512；代价是每并行块多一次 AllGather + 一次 Reduce-Scatter；0.6B 激活占比小，端到端吞吐可能为负，收益预期首先出现在长序列容量（capacity_probe 扫 L 到 2048）。

**边界与易错**：短序列下边界通信吃掉收益；Megatron 把 f 与 g 融合成一次 Reduce-Scatter 省一半反向字节，本期不做。[易错] "SP 是新并行维度"——它只切 TP 区域之间的激活，序列维切分发生在 SP 区域。

**产品对应**：Megatron-LM Sequence Parallel、Megatron Context Parallel（扩展）`[业界通识]`。

**方案视角**：现场先问"卡间互联带宽够不够 TP"，再看训练序列长度——L 上千时激活账主导，SP 才值得付边界通信。

**讲给客户的一句话**：长序列训练的显存瓶颈在激活——把 LayerNorm 之间的激活按序列切开，权重布局不变。

**掌握目标**：能画出每层 SP 区域的 g/ḡ 位置；能答出"非 TP 权重梯度为什么从天然一致变成部分和"。

**延伸阅读**：[Megatron-LM (arXiv 1909.08053)](https://arxiv.org/abs/1909.08053)——序列并行思想的工程谱系起点

**深挖入口**：帖子未发，读仓库 chapter_38 README

### ch39 · GPipe Fill-Drain 流水

**先回答一个问题：模型按层切成两半放两卡，为什么反而更慢？**

因为一个 micro-batch 走流水时，任一时刻只有一张卡在算——另一张在等。Pipeline Parallel（PP，流水线并行：模型按层跨卡切成 Stage，数据像流水线一样流过）。本期把 28 层连续均分：PP=2 每 Stage 14 层、PP=4 每 Stage 7 层（0.6B 恰好整除）；唯一变量是多 micro-batch 的调度——逐个走完全流水（serial）对照先连续 Forward 再连续 Backward 的 GPipe Fill-Drain。

<!--fig:ch39-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Stage | 流水线上的一站：连续若干层+边界件 |
| Micro-batch | 把 batch 切小后逐个喂进流水线的单位 |
| Fill/Drain | 流水的填充与排空阶段，必然有空转 |
| Bubble | 流水空转的时间占比 |
| P2P | 相邻卡点对点直传激活与梯度 |

**为什么需要它**：PP 把模型放下了，但不调度就白切——serial 模式下 S 张卡的利用率只有 1/S。

**怎么工作（拆到计算步骤级）**：比起 serial，GPipe 只有调度变了，四个动作扫完一步：
1. **Fill（连做 M 个 forward）**：micro 1 的分段激活沿 Stage 0→1→…→S−1 的 P2P 流下去；Stage 0 不等 micro 1 走完全流水，紧接着发 micro 2 的分段——流水被逐段填满。
2. **Drain（连做 backward）**：末 Stage 从 loss 反传；其余 Stage 收到自己"输出激活的输入梯度"后 `autograd.backward(out, g)` 启动本段反传，把输入激活的 `.grad` 传回上一 Stage——梯度逆流排空。
3. **tied 梯度交换**：Qwen3 的 Embedding/LM Head 被 PP 拆在首末两个 Stage，各自累积不同的部分梯度——`exchange_tied_gradients` 反向结束后做一对 P2P（末→首、首→末），两侧都拿 `g_embed + g_lm_head` 再更新（每步 2 × vocab×hidden 字节）。
4. **更新**：所有 Stage 所有 micro 反传完成 + tied 交换后，各 Stage 用精确全局范数（1 次标量 AllReduce）裁剪、各自 AdamW step——与单卡逐参数一致。

baseline serial 没有第 1 步的"连续填充"：一个 micro 走完所有 Stage 才下一个，任一时刻只有一张卡在算，利用率 1/S。

**显存账**：激活图——serial 每 Stage 同一时刻只驻留 1 个 micro 的激活图，gpipe 要驻留 M 份（README 把这条"峰值显存差"列为机制性结论）。

**算一算**：填充-排空模型给出 Bubble 上界 = (S−1)/(M+S−1)。PP=2、M=8：1/9 ≈ 9%；M=2：1/3——micro 数直接决定流水效率；这也是显存账的另一面：压 Bubble 靠加 M，代价是驻留 M 份激活。

**数字**（README ch39）：28 层 PP=2 → [14,14]、PP=4 → [7,7,7,7]；micro 扫 {2,4,8}；serial 与 gpipe 峰值显存差（1 份 vs M 份激活图）是机制性结论。

**边界与易错**：0.6B 太小、P2P 走 PCIe，端到端加速可能被通信吃掉甚至为负。[易错] "PP=2 就快一倍"——没有流水调度是 1/S 的利用率，有调度还有 Bubble 与激活显存代价。

**产品对应**：Megatron-LM 的 GPipe schedule、torchgpipe `[业界通识]`。

**方案视角**：现场先问"模型层数和卡数怎么配 Stage"——层数要能被 Stage 数基本整除；再问 micro 数能不能开大，micro 太少流水白搭。

**讲给客户的一句话**：把模型按层切给多卡接力，再用多个微批把流水填满——卡越多切得越细，但"填不满"的空转要用微批数压下去。

**掌握目标**：能算出给定 S、M 的 Bubble 上界；能画出 Fill-Drain 时序并标出空转段。

**延伸阅读**：[GPipe (arXiv 1811.06965)](https://arxiv.org/abs/1811.06965)——Fill-Drain 调度与 bubble 分析的原始出处

**深挖入口**：帖子未发，读仓库 chapter_39 README

### ch40 · 1F1B 流水调度

**先回答一个问题：GPipe 的代价是"所有 micro 的激活图都攒在手里"，能不能边走边放？**

能。1F1B（one-forward-one-backward：稳态时每做一次 Forward 就做一次 Backward）把激活生命周期从"整个 Step"缩短到"一个 micro"。Fill-Drain 下每 Stage 从 Fill 开始驻留 M 份激活图；1F1B 下每 Stage 的 in-flight 激活峰值降到 **min(S−s, M)** 份（s 为本 Stage 编号、S 为 Stage 总数、M 为 micro 数）。本期唯一变量：Fill-Drain 对照非交错 1F1B，Stage 划分、micro 序列、P2P 协议完全相同。

<!--fig:ch40-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Warm-up | 开局连做若干次 Forward，凑出稳态需要的在飞深度 |
| Steady State | 一次 Forward + 一次 Backward 交替的主循环 |
| Cool-down | 结尾排空剩余 Backward |
| in-flight 激活 | 已 Forward 还没 Backward、必须驻留的激活图 |
| 1F1B | 稳态 F/B 交替的流水调度 |

**为什么需要它**：M 大时 GPipe 的 M 份激活图是显存灾难；1F1B 用"更早开始 Backward"换显存，Bubble 上界不变，仍服从 (S−1)/(M+S−1)。

**怎么工作（拆到计算步骤级）**：Stage s 的 Step 主体切成 Warm-up/Steady/Cool-down 三段，收尾与 ch39 相同；全部由 `(stage_index, S, M)` 公式决定，无需额外同步：
1. **Warm-up（连做 min(S−1−s, M) 个 forward）**：越靠前的 Stage 第一个 Backward 到达越晚，能提前做的 Forward 越多；`M` 封顶——micro 不够时 Warm-up 吃下全部，退化成 Fill-Drain 形状。
2. **Steady State（1F1B 交替）**：每轮按"收激活→Forward→收梯度→Backward→发梯度→发激活"执行 `M − warmup` 轮——**先收后发**是死锁关键：gloo 这类严格阻塞传输下，若先 send 激活再 recv 梯度，相邻 Stage 可能互相等成环；先收后发打断环（NCCL 异步入队两种顺序都行，所以 CPU/GPU 同一条代码路径）。
3. **Cool-down（排空 warmup 个 backward）**：稳态结束后清尾巴。
4. **收尾与更新与 ch39 完全相同**：tied 一对 P2P 交换 → 精确全局范数 → AdamW step——调度只重排时间，不改更新边界，梯度按 micro 序累加与 gpipe 逐位一致。

**显存账**：每个 Stage 的 in-flight 激活从 gpipe 的 **M 份**降到 **min(S−s, M) 份**（Warm-up 结束时驻留 warmup 份，稳态每轮先多拿一份再放掉一份）——这是 1F1B 的主要价值假设，benchmark 用每 Rank 峰值 allocated 对照。

**算一算**：S=4、M=4：末 Stage（s=3）Warm-up = min(0, 4) = 0，拿到激活直接开始 F/B 交替；首 Stage（s=0）Warm-up = min(3, 4) = 3，峰值在飞 min(4−0, 4)=4 份。

**数字**（README ch40）：in-flight 峰值 min(S−s, M) 对照 gpipe 的 M；benchmark 扫 micro {2,4,8}，W=2 主实验、W=4 对照；主要价值假设是峰值激活显存下降而非吞吐大增。

**边界与易错**：Bubble 不比 GPipe 小，收益全在显存；更多次收发等待在 PCIe 小模型上可能吃掉收益。[易错] "1F1B 更快"——它先解决"放得下"，其次才是速度。

**产品对应**：Megatron-LM 的 non-interleaved 1F1B schedule、DeepSpeed Pipeline Engine `[业界通识]`。

**方案视角**：现场先问"模型层数和卡数怎么配 Stage"，再看显存瓶颈在参数还是激活——激活主导时 1F1B 优先于加大 micro。

**讲给客户的一句话**：1F1B 让每张卡"算一个就还一个"激活，流水照样满，但显存占用从微批总数降到流水深度量级。

**掌握目标**：能算出给定 S、s、M 的 Warm-up 数与 in-flight 峰值；能画出 1F1B 时序并对照 Fill-Drain 标出显存差。

**延伸阅读**：[GPipe (arXiv 1811.06965)](https://arxiv.org/abs/1811.06965)——1F1B 之于它就是"按激活生命周期重排"的下一步

**深挖入口**：帖子未发，读仓库 chapter_40 README

### ch41 · Interleaved 1F1B（虚拟 Stage）

**先回答一个问题：流水级数 S 被物理卡数锁死，还能再"切细"吗？**

能——一个物理卡放多个非连续的虚拟 Stage（模型块）。虚拟 Stage s 位于物理 Rank `s % P`，total = P × vpp（P 为物理卡数、vpp 为每卡块数）。W=4、vpp=2 时 28 层切成 8 个虚拟 Stage [4,4,4,4,3,3,3,3]，每 Rank 合计仍 7 层、权重切分互补不重叠。流水级数从 P 增到 2P，Bubble 更短；代价是同一载荷要飞过 2P−1 条 Stage 边而非 P−1 条，P2P 次数约翻倍。

<!--fig:ch41-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 虚拟 Stage | 一张卡上的一个模型块，占流水线中的一级 |
| vpp | 每物理卡放的虚拟 Stage 数 |
| Placement 表 | 虚拟 Stage → 物理 Rank 的映射，唯一真相来源 |
| wrap 边 | 块边界从 P−1 绕回 Rank 0 的跨卡传递 |
| isend | 异步发送：投递后不等对方，先干别的 |

**为什么需要它**：Bubble 上界 (S−1)/(M+S−1) 里 S 越大 Bubble 越小；物理卡不够时，虚拟 Stage 是唯一能"加级数"的办法。

**怎么工作（拆到计算步骤级）**：把上一章的调度公式搬进"虚拟化 + 分组扫描"，四步：
1. **建 Placement 表**：虚拟 Stage s 位于物理 Rank `s % P`，是本 Rank 的第 `s // P` 块；`stage_owner` 表是唯一真相来源，所有收发先查表（28 层/P=4/vpp=2 → [4,4,4,4,3,3,3,3]，每 Rank 合计仍 7 层）。
2. **micro 按 P 分组扫描**：组内按块序扫 Forward、按块逆序扫 Backward；num_micro % P ≠ 0 直接报错不做静默重排。Warm-up 数 = min(2(P−1−r) + (vpp−1)P, vpp·M)。
3. **异步发送是死锁前提**：块边界依赖闭合成环（Rank A 下一块 Forward 依赖邻居 B 当前块 Backward，B 的这次 Backward 又依赖 A 上一块产物）——开发期 CPU smoke 用阻塞 send 实测死锁。实现统一 `dist.isend` 立即投递、步末统一 wait：所有阻塞等待都是 recv，链条必终于无 recv 的端点（Embedding 前向或 loss 反向），可证明无环。
4. **收尾不变**：所有反传完成后物理首末 Rank 一对 tied 梯度 P2P，精确全局范数（参数在所有虚拟 Stage 互斥，1 次标量 AllReduce）裁剪、各自 AdamW step。

**显存账**：每 Rank 常驻权重不变（仍是 7 层），变的是 in-flight 激活——Warm-up 数从上一章的 min(S−1−s, M) 涨到 min(2(P−1−r)+(vpp−1)P, vpp·M)——多存的正是"更快填流"垫的那几份。

**算一算**：P=4、M=8 时 P2P 激活消息数：interleaved (2×4−1)×8=60 条 vs 非交错 (4−1)×8=24 条——消息更小更密，次数与等待时间要分开读。

**数字**（README ch41）：total = P × vpp；W=4/vpp=2 下 8 块 [4,4,4,4,3,3,3,3]、每 Rank 7 层；micro 扫 {4,8}；warm-up 公式 min(2(P−1−r) + (vpp−1)P, vpp·M)。

**边界与易错**：0.6B + PCIe 上端到端可能为负；vpp>2 与自动负载均衡排除。[易错] "虚拟 Stage 免费"——它用约翻倍的 P2P 次数和更大的 in-flight 显存换 Bubble。

**产品对应**：Megatron-LM 的 interleaved 1F1B schedule（`--num-layers-per-virtual-pipeline-stage`）`[业界通识]`。

**方案视角**：现场先问"模型层数和卡数怎么配 Stage"——层数 ÷ (卡数 × vpp) 要基本整除；Bubble 仍是痛点时再评估 vpp=2 的通信增量。

**讲给客户的一句话**：给每张卡发两段不相邻的模型块，流水级数翻倍、空转更短——代价是卡间点对点传输次数约翻倍。

**掌握目标**：能算出给定 P、vpp、层数的块分布与 total Stage 数；能答出 Interleaved 为什么必须用异步发送。

**延伸阅读**：[GPipe (arXiv 1811.06965)](https://arxiv.org/abs/1811.06965)——Bubble 分析的起点；虚拟 Stage 是它之后 Megatron 谱系的延伸 `[业界通识]`

**深挖入口**：帖子未发，读仓库 chapter_41 README

### ch42 · 混合并行与双节点 4+4

**先回答一个问题：DP、TP、PP 各自学完了，生产环境怎么组合？**

三维并行（TP×PP×DP 同时启用）。本期不发明新原语，把三者拼起来并在双节点 4+4 上验证。拓扑由唯一的 rank 分解公式给出：`rank = dp_rank × (pp_size × tp_size) + pp_rank × tp_size + tp_rank`。TP=2、PP=2、DP=2（8 rank）时：TP 组 {0,1}{2,3}{4,5}{6,7}、PP 组 {0,2}{1,3}{4,6}{5,7} 都在节点内，DP 组 {0,4}{1,5}{2,6}{3,7} 跨节点——每个节点一份完整 TP2×PP2 模型副本，训练主路径只有 DP 的梯度 AllReduce 跨节点。

<!--fig:ch42-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| DP/TP/PP | 数据/张量/流水三种并行的组合维度 |
| Process Group | 一次集合通信的"参与人名单" |
| 逻辑 Rank | 一张卡在 (dp,pp,tp) 三维坐标系里的坐标 |
| Strong Scaling | 同一全局工作量下比较不同并行布局 |
| LOCAL_RANK | 一张卡在本节点内的插槽编号 |

**为什么需要它**：单维并行各有硬边界：TP 受限于节点内带宽、PP 受限于层数与 Bubble、DP 受限于显存；生产级训练（如 Llama 系列公开配置 `[业界通识]`）都是三维组合。

**怎么工作（拆到计算步骤级）**：三维并行的全部工作就是把"谁和谁通信"建表，再沿用已知机制：
1. **建表**：rank 公式 `rank = dp_rank × (pp_size × tp_size) + pp_rank × tp_size + tp_rank` 唯一确定每张卡在 (dp,pp,tp) 坐标系的位置；三族组表由此导出并逐 Rank 断言"恰好划分 world"。`dist.new_group` 是全局集合操作——每个 Rank 必须按相同顺序创建所有组、只保留自己所在的那个；只建"自己的组"会让其他 Rank 永久等待。PP 不建组，邻接 P2P 直接用全局 rank 寻址。
2. **数据只按 DP 切**：DP rank 按 stride 取分片再切成 num_micro 个 micro-batch；TP 组内每 Rank 同一 micro-batch（TP 切权重不切数据）；PP 按 micro 顺序流过 Stage。
3. **梯度三类不变 + DP 一次**：TP shard 权重梯度 rank-local；残差流输入梯度被 f 算子已在 TP 组内求和；复制参数在 TP 组内值相同（组内再 AllReduce 会乘上 tp_size，是错误）。DP 维度上所有参数只需**一次**梯度 AllReduce——跨节点的就是它。
4. **精确全局范数防重复计数**：先 TP 组归约 shard 部分，再全 world 归约后除以 tp_size——每个 (dp,pp) 单元恰好被计 tp_size 次；跳过 TP 归约会把 replicated 部分乘上 tp_size。PP=1 时 tied 权重就地绑定成单参数（首末同 Stage 的特例）。

**显存账**：每个节点一份完整的 TP2×PP2 副本——单卡常驻 = 模型/TP + Stage 层块（PP2 下每卡 14 层），DP=2 意味着节点 0 和节点 1 各重复一份全套。
**算一算**：rank=5 分解：pp_size×tp_size=4，5 ÷ 4 = 1 余 1 → dp_rank=1；余 1 ÷ 2 = 0 余 1 → pp_rank=0、tp_rank=1——rank5 在节点 1、GPU 1（与 README 映射表 rank 5 / dp 1 / pp 0 / tp 1 一致）。再看冗余：每条 Stage 边界消息按 tp_rank 配对飞 tp_size 次，字节冗余 = tp_size 倍（计入 p2p_bytes 不隐藏）。

**数字**（README ch42）：实验 D 在双节点 4+4 上跑 TP2/PP2/DP2；Stage 连续均分 28 层 / PP=2 → [14,14]。

**边界与易错**：本期的调度是第 39 期 GPipe，无 1F1B/Interleaved；0.6B 上组合可能是负优化。[易错] "任意两维布局的性能差能归因到某一维"——README 明确：同时改多个维度的横向差异不写成单变量因果结论，只有单变量差分（A1→A2、B1→B2、C1→C2）才有因果意义。

**产品对应**：Megatron-LM / DeepSpeed 的三维并行配置、NCCL 进程组管理 `[业界通识]`。

**方案视角**：现场先问"卡间互联带宽够不够 TP"并把 TP/PP 锁在节点内、DP 跨节点——这是机柜内 NVLink、机柜间以太网时代的标准布局；用 rank 公式核对客户集群的拓扑映射。

**讲给客户的一句话**：单维并行各有天花板——TP 藏进节点内、PP 接力层、DP 只在节点间同步梯度，三维各司其职才能同时吃满显存、算力和网络。

**掌握目标**：能由 rank 公式推出给定 (dp,pp,tp) 的组表；能判断"TP/PP 留节点内、DP 跨节点"布局下跨节点通信的口径。

**深挖入口**：帖子未发，读仓库 chapter_42 README

### ch43 · 分布式 Checkpoint 与恢复

**先回答一个问题：训练跑到第 N 步断电，前面烧的卡时怎么办？**

靠 Checkpoint（训练状态快照：参数、优化器状态、进度、随机数种子等一起存盘，崩溃后从快照续跑）。第 42 期之后训练状态只在各 Rank 内存里——崩溃或停机即从头再来。本期唯一变量：完整状态集中保存为单文件（rank 0 聚合写出，其他 Rank 在 barrier 等待），还是各 Rank 并行保存 Shard（分片 + manifest 清单）；`torch.distributed.checkpoint`（DCP）作为官方对照。

<!--fig:ch43-->

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

**怎么工作（拆到计算步骤级）**：快照 life-cycle 是"存 → 崩 → 载 → 续"，四步：
1. **分片写出 + manifest**：sharded 模式每个 Rank 并行写自足分片（`*.tmp` 写完后 `os.replace` 原子替换——被 Kill 的进程不可能留下会被信任的半写文件），rank 0 额外写 manifest（文件名、Rank、校验和、字节）；centralized 是 rank 0 聚合单文件、其他 Rank 在 barrier 等待。保存前先聚合各 Rank 模型校验和、必须逐位相等——AllReduce 契约一旦破坏，Checkpoint 会把分叉静默固化。
2. **恢复时按 Rank 重分片**：DDP 各 Rank 在 AllReduce 后持有逐位相同的副本，所以"分片"是可互换的完整副本——重分片退化为**副本选择 + 校验**：4 卡保存 → 2 卡加载，新 Rank r 读 `shard-{r % 4}`（rank 0 先核对 manifest 里各分片校验和一致）。跨 World Size 后 loss 只有全局值可比。
3. **状态回填**：两处著名陷阱——① `LRScheduler.load_state_dict` 不回写 optimizer 的 param_groups，新建调度器已把 LR 重置为 Warm-up 初值，必须用 `_last_lr` 回填（第 16 期的测试恰好被平坦 Warm-up 掩盖了这个错误）；② RNG 是每 Rank 状态：同 World Size 各取自己的槽位，重分片时新 Rank r 取旧 Rank `r % W_old`。
4. **恢复即验证**：连续性脚本对照五样——接续段逐步全局 loss、LR 序列、下一 batch 预览、RNG 指纹、最终参数校验和；同构恢复要求逐位相等，不是容差内一致。

**显存账**：0.6B FP32 每副本 ≈ 7.2GB（参数 2.4GB + AdamW 状态 4.8GB）都要进快照，再加 LR/进度/Sampler/每 Rank RNG/拓扑与软件版本；sharded 不省字节（总体积约等于 centralized），收益只在不并行就得不到的 I/O 时间。

**算一算**：4 卡并行保存 7.2GB，理想写盘时间 ≈ 单 Rank 串行写的 1/4；保存时间/训练时间占比 = 保存墙钟 ÷ 单步均值（README 同进程测）——这个比值乘上保存间隔，就是训练为容错交的停摆税。

**数字**（README ch43）：0.6B FP32 Checkpoint ≈ 7.2GB/副本；sharded 不省字节，收益只在 I/O 并行；4 卡保存 → 2 卡加载重分片可跑。

**边界与易错**：异步/影子 Checkpoint、远端对象存储、ZeRO/FSDP 的真重分片（字节级实切）都排除。[易错] "存个参数就能续训"——LR、Sampler、RNG、拓扑缺一个，续跑轨迹就分叉；平坦 Warm-up 恰好会掩盖 LR 回填错误。

**产品对应**：`torch.distributed.checkpoint`（DCP）、DeepSpeed 的 save_checkpoint、HF Trainer 的 resumable checkpoint `[业界通识]`。

**方案视角**：现场先问"恢复时间目标（RTO）是多少"——它决定保存频率（保存时间占训练时间的比例）和模式：吞吐敏感用 sharded 并行写，网络出口受限再谈集中式。

**讲给客户的一句话**：Checkpoint 保的不是权重一个文件，而是"断在哪一步、学到什么状态、随机数走到哪"的全套现场——目标只有一个：从最近快照逐位接上，一刻不从头再来。

**掌握目标**：能列出 Checkpoint 保存清单的七类内容；能答出"4 卡存的 Checkpoint 能否被 2 卡加载"及 DDP 下它为什么合法。

**延伸阅读**：[ZeRO (arXiv 1910.02054)](https://arxiv.org/abs/1910.02054)——快照里最大的那 12N 字节训练状态从哪来

**深挖入口**：帖子未发，读仓库 chapter_43 README
