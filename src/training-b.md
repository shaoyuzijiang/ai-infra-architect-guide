# 训练篇下半（ch30–ch43）：从通信隐藏到三维并行

承接 ch29 的朴素 DDP：从这一章起，每一章仍然只改一个变量，但改的对象从"数据怎么切"升级为"状态放哪里、通信何时发、流水怎么排"。14 章连起来是一条完整的推导链：通信重叠（ch30–31）→ ZeRO 三级状态分片（ch32–34）→ 张量并行三件套（ch35–37）→ 序列并行（ch38）→ 流水线三代调度（ch39–41）→ 三维并行组合（ch42）→ 可恢复训练（ch43）。读者是不跑实验的架构师：每一章都给"机制是什么、显存账怎么变、什么时候用、什么时候是负优化"。

术语约定（首现白话）：**AllReduce**=所有卡互相交换数据，最终每张卡都拿到全体之和；**Reduce-Scatter**=求和后只把属于自己的一段留下；**AllGather**=每卡交出自己的分片，换回所有人的分片拼成全量；**Shard**=一块连续的切片；**micro-batch**=一个优化步内部再切出的小批；**Stage**=流水线里一台卡负责的连续层段。

### ch30 · DDP 梯度分桶与通信计算重叠

**一句话定位**：把梯度装进桶里，让 AllReduce 藏进 Backward。

**解决什么问题**：上一章的 DDP 要等所有参数的梯度都算完，才发起一次全量 AllReduce——整个通信时间赤裸裸排在关键路径上，GPU 在这段时间里纯粹等网络。真实的训练步里，靠后的参数其实早就反向算完了，为什么不让它们先去通信？

**原理图解**：<!--fig:ch30-->

**原理要点**：
- baseline 是"Backward 结束后一次性 AllReduce"（通信全暴露）；variant 把参数按反向就绪顺序（参数逆序）装进若干桶，某桶梯度一齐就绪就立刻以异步方式发起 AllReduce，与后续 Backward 计算重叠，步末统一等待并挂回梯度。
- 通信数据总量不变（约 2.4 GB/步，FP32、2 卡、全局 batch 4），变的只是启动时机与分批粒度。
- 桶大小是个双向权衡：桶过小，每次 collective 的固定延迟占比上升反而变慢；桶过大退化为基线的全暴露形态。
- 收益上限受物理约束：无 NVLink 时走 PCIe，通信与计算共享带宽，重叠是"抢带宽"不是"变出带宽"。

**重点**：**Bucket 化不减少通信字节数，只改变通信与计算的时间重叠关系。**

**边界与常见误区**：
- 本期最重要的边界：通信与计算共享带宽，重叠收益可能有限，小模型 + PCIe 上甚至接近零。
- [易错] "分桶就是省流量"——错。分桶省的是暴露的等待时间；总字节一克都没少，桶太大还等于没分。

**产品对应**：PyTorch DDP 的 bucket 机制（`bucket_cap_mb`，默认 25MB）；DeepSpeed、Megatron-LM 的梯度引擎同构 `[业界通识]`。

**方案视角**：当客户的步时间被通信主导（梯度大、互联弱）时，bucket 重叠是零改动的第一档优化；判据是问"通信时间占步时间的比例"，以及节点内有没有 NVLink。若计算本身是大头，这一档收益不值得放进承诺。

**讲给客户的一句话**：把"算完再传"改成"算一部分传一部分"，通信尽量藏进计算里——在共享带宽的机器上，这是不花钱的一档优化，但收益取决于互联。

**掌握目标**：能画出按桶异步 AllReduce 与 Backward 重叠的时间线；能判断给定桶大小是偏"尾部等待"还是偏"退化全暴露"。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_30`）。

### ch31 · 梯度累积期间延迟同步（no_sync）

**一句话定位**：K 个微步只同步最后一次，通信量降为 1/K。

**解决什么问题**：当一个优化步由 K 个 micro-batch 累积而成时，朴素的 DDP 会在每个微步 backward 后都做一次全量 AllReduce——前 K−1 次同步的结果马上又被本地覆盖，纯属浪费。能不能前面攒着、最后一次性同步？

**原理图解**：<!--fig:ch31-->

**原理要点**：
- baseline 是"每微步都同步"；variant 是前 K−1 步 `no_sync`（纯本地累积），最后一步统一做一次 AllReduce SUM。梯度数学不变，每步通信量降为 1/K；K=2 时省一次全量 AllReduce。
- 正确性前提：每微步损失按监督 Token 占比缩放后，两种模式都收敛到同一个全局 token_mean 梯度。
- [关键机制] "每微步同步"的正确实现必须用独立累加器——对已归约的累积缓冲再次 AllReduce，会把先前微步的贡献放大 World Size 倍。
- 累积维度（本章）与单步内的重叠（ch30）是正交的两个变量，本期刻意不用 bucket 机制。

**重点**：**no_sync 省的是同步次数：K 个微步只付 1 次 AllReduce，通信量 1/K。**

**边界与常见误区**：
- 本期最重要的边界：主对照 K=2（全局 batch 4）；K 增大到 4 时每卡每微步样本数在变，容量口径要跟着改。
- [易错] 复用同一个梯度缓冲边同步边累积——会把之前微步的梯度放大 W 倍，训练悄悄发散。

**产品对应**：PyTorch DDP 的 `no_sync()` 上下文、HF accelerate 的 `gradient_accumulation_steps` 都是这个语义 `[业界通识]`。

**方案视角**：显存吃紧被迫用小 micro-batch 时，梯度累积是标准解；配 no_sync 后累积几乎不额外付通信。判据：看 K 与全局 batch 的关系——K 越大，这一档节省越显著；K=1 时无收益。

**讲给客户的一句话**：小批量攒梯度时，中间那些次同步本来就是白做的；攒到最后同步一次，梯度一样，通信开销按累积轮数成比例下降。

**掌握目标**：能算出给定 K 下每步 AllReduce 次数与通信量；能区分"独立累加器"与"缓冲复用"两种实现的梯度差异。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_31`）。

### ch32 · ZeRO-1：切分优化器状态

**一句话定位**：AdamW 动量按 Rank 切片，状态显存除以 W。

**解决什么问题**：DDP 里每张卡都存着完整模型、完整梯度、完整 AdamW 状态（一二阶动量）。0.6B 模型在 12GB 显卡上，光优化器状态就足以构成容量压力——而动量这个东西，每张卡存一份其实存了 W 份一模一样的。

**原理图解**：<!--fig:ch32-->

**原理要点**：
- baseline 是每卡全量 AdamW 状态；variant 把动量分片存放在各 Rank（1/W），参数与梯度仍每卡全量。AdamW 是逐元素更新，"本 Rank 状态分片 × 本 Rank 梯度分片"与全量 AdamW 逐位一致。
- 更新只发生在参数分片的所有者上；更新完成后用一次分块 AllGather（128MB 块缓冲）把新参数同步给其他 Rank。
- 显存账（README 口径）：优化器状态从全量 12 字节/参数降为 12N/W，W=2 时状态显存减半；参数、梯度账目不变。
- 通信账：每步新增约 1×参数字节的 AllGather——这是 DDP 没有的开销。
- 准入含义：加大 micro-batch 或序列长度时，ZeRO-1 应比复制式 DDP 更晚 OOM。

**重点**：**ZeRO-1 用"每步多一次参数 AllGather"换"优化器状态降到 1/W"。**

**边界与常见误区**：
- 本期最重要的边界：训练语义与复制式 AdamW 的等价性靠"公式逐操作镜像"保证，分片边界是非均匀的（余数分给前几个 Rank）。
- [易错] 以为切了状态就省了通信——方向反了：ZeRO-1 是花通信（+1×参数字节 AllGather）买显存。

**产品对应**：DeepSpeed ZeRO Stage 1；PyTorch 侧等价思路见 FSDP 的状态分片 `[业界通识]`。

**方案视角**：模型在单卡显存里"装得下但转不开"（状态挤占激活空间）时，ZeRO-1 是第一刀；判据是看容量探针里 OOM 首先出现在状态还是激活。若模型本身就放不下，需要 ch34 的参数分片。

**讲给客户的一句话**：每张卡其实存了 W 份相同的动量，把它们切开只留自己那份，更新后把新参数广播回去——显存立省，代价是每步多一次参数同步。

**掌握目标**：能算出 N 参数模型在 W 卡上 ZeRO-1 相对 DDP 的状态显存变化；能说出新增通信的字节量级（1×参数字节/步）。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_32`）。

### ch33 · ZeRO-2：再切分梯度

**一句话定位**：Reduce-Scatter 只留本 Rank 那份梯度。

**解决什么问题**：ZeRO-1 之后每卡还留着完整梯度——backward 结束那一刻它必须存在，可之后只有本 Rank 的那份会被本 Rank 的优化器用到。梯度也切了会怎样？

**原理图解**：<!--fig:ch33-->

**原理要点**：
- baseline 是完整 AllReduce（每卡拿到全量梯度）；variant 用 Reduce-Scatter：全局求和后只保留本 Rank 的分片，其余立刻释放。
- 通信账（README 口径）：Reduce-Scatter 与 AllReduce 的每链接数据量同为 2(W−1)/W·D，W=2 时通信时间应接近——省显存几乎不多付时间。
- 显存账（README FP32 账本）：每卡常驻模型状态从 ZeRO-1 形态降为 ZeRO-2 的 2N+14N/W——完整梯度消失，只剩 1/W 分片。
- 代价之一：`clip_grad_norm_` 不再直接可用，全局范数要由"各 Rank 分片局部范数平方和 → 一次标量 AllReduce"合成，再缩放本地分片。

**重点**：**Reduce-Scatter = AllReduce 的省显存版：求和完只留自己那段，其余当场扔。**

**边界与常见误区**：
- 本期最重要的边界：真实 ZeRO 用扁平桶做 Reduce-Scatter，本期逐参数发起，固定开销如实计入、不美化。
- [易错] 切了梯度之后照用 `clip_grad_norm_`——拿到的是错的范数；分片世界里连"裁剪"都需要一次跨 Rank 合成。

**产品对应**：DeepSpeed ZeRO Stage 2；FSDP 的梯度 Reduce-Scatter 同机制 `[业界通识]`。

**方案视角**：ZeRO-1 与 ZeRO-2 之间的选择几乎不需要权衡——W≥2 时梯度分片通信时间与完整 AllReduce 接近而显存直降。判据：确认框架版本的梯度裁剪路径是分片感知的。

**讲给客户的一句话**：全量梯度在每卡上只活到求和那一刻，之后 1−1/W 的部分注定没用；Reduce-Scatter 把它们当场丢掉，通信时间基本不变、梯度显存除以卡数。

**掌握目标**：能画出 AllReduce 与 Reduce-Scatter 的每链接数据量对比；能答出分片梯度下全局范数的三步合成方法。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_33`）。

### ch34 · ZeRO-3 / FSDP：再切分参数

**一句话定位**：参数也分片常驻，用时 AllGather、算完即释放。

**解决什么问题**：状态和梯度都切了，参数还整份占着显存——它只在 forward/backward 里被读，却在两个优化步之间整份闲置。能不能让它也不常驻？

**原理图解**：<!--fig:ch34-->

**原理要点**：
- baseline（ZeRO-2）参数全量常驻；variant 里参数只以 Shard 形式常驻，获得完整生命周期：分片常驻 → Forward 前 AllGather → 计算 → Reshard（释放全量存储）→ Backward 前再 AllGather → Reduce-Scatter 梯度 → 分片 Optimizer Step。
- 显存账（README FP32 账本）：每卡常驻从 ZeRO-2 的 2N+14N/W 降到 ZeRO-3 的 16N/W——参数、梯度、状态全部 1/W。
- 通信账：ZeRO-1/2 的 post-step AllGather 被取消，下一个 Forward 的 AllGather 本身就是参数刷新；但通信搬进了 Forward/Backward 热路径。
- 0.6B 上 AllGather 进热路径很可能让 Step Time 变长——负优化如实保留，收益首先出现在容量。
- FSDP 的 Reduce-Scatter 取平均，本地损失需乘 W 校正才能对齐全局 token_mean 梯度。

**重点**：**ZeRO-3 把常驻显存压到 16N/W，代价是 AllGather 进入前反向热路径。**

**边界与常见误区**：
- 本期最重要的边界：0.6B 这种单卡本可容纳的模型，ZeRO-3 往往是负优化；它的主场是大模型装不下的场景。
- [易错] Reshard 不是"逻辑上标记释放"——全量存储必须在单元返回时真正消失，否则账是假的。

**产品对应**：PyTorch FSDP（`FULL_SHARD`）、DeepSpeed ZeRO Stage 3、Megatron-LM 的分布式优化器 `[业界通识]`。

**方案视角**：模型权重本身超过单卡显存时，ZeRO-3/FSDP 是必选项；判据直接看容量：单卡放不下参数就上 ZeRO-3，放得下就优先 ZeRO-1/2 保吞吐。还要问激活重算与 offload 是否需要一起开。

**讲给客户的一句话**：参数平时也只留 1/W，算哪层临时凑齐哪层、用完立刻散伙——显存账降到 16N/W，代价是通信从步末挪进了计算热路径。

**掌握目标**：能画出参数"分片→AllGather→Reshard→再 AllGather"的完整生命周期；能算出 W 卡上 ZeRO-2 与 ZeRO-3 的每卡常驻账（2N+14N/W 对 16N/W）。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_34`）。

### ch35 · MLP Tensor Parallel

**一句话定位**：SwiGLU MLP 沿中间维切，卡间两次求和。

**解决什么问题**：ZeRO 系列从数据并行维度切了模型状态，但单层内部的矩阵计算与权重仍每卡完整。单层能不能也切开，让"放不下"的大层摊到多卡上？

**原理图解**：<!--fig:ch35-->

**原理要点**：
- baseline 是 MLP 权重完整复制（tp=1）；variant 用 Column Parallel（Gate/Up 沿输出维切）+ Row Parallel（Down 沿输入维切），Attention、Embedding、LayerNorm 仍每卡复制，单独测清 MLP TP 这一个机制。
- 每层新增 2 次 TP AllReduce：forward 的 down_proj 输出求和、backward 的 MLP 输入梯度求和（Megatron 的 f 算子），全部暴露在关键路径上。
- 梯度所有权三分：分片权重梯度纯本地（不通信）；MLP 输入残差流梯度必须做 backward AllReduce，漏了会有"部分梯度"静默污染下游；复制参数梯度各 Rank 天然相同，再做 AllReduce SUM 会放大 tp_size 倍——是错误不是浪费。
- 数学上与完整 SwiGLU 等价（FP32 下仅归约顺序分叉）；0.6B 单卡放得下，TP 更可能是负优化。

**重点**：**TP 切的是单层权重：省参数省激活，代价是每层固定两次 AllReduce。**

**边界与常见误区**：
- 本期最重要的边界：0.6B 上 TP 是预期内的合法负优化——机制成立与否与"当下值不值"是两回事。
- [易错] 把复制参数的梯度也 AllReduce——直接放大 tp_size 倍；TP 组内数据相同，只有 DP 组之间才需要梯度同步。

**产品对应**：Megatron-LM 的 `ColumnParallelLinear`/`RowParallelLinear`；vLLM、SGLang 的 TP 推理同源 `[业界通识]`。

**方案视角**：TP 只该用在"单层放不下"的模型上，且 TP 组必须放在 NVLink/同节点内——每层的 AllReduce 对延迟极其敏感。判据：单卡（或单 NVLink 域）能否容纳最大层；跨节点做 TP 通常是方案红旗。

**讲给客户的一句话**：把一层的大矩阵掰成几份各算各的，每层付两次卡间求和作门票——它解决"单层放不下"，不解决"跑得更快"。

**掌握目标**：能画出 Column+Row Parallel 的切分位置与两次 AllReduce 时机；能区分"必须通信/禁止通信"的梯度类别。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_35`）。

### ch36 · Attention Tensor Parallel

**一句话定位**：Attention Head 按 TP Rank 切，GQA 整除约束。

**解决什么问题**：MLP 切完之后，每层的 Attention 投影与 Head 还是完整的。Attention 天然按 Head 并行——能不能让每张卡只拿一部分 Head？

**原理图解**：<!--fig:ch36-->

**原理要点**：
- baseline 是 MLP TP + Attention Head 完整复制；variant 在其上把 Q/KV Head 按 TP Rank 连续切分。Qwen3-0.6B 是 GQA：32 个 Q Head、8 个 KV Head（4:1），TP=2 每 Rank 16Q/4KV、TP=4 每 Rank 8Q/2KV，每 Rank 局部 GQA 组数不变（4）。
- 硬约束：每 Rank 必须拿到整数个完整 Head；`head_dim` 永不切分；per-head RMSNorm 作用在 head_dim 内，随 Head 留在本 Rank。RoPE 是 Head 内固定旋转，局部重放与全局等价。
- 通信账：attn_tp 臂每层 2 次前向（o_proj 输出、down_proj 输出）+ 4 次反向（两个块的输入梯度 + q/k norm 梯度同步）；对照 mlp_tp 臂每层各 1 次。
- 本期特有陷阱：q/k norm 权重复制但梯度是部分和，必须 hook 同步，否则副本从第一次更新起静默分叉。

**重点**：**切 Attention 就是切 Head：整除是硬前提，head_dim 永不切。**

**边界与常见误区**：
- 本期最重要的边界：0.6B 的 Attention 占比小，预期收益有限甚至为负。
- [易错] TP 度与 Head 数不整除还硬切——GQA 的 32Q/8KV 对 TP=2/4 都整除，但对任意 tp_size 不保证；不整除必须报错而非就近凑合。

**产品对应**：Megatron-LM 的 ParallelAttention；vLLM/SGLang 按 Head 切分 TP 组 `[业界通识]`。

**方案视角**：选择 TP 度时把"Q/KV Head 数能否整除"列为硬检查项；GQA 模型的 KV Head 数往往是更紧的上界（8 KV Head 天然封顶 TP=8 的头数约束 `[业界通识]`）。

**讲给客户的一句话**：Attention 天然按头并行，每卡领走整数个完整 Head 即可；但 KV Head 总数决定了能切多细，选 TP 度前先做整除检查。

**掌握目标**：能算出 GQA 32Q/8KV 在 TP=2/4 下每 Rank 的 Head 布局；能答出 q/k norm 梯度为什么需要额外同步。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_36`）。

### ch37 · 词表并行 Embedding 与分布式 Cross Entropy

**一句话定位**：词表按 Rank 切行，Logits 不再整块物化。

**解决什么问题**：权重都切完了，但最后一层的 Logits 仍是全词表的大矩阵：batch 2 × seq 512 时，151936 词表的完整 Logits 约 622 MiB——TP 模型里最大的单项激活。训长序列、大 batch 时它先爆。

**原理图解**：<!--fig:ch37-->

**原理要点**：
- baseline 是完整词表 Logits + 完整 CE；variant 把 Embedding/LM Head 沿词表切（TP=2 每 Rank 75968 行），CE 用分布式算法：全局 max 用 AllReduce MAX、全局 exp 和用 AllReduce SUM、目标 token logit 按所属 Rank 取回。
- 分布式 CE 的通信账：forward 新增 3 次标量级 Collective（尺寸是 token 数而非词表数），backward 0 次通信——log-sum-exp 可分块归约，softmax−onehot 只需本 Rank 分片和三个全局标量。
- 显存账：物化的 Logits 从完整词表降为 1/tp_size（622 MiB 量级的账按 TP 度均摊，README 口径）。
- backward 侧固定代价：LM Head 的输入梯度 `dlogits_r @ W_r` 是部分和，需一次 backward AllReduce（f 算子）；漏掉会让 LayerNorm 梯度跨 Rank 不一致。
- Qwen3 的 Embedding 与 LM Head 共享权重，词表分片后两者共享同一个分片 Parameter，两条路径梯度自动累加。

**重点**：**词表并行的本质是让 622 MiB 量级的 Logits 不物化：CE 拆成三次标量级归约。**

**边界与常见误区**：
- 本期最重要的边界：tp=2 下词表切分在 0.6B 上收益可能仍为负——它是长序列/大词表场景的容量技术。
- [易错] 以为分布式 CE 的 backward 也要通信——恰好相反：backward 0 次通信，代价在 forward 的三次 Collective 和 LM Head 输入梯度那次 AllReduce。

**产品对应**：Megatron-LM 的 `VocabParallelEmbedding`/`VocabParallelLMHead` 与 parallel cross entropy `[业界通识]`。

**方案视角**：客户抱怨"加大 batch/长度就在 CE 层 OOM"时，先算 Logits 账：vocab×tokens×4 字节。它超过激活大头时词表并行+分布式 CE 才是对症的刀；否则先调 micro-batch。

**讲给客户的一句话**：输出层的 Logits 矩阵随词表和序列长度平方级膨胀；把词表切到各卡、交叉熵只传三个标量，这层显存就不再是最先爆的那块。

**掌握目标**：能算出给定 batch/seq/vocab 的完整 Logits 字节数；能画出分布式 CE 的三次 forward Collective 与"backward 0 通信"结构。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_37`）。

### ch38 · Sequence Parallel

**一句话定位**：TP 区域之外的激活沿 Sequence 维切开。

**解决什么问题**：MLP、Attention、词表都切了，但残差流和 LayerNorm 的激活仍是完整的 `[batch, seq, hidden]`，每卡重复一份——它们和权重是否分片无关。长序列时这块越来越贵。

**原理图解**：<!--fig:ch38-->

**原理要点**：
- baseline（纯 TP）TP 区域间激活全量复制；variant 把残差/LN 激活沿 Sequence 维切成 `[b, s/N, h]`：进 TP 区域前 AllGather（g 算子），离开时 Reduce-Scatter（ḡ 算子）。
- 数学前提：RMSNorm 逐 Token 独立、残差逐元素，所以在序列分片上计算等于全序列算完再切片；Attention 是唯一的序列耦合算子，仍在 TP 区域内部拿全序列计算。
- 显存账：saved-for-backward 激活字节按 TP 度下降（每 Rank 只存自己 Token 窗口）；代价是每个并行块多一次 AllGather + 一次 Reduce-Scatter 的边界通信。
- 易错点：序列切分后 LayerNorm 等复制参数的梯度从"天然一致"变成部分和，必须 hook 同步，否则副本静默分叉——与第 35/36 期的"禁止 AllReduce"正好相反。
- 收益预期首先出现在长序列容量；0.6B、L=512 上端到端吞吐可能是负优化，如实保留。

**重点**：**SP 切的是权重切不到的激活：序列维进 TP 区、出 TP 区，一聚一散。**

**边界与常见误区**：
- 本期最重要的边界：Context Parallel / Ring Attention 不在本期——Attention 内部仍跑全序列，切分只在 TP 区域之外。
- [易错] 把 g 算子的 backward 也做成 Reduce-Scatter——f 算子已把梯度求和成一致副本，切片即可；再 Reduce-Scatter 会放大 tp_size 倍。

**产品对应**：Megatron-LM Sequence Parallel；Hugging Face 凡带 `sequence_parallel` 开关的 TP 实现同构 `[业界通识]`。

**方案视角**：长上下文训练（序列 8K+）激活显存成为大头时，SP 是与 TP 搭配的标准件 `[业界通识]`；短序列上它只添通信。判据：激活账占每卡显存的比例。

**讲给客户的一句话**：权重切完之后，激活里还有一大块每卡重复着——沿序列把它们也切开，长上下文训练的显存大头才真正松下来，代价是每层多一进一出两次通信。

**掌握目标**：能画出 g（AllGather）与 ḡ（Reduce-Scatter）在层内的位置；能解释为什么 SP 下 LayerNorm 梯度反而需要同步而纯 TP 下不需要。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_38`）。

### ch39 · GPipe Fill-Drain 流水

**一句话定位**：模型按层跨卡切，多个微批先填满再排空。

**解决什么问题**：把 28 层均分到 4 张卡（PP=2 每 Stage 14 层）后，如果一个 micro-batch 从头走到尾才放下一个，任一时刻只有一张卡在算、其余全在等——并行度是假的。

**原理图解**：<!--fig:ch39-->

**原理要点**：
- baseline 是"一个 micro-batch 走完全流水再下一个"（serial）；variant 是 GPipe Fill-Drain：先连续做完所有 Forward（Fill），再连续做所有 Backward（Drain）。Stage 划分、权重、micro 序列、loss 口径完全一致。
- Bubble 收益（README 口径）：填充-排空模型给出 Bubble 上界 (S−1)/(M+S−1)，micro 数 M 越大 Bubble 越小。
- 显存账：serial 每 Stage 同一时刻只驻留 1 份激活图；gpipe 驻留 M 份——峰值激活显存随 M 线性上升，这是调度的机制性代价。
- Backward 靠 P2P 梯度种子驱动：每个 Stage 收到输出激活的输入梯度后启动本段反传，再把输入梯度传回上一 Stage；PP 把 tied 的 Embedding/LM Head 拆到首末 Stage，每步要做一对 P2P 梯度交换。
- 小模型走 PCIe P2P 时，调度收益可能被传输吃掉甚至为负。

**重点**：**流水的本质是让 M 个微批在 Stage 间错峰：Bubble 压到 (S−1)/(M+S−1)，激活驻留涨 M 倍。**

**边界与常见误区**：
- 本期最重要的边界：PP=2 每 Stage 14 层是连续均分，不做自动负载均衡；层不均的 Stage 会拖累整条流水。
- [易错] 以为切了层就自动并行——serial 调度下 GPU 利用率不变；真正产生重叠的是调度，不是切分。

**产品对应**：PyTorch 的 `torch.distributed.pipelining`、DeepSpeed Pipeline Engine、Megatron-LM 的 GPipe 式调度 `[业界通识]`。

**方案视角**：模型按层都放不下单卡、或单卡算一层利用率不足时才用 PP；PP 的 Bubble 上界决定它偏好配大 M。判据：Stage 划分是否均衡、M 能开多大（受激活显存反压）。

**讲给客户的一句话**：按层切四段后让四个微批流水错峰，卡与卡互相喂活；微批越多流水越满，但每个 Stage 要同时攥着所有微批的激活，显存按批数涨。

**掌握目标**：能画出 S=4、M=8 的 Fill-Drain 时空图并标出 Bubble；能算出 serial 与 gpipe 的激活驻留份数差。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_39`）。

### ch40 · 1F1B 流水调度

**一句话定位**：热身几张后一前一后交替，激活驻留砍到 min(S−s, M)。

**解决什么问题**：Fill-Drain 的代价是每个 Stage 从 Fill 开始就攥着全部 M 份激活图直到 Drain 结束。批越大显存越爆——能不能让 Backward 提前排、激活用完就扔？

**原理图解**：<!--fig:ch40-->

**原理要点**：
- baseline 是 Fill-Drain（先所有 F 再所有 B）；variant 是非交错 1F1B：Stage s 先做 min(S−1−s, M) 个 Warm-up Forward，Steady State 交替一 Forward 一 Backward，最后 Cool-down 排空。
- 激活生命周期（README 口径）：同时驻留的激活图从 M 份降到 min(S−s, M) 份——Warm-up 结束驻留 warmup 份，Steady 每轮先拿一份再放一份。1F1B 的主要价值假设是峰值激活显存下降，而非吞吐大涨。
- Bubble 上界仍服从 (S−1)/(M+S−1)：micro 增大时两臂差距收窄。
- 执行序完全由 (stage_index, S, M) 决定，无需额外同步；Steady 一轮固定"先收后发"以打断阻塞传输下的死锁环。
- 梯度按 micro 序累加，与 Fill-Drain 逐位一致；tied 交换与更新边界不变。

**重点**：**1F1B 用提前的 Backward 换激活早释放：驻留份数从 M 降到 min(S−s, M)。**

**边界与常见误区**：
- 本期最重要的边界：小模型 + PCIe P2P 上，调度收益可能被更多次收发等待吃掉。
- [易错] 以为 1F1B 主要提速——它主要省显存；Bubble 上界和 Fill-Drain 相同，micro 足够大时两者时间线趋同。

**产品对应**：Megatron-LM 的 non-interleaved 1F1B schedule；DeepSpeed 的流水引擎默认即此形态 `[业界通识]`。

**方案视角**：PP 配置评审时问两件事：调度是 Fill-Drain 还是 1F1B、micro 数与激活驻留的乘积是多少。长序列大 batch 场景，1F1B 是让 PP 可用的前提而非可选项。

**讲给客户的一句话**：让每个 Stage 攒几张"热身批"后就开始边算前向边还后向账，激活用完立刻释放——显存峰值从"全部批"降到"一截流水"，批数再大也压得住。

**掌握目标**：能写出 Stage s 的 Warm-up 数公式并画出 S=4、M=8 的 1F1B 时空图；能判断给定显存预算下最大可用的 M 受哪一项约束。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_40`）。

### ch41 · Interleaved 1F1B（虚拟 Stage）

**一句话定位**：每卡存多个非连续虚拟 Stage，流水级数翻倍压 Bubble。

**解决什么问题**：1F1B 仍把流水级数 S 锁死在物理卡数 P 上。Bubble 上界 (S−1)/(M+S−1) 想再降就得加大 S——卡不够，能不能让每张卡假装成多级？

**原理图解**：<!--fig:ch41-->

**原理要点**：
- baseline 是每卡一个连续 Stage（1F1B）；variant 是每卡 2 个非连续虚拟 Stage（total = P × vpp）：虚拟 Stage s 位于物理 Rank `s % P`（strided placement），28 层 / P=4 / vpp=2 时 8 个虚拟 Stage 依次 [4,4,4,4,3,3,3,3] 层，每 Rank 合计仍 7 层。
- 调度是 Interleaved 1F1B：micro 按 P 分组，组内按块扫 Forward、按块逆序扫 Backward；micro 必须能被 P 整除，否则显式报错。
- Bubble 与通信的交换（README 口径）：Bubble 因流水级更密而下降，但同一 `[b, s, h]` 载荷要飞 2P−1 条 Stage 边而非 P−1 条，P2P 次数约翻倍；Warm-up 数增大，in-flight 激活上升。
- 依赖在块边界闭合成环，阻塞式 send 在严格阻塞传输上会死锁——统一用异步 `isend` 投递、所有阻塞等待都是 recv，链条必终于无 recv 的端点，可证明无环。
- 梯度语义与更新边界不变；tied 交换发生在物理首末 Rank 之间。

**重点**：**虚拟 Stage 是用通信次数买 Bubble：流水级数 P×vpp，P2P 次数约翻倍。**

**边界与常见误区**：
- 本期最重要的边界：0.6B 小模型 + PCIe P2P 上端到端吞吐可能为负优化；interleaved 的主场是大模型 + 高带宽互联。
- [易错] 延迟发送激活"等反向完再发"——会静默打乱接收端的 FIFO 匹配，把梯度当激活；消息必须按每对每方向固定顺序立即投递。

**产品对应**：Megatron-LM 的 interleaved schedule（`--num-layers-per-virtual-pipeline-stage`） `[业界通识]`。

**方案视角**：vpp>1 是大模型 + 节点内高带宽下的精调手段，属于"压 Bubble 的最后一档"；小模型或弱互联集群上要求开 vpp 的方案应当追问动机。判据：Bubble 占比与 P2P 次数曲线的交点。

**讲给客户的一句话**：让每张卡兼两段不连续的层，流水从 4 级变 8 级、空窗更短；代价是块与块之间飞行的消息翻倍——高带宽互联上划算，PCIe 上未必。

**掌握目标**：能画出 P=4、vpp=2 的 strided placement 表；能算出 interleaved 相对 1F1B 的 P2P 次数比与 Bubble 变化方向。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_41`）。

### ch42 · 混合并行与双节点 4+4

**一句话定位**：DP×TP×PP 三维组合，TP/PP 留节点内、DP 跨节点。

**解决什么问题**：DP、TP、PP 各自验证过了，真实集群要三者同时开：8 张卡分两个节点，哪维放节点内、哪维跨节点、rank 怎么编号——这是每个训练集群方案的顶层拓扑题。

**原理图解**：<!--fig:ch42-->

**原理要点**：
- baseline 是单维并行（TP2/PP1/DP1 等已验证配置）；variant 是三维组合。本期没有单一两臂对照，而是三组单变量差分（只加 PP、只加 DP、只加 DP）加一个综合配置 D（TP2/PP2/DP2，8 rank，双节点 4+4）。
- rank 分解公式是唯一的 placement 真相来源：`rank = dp_rank × (pp_size × tp_size) + pp_rank × tp_size + tp_rank`。实验 D 的映射表：TP 组 {0,1}{2,3}{4,5}{6,7} 节点内，PP 组 {0,2}{1,3}… 节点内，DP 组 {0,4}{1,5}… 跨节点——每节点一份完整 TP2×PP2 副本，训练主路径只有 DP 同步跨节点。
- 梯度所有权三类不变，DP 只做一次：TP 分片梯度 rank-local、残差流梯度 f 算子已按 TP 组求和、replicated 参数只在 DP 维做一次 AllReduce。
- 精确全局范数要防重复计数：先在 TP 组归约分片部分，再全 world 归约后除以 tp_size。
- 本布局的真实代价：PP 边界消息按 tp_rank 配对飞行 tp_size 次（字节冗余 = tp_size 倍），如实计入。

**重点**：**三维并行不发明新原语：拓扑表定组，通信各回各组，TP/PP 不跨节点。**

**边界与常见误区**：
- 本期最重要的边界：排除 1F1B/Interleaved 调度、ZeRO 组合、SP/Vocab Parallel——任何两维同时改变的横向对比不构成单变量因果结论。
- [易错] 只建"自己所在的组"——`new_group` 是全局集合操作，每个 rank 必须按相同顺序创建所有组，否则其他 rank 永久等待。

**产品对应**：Megatron-LM / DeepSpeed 的 3D parallelism 配置（`tensor_model_parallel_size` 等）；NCCL 的组划分 `[业界通识]`。

**方案视角**：给客户排拓扑时的标准顺序：TP 放 NVLink 域内、PP 放节点内、DP 承担跨节点——DP 通信是每步一次的梯度归约，最耐弱互联。判据：问节点内卡数与互联带宽，TP 度一旦要跨节点就是方案红旗。

**讲给客户的一句话**：8 卡两节点，节点内四张卡组成 TP2×PP2 的完整模型副本，两个副本之间只做一次梯度同步跨节点——把怕慢的通信关在节点内，把扩展性交给数据并行。

**掌握目标**：能写出 rank 分解公式并对实验 D 算出任意 rank 的 (dp, pp, tp) 三元组；能判断给定集群拓扑下三维各自该放哪一层。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_42`）。

### ch43 · 分布式 Checkpoint 与恢复

**一句话定位**：各 Rank 并行保存分片 + manifest，替代 rank 0 单文件聚合。

**解决什么问题**：三维并行跑起来后，训练状态只存在于各 Rank 内存：崩溃或停机即从头再来。保存时若让 rank 0 收齐所有 Rank 的状态再写一个文件，rank 0 的网络和磁盘就是单点瓶颈——卡越多越不可用。

**原理图解**：<!--fig:ch43-->

**原理要点**：
- baseline 是 centralized（rank 0 聚合写单文件，其他 Rank 在 barrier 等待）；variant 是 sharded（每 Rank 并行写自足分片 + rank 0 写 manifest 清单：文件名、Rank、校验和、字节）。DDP 分片是可互换的完整副本，两种模式恢复结果必须逐位一致，差别只在 I/O。
- 显存/字节账（README 口径）：0.6B FP32 的 Checkpoint 约为参数 2.4GB + AdamW 状态 4.8GB ≈ 7.2GB/副本；sharded 不省总字节，收益在不并行就得不到的 I/O 时间。
- 保存清单（两模式一致）：参数、优化器一二阶矩、LR 调度器、进度/Sampler 游标、每 Rank 各一份 RNG、拓扑元数据、软件版本；全部 `*.tmp` + 原子替换。
- 恢复陷阱：`LRScheduler.load_state_dict` 不回写 optimizer 的 param_groups，恢复后必须用 `_last_lr` 回填，否则续训第一步就用错学习率。
- 重分片（4 卡保存 → 2 卡加载）：DDP 下退化为"副本选择 + 校验"，新 Rank r 读 `shard-{r % 4}`；字节级真重分片属于 ZeRO/FSDP 的状态形态。

**重点**：**sharded checkpoint 的语义 = 并行 I/O + manifest：总字节不省，省的是训练停摆时间。**

**边界与常见误区**：
- 本期最重要的边界：异步/影子 Checkpoint、增量保存、远端对象存储都不在本期；`torch.distributed.checkpoint` 是官方对照路径，不是被替代的理解。
- [易错] 认为恢复就是 `load_state_dict`——LR 回填、每 Rank RNG 槽位、校验和核对缺任何一项，续训都会静默偏离。

**产品对应**：`torch.distributed.checkpoint`（DCP）；Hugging Face Trainer / DeepSpeed 的分片 checkpoint 格式 `[业界通识]`。

**方案视角**：长训练预算评审必问三件事：Checkpoint 频率、保存占训练时间比、恢复演练有没有做过。分片保存是大规模集群的默认选项；判据是"保存墙钟占步时间比例"随卡数的走势，以及共享文件系统是否可用。

**讲给客户的一句话**：让每张卡并行写自己那份状态、用一个清单登记校验和，保存时间不再被 rank 0 单点卡死；配合故障注入演练，中断后能从最近检查点逐位接上。

**掌握目标**：能列出 sharded 模式 manifest 的必备字段；能答出恢复后 LR 回填与每 Rank RNG 各解决什么静默错误。

**深挖入口**：帖子未发，读仓库 README（`miniinfra/chapter_43`）。
