# 推理篇（chapter_01–chapter_15）

### ch01 · 手写 Qwen3 推理

**解决什么问题**：把"大模型推理到底在算什么"拆到最底层——不用任何推理框架，只用 Python 和 PyTorch 完成 Qwen3-0.6B 的 Tokenizer 与单请求自回归生成，为后续所有优化建立 baseline。

**机制通俗版**：baseline 是"裸生成"：读取模型配置和 Safetensors 权重，实现 Byte-level BPE 和 non-thinking Chat Template，逐层手写 Qwen3 前向，greedy decoding 逐 Token 生成。没有 variant——本期唯一变量是"能否自己跑通"，对照物是 Transformers：相同输入、greedy 条件下逐 Token 比较，三组 Prompt 的 Logits 最大误差均为 0。成本侧也裸露了问题：每次 Decode 都重新计算完整上下文（无 KV Cache），19 个输入 Token + 32 输出在 3080 Ti 上 Prefill 34.43 ms、平均 Decode 33.40 ms/token，峰值显存 1173.80 MiB。

**架构上的位置**：整个系列的起点。它确立了两条主线：一是 Prefill（一次处理整段 Prompt）与 Decode（逐 Token 生成）的延迟结构差异，二是"每步重算全部历史"这一明显浪费，正是 ch02 KV Cache 要解决的。边界：只支持固定 revision 的 0.6B dense 模型、batch size 1、greedy、关闭 thinking；聊天模板只覆盖 system/user 纯文本，不支持工具调用和多模态；单次测量不能外推。

**业界对应**：对应任何推理引擎的最小可运行内核；Transformers 在这里只作正确性对照，不进入推理路径。

**讲给客户的一句话**：在谈任何加速比之前，先建立一份可复现的 baseline——否则"优化"无从谈起。

**深挖入口**：帖子「手写Qwen3推理」（8/24）· 教材 p1

### ch02 · KV Cache

**解决什么问题**：Decode 每生成一个 Token，都要对全部历史 Token 重跑模型层——历史越长浪费越大。KV Cache（把每层历史 K/V 存下来复用）是第一个推理优化。

**机制通俗版**：baseline 每次 Decode 用完整上下文重算；variant 保留每层历史 K/V，Decode 只算新 Token 的 Q/K/V 再追加进 Cache。Cache 是普通 Tensor，用 `torch.cat` 增长；Key 在 QK Norm 和 RoPE 之后写入，位置编号从 Cache 长度续算。实测揭示了关键边界：16 Token Prompt 下 KV Cache 反而**负优化**（端到端 1191 vs 1012 ms）——小模型短上下文中，Python 调度、Kernel Launch 和 `torch.cat` 重新分配的开销足以覆盖省下的计算；到 1024 Token 才有约 `1.85×` 加速（Decode 均值 60.43→32.64 ms）。Cache 数据量按公式 `2 × layers × num_kv_heads × head_dim × seq_len × batch × dtype_bytes` 线性增长，1024 Token Prompt 实测 115.39 MiB。

**架构上的位置**：依赖 ch01 的手写模型。两条关键边界：KV Cache 不减少 Prefill 计算，收益只在 Decode；Query 仍要读全部历史 K/V，所以 Decode 不是与上下文长度无关的常数复杂度。`1.85×` 只属于本软硬件组合，不能外推。

**业界对应**：vLLM 等引擎的 Cache 预分配与分页管理（ch05 起）远优于动态 `torch.cat` `[业界通识]`。

**讲给客户的一句话**：KV Cache 的收益随上下文变长才显现，短 Prompt 小模型上管理开销可能吃掉全部节省。

**深挖入口**：帖子「超轻量AI Infra入门教程02 kv cache」（8/25）· 教材 p45

### ch03 · Static Batching

**解决什么问题**：单请求利用率太低——把多个请求合并成一个成员固定的 Batch，共享一次模型前向，提高 GPU 并行度。

**机制通俗版**：baseline 是多个带 KV Cache 的请求逐个串行执行；variant 把它们组成固定 Batch：不同长度 Prompt 左 Padding + Attention Mask，一次 Batched Prefill，之后按 Batch 槽位组织 KV Cache 同步 Decode。关键约束：Batch 开始后不进新请求，完成请求的槽位空转到最长请求结束。代价随即暴露：混合长度组的有效槽位利用率仅 `46.88%`，有效吞吐从 115.30 掉到 57.63 token/s——这是 Padding 浪费，不是"短 Prompt 拖慢 Batch"。收益侧：Batch Size 16 相对串行 makespan 约 `13.98×`；但 Batch Size 192 吞吐 2981.83 token/s 高于 196 的 2931.97——接近显存边界后延迟上升，最大 Batch ≠ 吞吐最优 Batch。float32 下无请求串扰；bfloat16 下因矩阵 Shape 改变可能分叉，不属于准确性问题。

**架构上的位置**：依赖 ch02 的 KV Cache（扩展为 Batched KV Cache）。它是 ch04 Continuous Batching 的 baseline：固定批次的"完成槽位无法服务等待请求"正是后者的优化点。离线结果不含排队和网络，不代表在线 SLO。

**业界对应**：离线批量任务仍用固定 Batch 思路；在线服务已被 Continuous Batching 取代 `[业界通识]`。

**讲给客户的一句话**：拼批能提吞吐，但请求长短不齐时 padding 空转可能浪费一半算力。

**深挖入口**：帖子「Static Batching」（8/26）· 教材 p72

### ch04 · Continuous Batching

**解决什么问题**：固定 Batch 里已完成请求的槽位空转到整批结束。Continuous Batching 允许这些名额在下一次迭代被等待请求复用。

**机制通俗版**：baseline 是成员固定的批次；variant 在每次模型迭代前执行"回收完成槽位 → FCFS 填补 → 有新请求做 Batched Prefill，否则 Decode"。唯一变量是"完成名额能否复用"。异长输出场景 Makespan 加速 `1.169×`、吞吐 +16.8%、槽位有效率 62.50%→74.07%；等长输出负对照两条路径几乎无差（约 0.4%）——没有可提前复用的槽位就没有收益，机制成立的前提是负载不整齐。固定到达轨迹下加速 `1.486×`、TTFT p95 从 1511 降到 205 ms，但 ITL p95 从 36.50 升至 71.48 ms：新请求 Prefill 会阻塞已有请求的 Decode，这是留给 ch07 的伏笔。KV Cache 本期仍是 dense 槽位，扩容需搬移。

**架构上的位置**：依赖 ch03 的 Batched 执行与 KV Cache。关键边界：本期 Prefill 优先、FCFS 是为了可复现而非最优；dense Cache 的扩容搬移引出 ch05 Paged KV Cache。

**业界对应**：vLLM/SGLang 等在线引擎的标准调度方式 `[业界通识]`，源自 Orca 论文 `[业界通识]`。

**讲给客户的一句话**：Continuous Batching 让新请求"见缝插针"，但 Prefill 与 Decode 会互相干扰——延迟分位数要一起看。

**深挖入口**：帖子「连续批处理」（8/27）· 教材 p102

### ch05 · Paged KV Cache

**解决什么问题**：dense KV Cache 按全局最长请求预留物理宽度：一个 4096 Token 请求会让所有槽位在 4096 级别上跑 masked Attention，长度不均时显存浪费严重。

**机制通俗版**：baseline 沿用 dense 槽位，长请求加入时所有层所有槽位一起左扩容搬移；variant 借鉴操作系统虚拟内存分页（把显存切成固定大小的物理 Block，用 Block Table 把逻辑 Token 位置映射到物理块）：请求只为实际长度申请 Block，尾块内的浪费被限制在最后一个 Block，完成后物理块直接归还复用，不动其他请求的 K/V。实测：长度越不均，KV 读取减少越多（3072 Token 长请求下 KV 读取减少 94.1%、Cache Pool 减少 82.6%），但同并发吞吐反而降 36–45%——PyTorch reference 的 Block gather、分组 Attention 和额外 Kernel Launch 吃掉了计算节省。真正的收益在容量：12GB 卡上 dense 最大成功并发 8（12 起 OOM），paged 跑到并发 16，吞吐 157.80 token/s 略高于 dense 最优、峰值显存低约 43%。

**架构上的位置**：依赖 ch04 的 Continuous Batching。两条边界：减少 K/V 访问 ≠ 端到端更快，Kernel 效率同样关键；Block Pool 容量按已知上界预分配，无动态准入。它是 ch08 前缀共享和 ch12 换出的物理基础。

**业界对应**：vLLM 的 PagedAttention `[业界通识]`，专用 CUDA/Triton Kernel 直接读 Block Table，无本期 reference 路径的开销。

**讲给客户的一句话**：分页的价值是把显存从"按最长请求预留"变成"按需分配"，同一张卡能多接一倍请求。

**深挖入口**：帖子「从零实现Paged KV Cache」（8/28）· 教材 p144

### ch06 · 迭代级推理请求调度器

**解决什么问题**：ch05 的调度没有算力预算：Prefill burst 一次涌入时，调度器可能连续多轮做 Prefill，已有 Decode 请求被长时间饿停。

**机制通俗版**：baseline 是纯 FCFS——有空闲槽位就一次 Prefill 所有可加入请求；variant 是软 Token Budget：按 padded cost（`batch_size × max(prompt_length)`）扣预算，预算不足即停，且已有 Decode 时两个 Prefill iteration 之间至少插一次 Decode iteration。"软"的含义：单条 Prompt 超预算时不拆分，允许队头超预算执行避免永久阻塞。突发 12 条长短混合 Prompt 的实测：budget 2048 相对 baseline 吞吐 +13.1%、既有请求最大 ITL 降 58.5%、Padding 减少 53.0%，但 ITL p95 从 98.52 升到 207.37 ms——把一次长停顿拆成多次中等停顿，不是延迟全面改善。budget 256 过紧，反而因超预算请求吞吐 -6.8%。

**架构上的位置**：依赖 ch05 的 Paged KV Cache 与 Continuous Batching，是 ch07 的前半步。边界：不做显存准入、抢占、优先级和 P/D 分离；预算按 padded 而非有效 Token 计，Padding 本身也耗预算。

**业界对应**：vLLM/SGLang 迭代级调度与 Decode 保护 `[业界通识]`。

**讲给客户的一句话**：调度器的核心不是"多接请求"，而是控制单轮 Prefill 的预算，让已开始的对话不被卡住。

**深挖入口**：帖子「MiniInfra系列之推理请求调度器」（8/31）· 教材 p183

### ch07 · Chunked Prefill

**解决什么问题**：软预算下长 Prompt 必须整条执行：一条 2048 Token 的 Prefill 会让同批 Decode 停数百毫秒。只有单条 Prompt 可拆，Token Budget 才能成为硬约束。

**机制通俗版**：baseline（full）沿用 ch06 软预算，长 Prompt 一次算完、可以超预算；variant（chunked）给请求新增 `prefilling` 状态和 `prefill_cursor`，把 Prompt 按 Token Budget 切成多轮，每轮只把本段 K/V 追加到 Block Table，中间 Chunk 不出 Token，最后一块完成才生成首 Token。2048 Token 长 Prompt 干扰实验：budget 256 把最大单轮 Prefill 从 316.28 缩到 65.39 ms（-79.3%）、既有请求最大 ITL 降 67.8%、峰值显存降 16.8%；代价是长请求 TTFT +141.6%、吞吐 -6.2%。两个负对照很说明问题：短 Prompt 场景 chunked 没有可拆工作，反而慢 6.4%；无并行 Decode 场景收益按定义为零。budget 1024 是本负载下吞吐与 TTFT 更平衡的档位。

**架构上的位置**：依赖 ch05 的 Paged Cache（支持跨块追加）和 ch06 的调度器，完成"无预算 → 软预算 → 硬预算 + Chunked Prefill"的调度三连。边界：这是引擎内部的序列增量计算，不是 RAG 切分或流式接收；chunk 越小停顿越短，但迭代与 Kernel Launch 开销越多。

**业界对应**：vLLM/SGLang 均默认启用 Chunked Prefill `[业界通识]`。

**讲给客户的一句话**：Chunked Prefill 把长 Prompt 拆成多次小步，用首 Token 稍晚换取其他请求不掉帧——本质是延迟在请求间的再分配。

**深挖入口**：帖子「从零实现Chunked Prefill」（9/3）· 教材 p225

### ch08 · Prefix Cache

**解决什么问题**：多条请求共享同一长前缀（系统提示、few-shot、文档）时，前缀 KV 被反复重算。块级 Prefix Cache 让后续请求复用已算好的完整 Prompt KV Block。

**机制通俗版**：baseline 每条请求重算完整 Prompt；variant 匹配最长连续完整前缀，只算未命中后缀。两个安全设计是关键：一是缓存键不是对当前块单独哈希，而是 `SHA256(model_namespace || parent_hash || block_token_ids)`——相同 Token 块在不同历史上下文里 KV 不同，不能误命中；二是只共享已填满并发布为不可变的 Block，未填满的尾块和 Decode Block 私有，因此无需 Copy-on-Write；且 KV 不含首 Token Logits，最多复用 `prompt_length - 1`，至少重算最后一个 Prompt Token。实测 8 条请求共享 2048 Token 前缀：热请求 service TTFT p50 降 86.30%、Prefill Token 减少 84.85%，但 Makespan 只降 11.33%——Decode 不受益，且缓存池多占 280 MiB。生命周期用引用计数 + leaf-LRU：请求结束只减请求引用，缓存索引仍持有的块不回池。

**架构上的位置**：依赖 ch05 的物理块与 ch07 的 Chunked Prefill。边界：不做部分块共享/COW、并发冷请求去重、跨模型或跨进程共享（后者是 ch14）。

**业界对应**：vLLM Prefix Caching、SGLang RadixAttention `[业界通识]`；跨进程共享见 LMCache（ch14）。

**讲给客户的一句话**：命中率取决于业务前缀是否整齐；命中省的是 TTFT 和 Prefill 算力，Decode 一秒都省不了。

**深挖入口**：帖子「从零实现Prefix Cache-infra系列教程8」（9/4）· 教材 p256

### ch09 · FlashAttention

**解决什么问题**：Eager Attention 显式物化 `QK^T → softmax → PV` 的完整 Score 和 Probability 矩阵，中间显存随序列长度平方增长——4096 Token 增量峰值 2576 MiB，8192 直接 OOM。

**机制通俗版**：baseline 是 Eager 路径；variant 用 Triton 手写 inference-only FlashAttention 前向：分块读取 K/V，每个 Query 行维护最大值 `m`、归一化和 `l` 与累加器 `acc`，用 Online Softmax 逐块修正（`alpha = exp(m - m_new)`），完整中间矩阵不落显存。数学结果相同，计算量仍是二次方，降的是显存读写。微基准（因果 Attention、head_dim 128）：1024 Token 加速 3.38×、4096 达 3.70× 且增量峰值仅 16 MiB；但接入完整引擎后端到端只有 1.08–1.10×——Chunked Prefill 把 Query 限制在 256，Projection、MLP、调度和 Cache 管理稀释了 Kernel 收益。这是"Kernel 加速比 ≠ 模型加速比"的典型教材。工程细节：Chunked Prefill 要用全局因果偏移 `k <= P + q`，不能按局部位置算。

**架构上的位置**：依赖 ch05–ch08 的引擎底座；不依赖第三方 flash-attn 包。边界：不支持 backward/任意 Mask/直接读 Block Table 的 Paged Attention。

**业界对应**：FlashAttention 官方库、vLLM/SGLang 内置融合 Attention `[业界通识]`。

**讲给客户的一句话**：FlashAttention 省的不是计算量而是显存搬运，序列越长收益越大，但整体加速要看它占总时间多少。

**深挖入口**：帖子「手写FlashAttention」（9/6）· 教材 p291

### ch10 · CUDA Graph

**解决什么问题**：Decode 每步计算量很小，而 Eager 路径单步约 1903 次 `cudaLaunchKernel`——Host 提交开销占了主导。

**机制通俗版**：baseline（static_eager）每轮逐个提交 Kernel；variant 把同一组 Kernel 提前捕获成 CUDA Graph，之后更新输入缓冲、一次 `cudaGraphLaunch` 重放。代价是静态化：每个 `(batch_capacity, context_bucket)` 建 Runner，预分配全部 Buffer 和覆盖 Bucket 的 KV Block，地址形状不变，无效位置用 Mask 屏蔽。实测：capacity 1、128 Token 下 Decode TPOT 从 37.90 降到 5.22 ms（7.26×）；但上下文与 Batch 变大后 GPU 工作变重，加速比降到 2.25×——Graph 不减少计算，能消除的只有 Host 开销。Capture 耗时 95–140 ms，约 3–5 个 Decode 步即可摊销。端到端提示：Prefill 仍是 Eager，2048 Token 请求 TPOT 提升 4.03× 但端到端只有 2.53×。

**架构上的位置**：依赖 ch09 的 FlashAttention 和静态 KV Pool。边界：需要 Bucket/Padding 换取可捕获性，Bucket 尾部有无效读取；请求准入和 Block 分配仍在 Graph 外。逐 Token `.item()` 同步会掩盖收益。

**业界对应**：vLLM/SGLang 均对 Decode 使用 CUDA Graph，按 batch/context 分档缓存 Graph 实例 `[业界通识]`。

**讲给客户的一句话**：CUDA Graph 解决的是"CPU 指挥慢于 GPU 计算"的问题，Batch 和上下文越大、剩余收益越小。

**深挖入口**：帖子「CUDA Graph」（9/8）· 教材 p324

### ch11 · Greedy Speculative Decoding

**解决什么问题**：Decode 逐 Token 串行调用大模型。Speculative Decoding 用小 Draft Model 快速提候选，大模型一次并行验证多个，减少串行 Target 调用。

**机制通俗版**：Qwen3-0.6B 做 Draft 连续提出 γ 个候选，Qwen3-4B 做 Target 一次块验证：接受最长匹配前缀，拒绝处补 Correction Token，全对时补 Bonus Token；两套 KV Cache 按同一接受边界提交或回滚。数学上 Greedy 输出与 Target-only 完全等价（FP16 下实测逐 Token 一致）。但实测在 3080 Ti 上几乎全是负优化：中文解释负载 γ=8 接受率仅 0.173，TPOT 加速比 0.284×；只有接受率 1.0 的代码样例在 γ=8 得到 1.023×。根因在组件成本：Draft 单步 42.51 ms 只比 Target 的 48.16 ms 快约 12%——Eager 教学栈里两者都被小算子和 Host Launch 拖住，参数量 6.7 倍差距没有转化为速度差，而一轮推测还要多付 γ+1 次 Draft 前向。

**架构上的位置**：本期是独立实现（PyTorch SDPA、float16）。边界：只做 Greedy（不做随机 Speculative Sampling）、Batch 1、无 MTP/EAGLE/Medusa；KV 回滚必须与输出提交同一接受边界。

**业界对应**：vLLM/SGLang 内置 speculative decoding；MTP/EAGLE 是更轻的候选生成器 `[业界通识]`。

**讲给客户的一句话**：投机解码是"用便宜计算赌接受率"，Draft 不够快或接受率不够高时是纯亏——上线前必须按业务负载实测。

**深挖入口**：帖子「Greedy Speculative Decoding」（9/10）· 教材 p362

### ch12 · KV Cache Offloading（抢占式换出）

**解决什么问题**：GPU Block 不足时新请求进不来。乐观准入 + 抢占：暂停受害请求，把它的 KV 挪走腾出显存，让系统级并发超过 GPU 池容量——容量不足从"失败"变成"延迟"。

**机制通俗版**：两个抢占路径对比。`swap`：整请求 KV 经 PCIe 换出到 Pinned CPU 池，恢复时换回断点续算；`recompute`：KV 直接丢弃，恢复时对 prompt + 已生成部分重新 Prefill。为什么是整请求：Decode 每步 Attention 要读全部历史 KV，"运行中部分换出"不存在。实测决策依据清晰：swap 双向传输约 49–63 ms（实测约 11.9 GB/s，仅对 PCIe Gen3 x16 平台有效），recompute 的重新 Prefill 暴露 673–764 ms——本环境下 swap 机制成本明显更低，但端到端均值受负载支配，不能只看一次均值。容量收益成立：同池下 swap 吞吐 37.0 vs 保守准入 35.7 token/s（池 230）、44.9 vs 40.7（池 320）；但都低于 664 块大池的 45.5——池充足时抢占收益为零。ITL max 约等于受害请求的秒级暂停，主要代价落在被抢占请求自身。

**架构上的位置**：依赖 ch05 的 Paged Cache 与 ch07 的调度器；单请求上下文上限仍由 GPU 池决定。换出副本只用于恢复，不做共享复用（那是 ch14）；同步拷贝停顿是 ch13 的优化对象。

**业界对应**：vLLM 的 swap 抢占与 CPU Offload `[业界通识]`。

**讲给客户的一句话**：换出把"显存不够就失败"变成"慢一点但能跑"，代价由被抢占请求的延迟承担。

**深挖入口**：帖子「KV Cache Offloading」（9/12）· 教材 p381

### ch13 · KV Cache 异步传输

**解决什么问题**：ch12 的换出/换入在默认计算 Stream 上同步拷贝，拷贝期间 GPU 不推进其他工作——传输等待全部暴露。

**机制通俗版**：baseline（sync）在计算 Stream 逐块传输，完成后调度器才继续；variant（async）在独立 Copy Stream 上提交 D2H/H2D，用 CUDA Event 延迟转移 Block 所有权：D2H 完成前 GPU 源块不能释放，H2D 完成前目标块不能读；调度器只在无工作可做或下一步必须申请 Block 时才等 Event。请求新增 `swapping_out / swapped / swapping_in` 中间状态。受控微基准证明本机具备重叠能力（传输与 GEMM 并发 1.21–1.50×）；但主实验不是单向正优化：池 230 组 async 略慢（39.31 vs 39.61 token/s），池 320 组较快（47.62 vs 45.55）——D2H 在紧急水位触发、完成前源块不能释放，重叠窗口可能大部分暴露。机制结论要看分阶段 Event 指标：async 把暴露等待从 48.3 降到 20.0 ms（池 230）。

**架构上的位置**：依赖 ch12 的两级池与抢占语义。边界：能隐藏多少取决于有无独立计算和抢占触发时机；"异步提交"不等于"传输完成"，也不保证端到端更快。

**业界对应**：vLLM/SGLang offloading 与 NIXL 等传输层的异步流设计 `[业界通识]`。

**讲给客户的一句话**：异步传输的正确性靠 Event 管理块所有权，收益靠"传输时有没有别的计算可做"——不是开了就快。

**深挖入口**：帖子「KV Cache 异步传输」（9/14）· 教材 p406

### ch14 · 外部 KV Cache 与 LMCache

**解决什么问题**：ch08 的 Prefix Cache 只在同一进程内共享物理块；多引擎实例、重启后的进程都无法复用彼此的 KV。外部 KV Cache 用进程无关的标识把 KV 变成可跨进程、跨请求命中的服务。

**机制通俗版**：baseline（recompute）新请求完整重算；variant（external）用链式 CacheKey `SHA256(model namespace || parent digest || token ids)` 标识 256 Token 的外部 Chunk，向独立 Cache Service 发 Lookup/Load/Store，命中的 CPU KV 重建到新申请的本地 GPU Block（重建 Block Table，因为外部对象不含本地 Block ID），只 Prefill 未命中后缀。与 ch08 的关键差异：只存 `prompt_length - 1` 范围的完整 Chunk，KV 不含首 Token Logits，所以至少要重算最后一个 Prompt Token 才能出首 Token。正确性完美：跨进程命中 512 Token、warm Prefill 只算 1 Token、输出与重算逐位一致。但性能全是负优化：2048 Token 前缀 warm TTFT 2131.68 ms vs 重算 480.62 ms（0.23×）——Lookup 只要约 1 ms，成本在 TCP 传输（Payload 约 235 MB）与逐 Chunk 重建。"已经命中"不等于"值得加载"。

**架构上的位置**：与 ch12/13 不同：CPU KV 由 Token Chunk 标识，可被任意请求/进程命中，不只恢复原请求。业界映射：ExternalCacheClient→Connector、Memory Object、CPU Backend + LRU 等 LMCache 职责。边界：同步 TCP + 用户态复制，非共享内存/RDMA；命中收益取决于传输路径与模型规模。

**讲给客户的一句话**：外部 KV Cache 的价值在"命中率 × 复用范围"，但必须保证加载比重算快——慢速传输会让缓存变负资产。

**深挖入口**：帖子「外部KV Cache和LM Cache」（9/16）· 教材 p424

### ch15 · Prefill/Decode 分离

**解决什么问题**：Prefill 是计算密集、Decode 是访存密集，混在一个 Worker 里互相干扰。P/D 分离让 Prefill Worker 算完 Prompt KV 和首 Token，把 KV 移交给 Decode Worker 继续生成。

**机制通俗版**：baseline（monolithic）同一进程完成 Prefill、首 Token 和 Decode；variant（disaggregated）跨进程 Handoff：Prefill Worker 导出逻辑 KV Payload，经同步 GPU→CPU→TCP→CPU→GPU 路径发布，Decode Worker 重建本地 Paged KV 后返回 ACK，确认所有权转移后才继续生成、首 Token 才可发给客户端（ACK-gated）；Payload 损坏或 Namespace 不匹配时 fallback，Decode 侧完整重算保证正确。正确性达标：三个 Worker PID 不同、首 Token Logits 摘要一致、输出与 monolithic 一致。但单卡串行路径下明确负优化：2048 Token 的 KV Payload 224 MiB，split TTFT 是 mono 的 5.49×（2378 vs 433 ms），其中 Publish 约 896 ms、Export 约 433 ms；换 1.7B 压力模型倍数仅缩小到 5.23×，"负优化是模型太小"不成立——单请求 split 永远比 mono 多一次 KV 交接。

**架构上的位置**：与 ch14 的区别：Handoff 是同一 `Request ID + Attempt ID` 的一次性所有权交接，不做前缀 Lookup、不留副本。系列至此指出真正收益所在：多 GPU P/D 并行吞吐与 KV-aware 调度（如 Mooncake `[业界通识]`）不在单卡实测范围。

**业界对应**：vLLM disaggregated prefill、DistServe、Mooncake 等多卡 P/D 分离方案 `[业界通识]`。

**讲给客户的一句话**：P/D 分离卖的是大规模下的资源配比和干扰隔离，不是单请求更快——单卡串行演示里它只会更慢。

**深挖入口**：帖子「手写PD分离」（9/18）· 教材 p453
