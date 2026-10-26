# 推理篇（chapter_01–chapter_15）

> 事实来源：`miniinfra/chapter_NN/README.md`（唯一变量、实测表、可推广/不可推广结论）。教案不复制原版，只讲机制、产品对应与方案判断。

### ch01 · 手写 Qwen3 推理

**先回答最基本的问题：语言模型到底在做什么？**

你在对话框打"你好"，模型一个字一个字往外蹦。它不是想好了再说话——它在玩一个高级接龙：每次只猜"下一个 Token 是什么"，猜完接到句子末尾，再猜下一个。"猜下一个"这个动作，就是大模型推理的全部。本章把这个接龙循环徒手拆开：不用 vLLM，不用 Transformers，只用 Python 和 PyTorch，从权重文件一路写到第一个字蹦出来，为全系列 43 章建立 baseline。

<!--player:ch01-->

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Token | 模型眼里的"字"——词表里的一个编号，不等于汉字或单词；"unbelievable"可能切成 3 个 Token |
| 词表 vocab | 模型出厂自带的字典：全部可能 Token 与编号，Qwen3 有 151,936 条 |
| Embedding | 查表：每个 Token 编号对应一行数字（向量），模型用向量思考，不用文字 |
| Logits | 猜下一个时给出的 151,936 个分数，每个分数 = "下一个是这个词表条目"的得分 |
| Greedy 解码 | 每次只取最高分——永远选第一名，不掷骰子 |
| Chat Template | 把"你说的/它说的"装进模型认识的固定格式信纸 |
| 自回归 | 自己吃自己的产出：把刚生成的 Token 接回输入，再猜下一个 |
| Safetensors | 存放 6 亿个参数数值的权重文件格式，本章要亲手读它 |

**这一章怎么工作**：读 `config.json`（模型图纸：28 层、32 个注意力头等）→ 读 Safetensors 权重 → 手写 Byte-level BPE（文本↔编号互转）→ 套 non-thinking Chat Template → 28 层逐层前向算出 151,936 个 Logits → argmax 选出下一个 Token → 接回句子末尾 → 循环。对照物是 Transformers：它只出现在对照程序里，不进推理路径——相同输入、greedy 条件下，三组 Prompt 的生成 Token 逐个一致，Logits 最大误差均为 0。

**算一算**：19 个输入 Token + 32 个输出（3080 Ti、bf16、README 实测）：先把 19 个字一次性读完（Prefill）只用 34.43 ms；之后每写一个新字平均 33.40 ms，32 个字的端到端模型计算约 1069.80 ms，峰值显存 1173.80 MiB。**注意那个奇怪的点**：读 19 个字只要 34ms，"看一遍历史再写一个字"却也要 33ms——因为每一步都把全部历史从头重算了一遍（本章没有 Cache）。历史越长每步越贵，这正是 ch02 要修的第一个浪费。

**边界与常见误区**：仅支持固定 revision 的 0.6B dense 模型、batch size 1、greedy、关闭 thinking 的纯文本 system/user 模板，不支持工具调用与多模态。[易错] "手写的不如 Transformers 准"——对照路径下两者逐 Token 一致、误差为 0；Transformers 在本章只是正确性参照。单次测量不能外推到别的硬件。

**产品对应**：任何推理引擎的最小可运行内核 `[业界通识]`；Transformers/HF 在本章只作正确性对照，不进执行路径。

**方案视角**：评估任何加速方案之前，先要一份可复现 baseline。向供应商问两个问题："你们的加速比以什么为分母？""在什么模型、硬件、dtype 上测得？"——没有 baseline 的加速比无从审计。

**讲给客户的一句话**：谈加速之前，先建立一份可复现的 baseline——否则"优化"无从谈起。

**掌握目标**：能画出"接龙循环"全图（Prefill 读完 → 逐 Token Decode → 接回末尾）；能答出为什么 Decode 每一步的成本随历史长度增长。

**深挖入口**：帖子「手写Qwen3推理」（8/24）· 教材 p1 · 仓库 chapter_01

### ch02 · KV Cache

<!--player:ch02-->

先看一个日常场景：你和模型聊到第十轮，它每蹦出一个字，都要把你前面说过的一百个字从头读一遍再写一个字。这就是 ch01 手写推理的现状——历史 Token 的中间结果用完即扔，每一步 Decode 都对完整上下文重跑所有 28 层。KV Cache 是对这个浪费最直接的回应：算过一次的东西，存下来别扔。要理解它存的是什么，得先把"一个字怎么被处理"拆到注意力计算的每一步。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Q（Query） | 当前 Token 发出的"检索词"：我在找什么样的信息 |
| K（Key） | 每个 Token 的"标签"：我能被什么样的检索词匹配到 |
| V（Value） | 每个 Token 的"内容"：一旦匹配上，取走的就是它 |
| 注意力三步 | 出题（算 Q）→ 配对打分（Q 和所有 K 点积+softmax）→ 取内容（按权重加权 V） |
| KV Cache | 历史每层的 K/V 存下来复用，"配对打分"和"取内容"照常做，K/V 不再重算 |
| Prefill | 一次性读完整个 Prompt 的一次前向计算，产出首 Token 并建 Cache |
| Decode | 之后逐 Token 生成：每步只算 1 个新位置，但要读全部历史 K/V |
| RoPE | 位置编码，给 Token 标"位置编号"；位置从 Cache 长度续算，不从头数 |
| GQA | 多个查询头共用一组 KV 头（Qwen3 是 4:1），KV 存储按 KV 头数算 |

**先讲清注意力在算什么**（后面十几章全靠这个底子）。一层注意力对当前 Token 做三步：**第一步出题**——用当前 Token 的向量算出一个 Query（"我在找什么"的向量）；**第二步配对打分**——Query 和历史里每个 Token 的 Key 做点积，得到"该多看谁"的分数，softmax 归一成权重；**第三步取内容**——按权重把历史各 Token 的 Value 加权求和，得到当前 Token 在这一层的输出。所以 K 和 V 不是玄学：**K 是"我被怎么找到"的标签，V 是"找到我之后取走的内容"**——像图书馆检索：书脊标签是 K，书里的内容是 V，你脑中的检索词是 Q。

**为什么没有 Cache 时每一步都贵**：没有 Cache 的写法里，写第 21 个 Token 时，前 20 个 Token 的 Q/K/V 和全部中间结果都被重新计算（28 层 × 20 个位置）。但注意力的数学里藏着一个被浪费掉的免费午餐：**历史 Token 的 K/V 只依赖"过去"，不随"现在在写谁"变化**——写第 20 个字时算出的第 3 个字的 K/V，和写第 21 个字时算出的第 3 个字的 K/V，一模一样。RoPE 位置编号也是固定的。重复计算它们 = 纯浪费。

**variant 的做法**：Prefill（第一次读 19 个 Token）时顺手把每层每头的历史 K/V 存进 Cache；此后每步 Decode 只算**新 Token 自己的** Q/K/V（一个位置），K/V 追加进 Cache，Q 拿去和 Cache 里全部 K 打分、按权重取 V 加权。历史不再重算，只有新 Token 走 28 层。Cache 的形状是每层一对普通 Tensor：`[batch, num_kv_heads, seq_len, head_dim]`，教学实现用 `torch.cat` 逐步增长；Key 在 QK Norm 与 RoPE 之后写入。float32 对照下两条路径逐 Token 一致——Cache 没改错东西。

**两阶段时间线**（看上图下半）：第 1 步 Prefill，19 个位置**并行**过 28 层，建 Cache、出首 Token——算得多但一次做完；第 2 步起是 Decode，每步只算 **1 个新位置**，但要**读**前面所有位置的 K/V——"算得少、读得多"。这解释了 ch01 那个怪现象的另一半：读 19 个字只要 34.43ms（并行），写一个字也要 33.40ms（串行读全部历史）。也埋下后面所有章节的伏笔：Decode 的瓶颈从"算力"变成了"把 KV 从显存搬进计算单元的**带宽**"。

**算一算**：Cache 数据量 = `2 × layers × num_kv_heads × head_dim × seq_len × batch × dtype_bytes`。Qwen3-0.6B：28 层 × 8 个 KV 头 × 128 头维 × 2 字节（bf16）× 2（K 和 V）≈ **0.11 MB/Token**——1024 Token Prompt 生成到 1055 长度时实测 Cache 恰好 115.39 MiB，与公式吻合。

**数字**（3080 Ti、bf16、batch size 1、32 输出 Token、README 实测）：16 Token 短 Prompt 上 KV Cache 反而**负优化**——端到端 1191.07 vs 1011.79 ms，因为 Qwen3-0.6B 单步算量极小，Python 调度、Kernel Launch 和 `torch.cat` 再分配把省下的计算吃掉了；到 1024 Token 才有约 `1.85×` 加速，Decode 均值 60.43→32.64 ms，端到端 1930.00→1067.75 ms。收益随上下文变长才从负转正，这条曲线本身就是本章最重要的实测结论。

**重点**：**KV Cache 只优化 Decode、不优化 Prefill；收益随上下文变长才显现——短上下文上管理开销可能反超节省的计算。**

**边界与常见误区**：KV Cache 不减少 Prefill 对完整 Prompt 的计算；Decode 时 Query 仍要读全部历史 K/V，所以 Decode 并非与上下文长度无关的常数复杂度；`torch.cat` 动态增长的教学实现也谈不上高效。[易错] "开了 KV Cache 一定更快"——16 Token 组实测更慢，结论必须按负载给出。

**产品对应**：KV Cache 是所有推理引擎的核心组件；vLLM 等引擎的 Cache 预分配与分页管理（ch05 起）远优于本期的动态 `torch.cat` `[业界通识]`。

**方案视角**：先问客户的判据是业务 Prompt 的典型长度：上下文越长，KV Cache 及其显存管理越关键，长文档/RAG 场景几乎必谈；全是几十 Token 的短对话时，重点反而落在调度与其他开销上。看哪个指标——Prompt 长度分布，而不是模型大小。

**讲给客户的一句话**：KV Cache 让"每写一个字重读全部历史"变成"只算新字"，收益随上下文变长才显现，短 Prompt 小模型上管理开销可能吃掉全部节省。

**延伸阅读**：[Sebastian Raschka: What is a KV cache](https://sebastianraschka.com/faq/docs/kv-cache.html)（问答式精讲）· [Redis: Prefill vs Decode](https://redis.io/blog/prefill-vs-decode/)（两阶段分工）· [阿里云开发者：理解 KV Cache](https://developer.aliyun.com/article/1736244)（中文详解）

**掌握目标**：能分步说出一次注意力的三步计算，并指出 Cache 截掉的是三步里哪部分重复；能算出给定层数、KV 头数、head_dim、序列长度下的 KV Cache 字节数；能判断"开了 KV Cache 是否更快"取决于哪些条件。

**深挖入口**：帖子「kv cache」（8/25）· 教材 p45 · 仓库 chapter_02



### ch03 · Static Batching

<!--fig:ch03-->

ch02 修好了"单请求"的浪费，但服务台不可能只服务一个人。单请求跑模型时 GPU 利用率很低——每次 Decode 只算 1 个位置的小矩阵，算力大部分在空转。最朴素的"凑大批"是把多个请求拼进同一个 Batch、共享一次模型前向：GPU 一次算 16 个人的活。但请求长短不齐时，省下来的算力会从两条缝里漏回去：补位的无效 Token 也占计算，先完成的槽位只能干等。这两条缝，正是本章要把机制拆开看的地方。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Batch | 一组请求拼在一起，共享一次模型前向执行 |
| Padding | 把短序列补到统一长度，凑成矩形 Tensor |
| 左 Padding | 把补位放左边，让每个请求的有效 Token 右对齐 |
| Attention Mask | 标出哪些位置是真 Token、哪些是补位，补位不进注意力 |
| 槽位 | Batch 中的一个固定名额，配一块独立 KV Cache |
| makespan | 一批任务从开始到最后一个完成的全部耗时 |
| Position IDs | 每个 Token 的位置编号，按各自有效长度增长 |

**这一章怎么工作（拆到计算步骤级）**：baseline 是多个带 KV Cache 的请求逐个执行——每个请求各自 Prefill、各自 Decode，GPU 每次只吃一个小矩阵。variant 把 N 个异长 Prompt 装进一个成员固定的 Batch，分三步走。**第一步**：左 Padding——长度 `[16,32,64,128]` 的 Prompt 全部补到 128，凑成 `[batch, 128]` 矩形 `input_ids`，同时配两张辅助表：`attention_mask`（标记哪些列是真 Token）和 `position_ids`（每个请求的有效 Token 仍从逻辑位置 0 起 RoPE 编号，不用物理列号）。**第二步**：一次 Batched Prefill——矩形全体并行过 28 层，注意力的"配对打分"里补位列被 Mask 挡住不进打分，但**矩阵形状仍按 128 列执行**，无效列的计算一个不少；每层的 K/V 按槽位写入 `[batch, num_kv_heads, cache_length, head_dim]`。**第三步**：按槽位同步 Batched Decode——每轮 1 次前向同时给 4 个槽位各算 1 个新位置、各读各的历史 K/V；输出预算 `[8,16,32,64]` 的槽位分别在第 8、16、32、64 轮写完，之后它们的列被 Mask 成"哑槽"，矩形 Tensor 照跑——**但 Batch 一旦开始不进新请求**，空槽不复用。float32 对照下换同伴、换排列的 Logits 误差为 0，无请求串扰。

**回到两阶段时间线**：Batched Prefill 还是"算得多、只做一次"（N 个人的 Prompt 并行读完），Batched Decode 还是"算得少、读得多"（每槽 1 个新位置读各自历史）——批处理改变的是**同时算几个人**，没有改变每个人"算多少、读多少"的形状。这就是为什么 Padding 的账只出现在 Prefill（矩形按最长 Prompt），空转的账只出现在 Decode（矩形按最长输出轮数）。

**算一算**：输出预算 `[8,16,32,64]` 的四人批次必须跑到第 64 轮：有效槽位利用率 = (8+16+32+64)/(4×64) = **46.88%**——一半以上的执行轮次在做无效计算。同款账也适用于 Padding：长度 `[16,32,64,128]` 补到 128，有效率同为 (16+32+64+128)/(4×128) = 46.88%。

**数字**（3080 Ti、bf16、README 实测）：收益侧，Batch Size 16 相对串行 makespan 约 `13.98×` 加速，Batched Cache 随 Batch Size 线性增长。浪费侧，混合长度组 `[16,32,64,128]` 的 Prefill 有效吞吐 6880.90 token/s，几乎只有整齐组 14641.75 token/s 的一半，而 Prefill 延迟和峰值显存几乎不变——浪费来自无效 Token 也占满计算，不是"短 Prompt 拖慢 Batch"。容量侧还有个反直觉数字：Batch 192 吞吐 2981.83 token/s 高于 196 的 2931.97，且 197 起在 Prefill LM Head 处 OOM（该处需单次分配约 7.14 GiB 的 Logits Tensor）——最大可跑 Batch 不是吞吐最优 Batch。

**重点**：**固定 Batch 的吞吐收益来自并行度，而 Padding 浪费与空转槽位会把它抵消——拼批的关键是请求长度整齐，不是越大越好。**

**边界与常见误区**：bfloat16 下 Batch Shape 改变矩阵 Kernel 的归约顺序，候选 Logits 接近时可能分叉（Batch Size 2 时第 4 步分叉、Top-1 一致率 37.5%）——这是数值现象，不是"Batch 降低准确率"；离线结果不含排队与网络，不代表在线 SLO。[易错] 把 46.88% 的浪费读成"短 Prompt 拖慢别人"——真实原因是矩形 Tensor 按最长长度执行。

**产品对应**：离线批量任务（评测打分、数据标注）仍用固定 Batch 思路 `[业界通识]`；在线服务已被 Continuous Batching（ch04）取代。

**方案视角**：面向离线批量客户时先问"请求长度分布是否整齐？"——看哪个指标：请求长度分布的离散程度。整齐的可用静态拼批吃满并行度，不齐的收益先打个对折；在线交互业务不看这章，直接看下一章的调度。

**讲给客户的一句话**：拼批能提吞吐，但请求长短不齐时 Padding 和空槽可能浪费一半算力——批不是越大越好。

**掌握目标**：能算出给定长度分布下 Batch 的 Padding 有效率；能区分"批越大越好"与"吞吐最优 Batch"两个命题。

**延伸阅读**：[NVIDIA: Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)（推理优化总览，含批处理定位）

**深挖入口**：帖子「Static Batching」（8/26）· 教材 p72 · 仓库 chapter_03

### ch04 · Continuous Batching

<!--fig:ch04-->

固定 Batch 留下的最大裂缝是"先完成的不能退场"：在线场景里请求随时到达、输出长度随机，ch03 的批次却要等最长的那个结束才换人——槽位常年空着，排队的人干瞪眼。Continuous Batching 把换人粒度从"整批"细化到"每一次模型迭代"：谁写完最后一个字，谁的名额立刻让给等待队列里的人。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 迭代 | 模型前向的一次执行，Decode 每步就是一次迭代 |
| Continuous Batching | 每次迭代前回收完成槽位、补入新请求的动态拼批 |
| 状态机 | 每个请求的流转记录：waiting→running→finished |
| FCFS | 先到先服务：按到达顺序填空槽，最简单的调度规则 |
| TTFT | 首 Token 延迟：从请求到达到第一个字吐出 |
| ITL | Token 间隔：Decode 阶段相邻两个字的间隔时间 |
| dense Cache | 按最大并发和物理长度预留的整块矩形 KV Cache |

**这一章怎么工作（拆到计算步骤级）**：为每个请求维护 `waiting → running → finished` 状态机，然后**每次模型迭代前执行固定三拍**。**第一拍，回收**：扫描 running 集合，把写完最后一个 Token 的请求标记 finished，清空其槽位的有效 Mask，腾出空位。**第二拍，填补**：按 FCFS 从 waiting 队列按到达顺序把新请求填进空槽，空位与已到达请求同时存在时本期固定 Prefill 优先。**第三拍，执行**：本迭代有新请求就对它们做一次 Batched Prefill（读各自整个 Prompt、建 Cache、产首 Token）；没有新请求就对剩余槽位做一轮 Batched Decode（每槽算 1 个新位置、读各自全部历史 K/V）。baseline 与 variant 用相同的 FCFS 顺序、相同的最大运行请求数，唯一变量是**完成名额能否在迭代边界被复用**。正确性靠对照验证：float32 下两条 Batch 路径与单请求独立执行逐 Token 一致，换槽轨迹 `[0,1] → [2] → [3]` 没有串扰。KV Cache 本期仍是 dense 固定容量 `[max_running_requests, num_kv_heads, physical_length, head_dim]`，新请求更长时全批左扩容搬移——这个浪费留给 ch05。

**回到两阶段时间线**：批处理形状没变——Prefill 仍是"算得多、只做一次"（一次读完新请求的全部 Prompt），Decode 仍是"算得少、读得多"（每槽 1 个新位置读全部历史）。变的是**时间线上 Prefill 出现的时机**：ch03 里 Prefill 只在整批开头出现一次；ch04 里每当空槽遇到新请求，一轮完整 Prefill 就插进 Decode 流。这正是延迟恶化的根源——512 Token 的 Prefill 是"19 个位置的算术放大 27 倍"的一次大算，会把它撞上的那几轮 Decode 全部堵住。

**算一算**：固定到达轨迹（每 150 ms 一请求、Prompt 128、输出 32）里，第 5 个请求的 TTFT 差距可以直接推出来。固定批次：前 4 人占满 4 个槽位，须整批跑完（约 1 次 Prefill + 32 轮 Decode ≈ 34 + 32×34 ≈ **1500 ms** 量级）才有空槽——实测 TTFT p95 1511.07 ms 与这个推算吻合。Continuous：有人第 8 轮写完就腾槽，新请求在第 9 轮左右做完 Prefill 出首 Token——实测 TTFT p95 205.39 ms，差出约 7 倍的就是"迭代边界换人"这一件事。

**数字**（3080 Ti、bf16、逻辑时间线、README 实测）：异长输出场景 makespan 加速 `1.169×`、吞吐 +16.8%、执行槽位有效率 62.50%→74.07%；等长输出负对照里没有可提前复用的槽位，两条路径只差约 0.4%——机制成立的前提是负载不整齐。固定到达轨迹下收益更明显：makespan 加速 `1.486×`、TTFT p95 1511→205 ms，但 ITL p95 36.50→71.48 ms：新请求的完整 Prefill 会阻塞已有请求的 Decode，干扰实验里 512 Token Prompt 中途到达使 ITL 最大值 35.07→72.71 ms，而 p95 只有 34.89 ms 会掩盖这个局部尖峰。

**重点**：**Continuous Batching 在迭代级复用完成槽位，但新请求的完整 Prefill 会卡住已有请求的 Decode——这个矛盾要等 Chunked Prefill 才能解决。**

**边界与常见误区**：本期的"Prefill 优先 + FCFS"是为了得到唯一、可复现的执行路径而固定的策略，不代表生产调度器的最优选择；CPU 调度与 Cache 管理仍有约 2.3% 开销；dense Cache 扩容须搬移。[易错] 把 ITL p95 从 36.50 恶化到 71.48 ms 读成机制失效——这是 Prefill 阻塞其他请求的确认代价，不是槽位复用本身的错误。

**产品对应**：vLLM/SGLang 等在线引擎的标准调度方式，思路源自 Orca 论文的 iteration-level scheduling `[业界通识]`。

**方案视角**：评估在线引擎时这是必选能力，没有它在线并发吞吐无从谈起；现场先问"TTFT p95 和 ITL p95 是否同时可接受"——只看一项会被 Prefill 干扰欺骗，p95 也会掩盖局部尖峰，要连最大值一起看。

**讲给客户的一句话**：Continuous Batching 让新请求见缝插针补进批次，新请求更少等，但 Prefill 与 Decode 会互相干扰，延迟分位要一起看。

**掌握目标**：能画出请求在 waiting→running→finished 状态机中的槽位复用流程，并说出每次迭代前的固定三拍；能判断什么负载下 Continuous Batching 无收益。

**延伸阅读**：[Redis: Prefill vs Decode Explained](https://redis.io/blog/prefill-vs-decode/)（两阶段分工与干扰）· [NVIDIA: Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)（含 Chunked Prefill 的总览）

**深挖入口**：帖子「连续批处理」（8/27）· 教材 p102 · 仓库 chapter_04

### ch05 · Paged KV Cache

<!--player:ch05-->

ch04 的 KV Cache 还是 dense 整块预留：物理宽度等于这批请求里出现过的最长 Cache。于是一个 4096 Token 的长请求会让所有槽位——哪怕只装 128 Token——都在 4096 级别上跑 masked Attention；长请求走后，宽度也不缩回来。长度不均时显存浪费严重、并发上不去，而"上万人同时用"恰恰意味着长度分布最杂乱。Paged KV Cache 借用操作系统的老办法：分页。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Paged KV Cache | 把 KV 显存切成固定小块，按需分配给各请求 |
| Block | 固定大小的物理 KV 块，一次装 block_size 个 Token |
| Block Pool | 预分配好的物理块大池子，所有请求共用 |
| Block Table | 每请求的逻辑页表：逻辑块号→物理块号 |
| 尾块 | 请求最后一个未装满的块，碎片只出现在这里 |
| gather | 按页表把散落的物理块收起来读的操作 |

**这一章怎么工作（拆到计算步骤级）**：baseline 沿用 dense 槽位——物理长度等于运行集合的历史最大 Cache 长度，长请求加入时所有层所有槽位一起左扩容搬移，完成后只清 Mask、宽度不缩。variant 借鉴操作系统的虚拟内存分页：显存切成固定大小的物理 Block，一个 Block 携带全部层 K/V（`[num_layers, 2, kv_heads, block_size, head_dim]`），所有层共用同一张映射表；每个请求只存一张 Block Table 和逻辑长度——逻辑连续不要求物理连续。执行流程固定四步。**第一步，申请**：Prefill 时按 `ceil(prompt_len / block_size)` 从 Block Pool 取空闲块，例如 258 Token 的 Prompt 在 block_size 16 下取 17 块；Prompt 的 K/V 逐层写入对应块。**第二步，写尾块**：每步 Decode 的新 Token K/V 追加进该请求最后一个未满的块（尾块）。**第三步，满了再取**：尾块装满时才从池里再取一个空闲块挂到 Block Table 末尾，平时零开销。**第四步，gather 读**：做注意力的"配对打分 + 取内容"时，按逻辑长度分组，经 Block Table 把散落的物理块收起来按序读——尾块超出逻辑长度的槽位被 Mask。请求完成后物理块归还空闲堆，给后续请求复用，全程不动其他请求已写入的 K/V。float32 对照下 dense 与 paged 逐请求 Token 完全一致，换槽与 Block 复用轨迹也没有串扰。

**回到两阶段时间线**：每槽"算多少、读多少"的形状没变——Prefill 一次读完整 Prompt，Decode 每步算 1 个新位置、读全部历史。分页改变的是"读到哪里去"：dense 读的是一整条连续预留的槽（短请求也在按最长宽度跑 masked Attention），paged 只读 Block Table 指到的有效块——同样读 20 份 K/V，不再附带 4000 份无效槽位的搬运。这是把 basics 里"读了存着别丢"的那条伏笔落了地。

**算一算**：dense 槽位显存 = 最长请求 × 槽位数 × 0.11 MB/Token（ch02 公式）。10 个槽位、最长 4096：4096×10×0.11 ≈ **4.5 MB/层×全部层**的预留，哪怕 9 个请求只有 128 Token。paged：每个请求按 `ceil(实际长度/16)` 申请，128 Token 只占 8 块，碎片被限制在每个请求自己的尾块里（不足 16 Token）——10 个 128 Token 请求按需只付约 128×10 的量，加上尾块零头。

**数字**（3080 Ti、bf16、README 实测）：硅上的双面结果——长度不均（3072 Token 长请求）时 KV 读取减少 94.1%、Cache Pool 减少 82.6%，但同并发吞吐反而降 36–45%：PyTorch reference 的 Block gather、分组 Attention 和额外 Kernel Launch 吃掉了计算节省。真正的收益在容量：12GB 卡上固定负载下 dense 最大成功并发 8（12 起 OOM），paged 跑到并发 16、吞吐 157.80 token/s 还略高于 dense 最优约 0.9%，峰值显存低约 43.0%。

**重点**：**分页把显存占用从"按最长请求预留"变成"按需分配"——收益主要在并发容量；同并发下的单步速度要靠专用 Kernel 而非分页本身。**

**边界与常见误区**：减少 K/V 访问不等于端到端更快——Block gather 的 Kernel 效率同样决定成败；Block Pool 按已知上界一次预分配，没有动态准入。[易错] "上了分页吞吐就涨"——本期同并发吞吐是负的，价值是容量换吞吐：同一张卡多接一倍并发。

**产品对应**：vLLM 的 PagedAttention 是此机制的代表实现；vLLM/SGLang 的专用 CUDA/Triton Kernel 直接读 Block Table，没有本期 PyTorch reference 的 gather 开销 `[业界通识]`。

**方案视角**：先问客户两个问题："Prompt 长度分布有多离散？显存余量有多紧？"——看哪个指标：长度分布的离散程度加显存余量。长度越离散、显存越紧张，分页收益越明显；然后算容量账——同一张卡能多接多少并发，能不能少买一张卡。

**讲给客户的一句话**：分页把 KV 显存从"按最长请求预留"变成"按需分配"，同一张卡能多接一倍请求，代价是单步吞吐依赖引擎 Kernel 的成熟度。

**掌握目标**：能画出逻辑 Token 位置经 Block Table 到物理 Block 的映射，并按序说出"申请→写尾块→满了再取→gather 读"四步；能判断"减少 KV 读取"与"端到端变快"何时不是一回事。

**延伸阅读**：[vLLM PagedAttention (arXiv 2309.06180)](https://arxiv.org/abs/2309.06180)（分页机制的原始论文）

**深挖入口**：帖子「Paged KV Cache」（8/28）· 教材 p144 · 仓库 chapter_05

### ch06 · 迭代级推理请求调度器

<!--fig:ch06-->

想想你正在和一个模型长对话，聊到一半页面上突然一个字都不蹦了，几秒后才恢复——不是模型算不动，而是调度器把 GPU 交给了刚涌进来的新请求：它连续做了好几轮 Prefill，把已在生成中的会话晾在一边。ch05 解决了显存怎么分，但"每一轮迭代该干什么"仍无约束：调度器默认 Prefill 优先，新请求的突发可以无限挤压老请求。本章给调度器装上"算力预算"。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 迭代级调度 | 每次模型前向就是一个调度机会：每轮都重新决定这轮做 Prefill 还是 Decode |
| FCFS | 先来先服务：按到达顺序排队处理，不插队 |
| Token Budget | 每轮 Prefill 允许消耗的 Token 上限，像每轮的"饭量额度" |
| padded cost | 预算按批补齐后的矩形宽度扣账，补的空白 Token 也算钱 |
| 软预算 | 单条 Prompt 超预算不拆分，允许它整条执行、排队不清空 |
| ITL | Token 间隔时间：正在生成的用户两次看到新字之间的等待 |
| TTFT | 首 Token 时间：请求从提交到看到第一个字的延迟 |
| Decode 保护 | 强制规则：有请求在生成时，两次 Prefill 轮之间至少插一次 Decode 轮 |

**为什么需要**：baseline 是纯 FCFS + Prefill 优先，有空闲槽位就一次 Prefill 所有可加入的请求。突发 12 条长短混合 Prompt 时，调度器可以连续多轮做 Prefill，已有 Decode 停摆数百毫秒到秒级。调度优化的价值不是"多收请求"，而是限制每轮 Prefill 强度、强制插入 Decode 轮保护已开始的会话——界面上的"系统卡了"多半缺的就是这个。

**怎么工作（拆到计算步骤级）**：引擎维护 `waiting / running / finished` 三个队列，每轮模型前向前调度器先走三拍：**第一拍，算预算**——variant（budgeted）按 `prefill_cost = batch_size × max(prompt_length_in_batch)` 扣 Token Budget（Prefill 用左 Padding，预算按补齐后的矩形算，Decode 轮的成本就是 `running_request_count`）；**第二拍，FCFS 准入**——按到达顺序把 waiting 队头逐个装进本轮 Prefill，预算不够装下一个时立即停止，且不跳过队头去找更短的 Prompt（跳过会饿死队头）；**第三拍，Decode 保护**——已有请求在 Decode 时，两个 Prefill 轮之间至少执行一次 Decode 轮。每轮调度只产出一个 `SchedulerOutput`（阶段、请求、Token 数、是否超预算），执行器照它跑一次 Batched Prefill 或一次 Decode。注意"软"字的含义：本期没有 Chunked Prefill，队头单条 Prompt 超预算时不拆分、整条放行——否则 FCFS 队头被永久跳过。自检里最能说明机制成立：baseline 最大连续 Prefill 轮数为 2–3，budgeted 恒为 1（README「代码使用方法·CPU 快速自检」）。

**两阶段时间线**：用"算几个位置 × 读多少历史"的语言看每一轮——Prefill 轮：批内每条请求**并行**算它的全部 Prompt 位置（矩形补齐后最长那条决定成本）、顺手建 KV Cache；Decode 轮：每条运行中请求只算 **1 个新位置**，但要**读**各自全部历史的 K/V——"算得少、读得多"。两种轮混在同一张时间线上互相挤占，就是调度器要管的核心矛盾。

**数字**（3080 Ti、Qwen3-0.6B、突发 12 条长短混合，README「Prefill burst 与 Token Budget 实验」实测）：budget 2048 相对 baseline 吞吐 63.01→71.26 token/s（+13.1%），既有请求最大 ITL 906.84→376.79 ms（降 58.5%），突发请求 TTFT p95 降 15.8%，Prefill padded token 减少 53.0%。但 ITL p95 从 98.52 升到 207.37 ms——baseline 把 Prefill 集中成少数次长停顿，budgeted 拆成多次中等停顿：最大值改善、p95 反而恶化，这不能写成"延迟全面改善"。另一头，budget 256 过紧，出现 6 次单请求超预算加更多轮次开销，吞吐反而 -6.8%。

**算一算**：同样 3000 ms 的 Prefill 总量：baseline 集中 3 次各 1000 ms 的停顿（最大 1000），budgeted 拆成 10 次各 300 ms（最大 300）——"最坏一次"改善，"经常等多久"变差。max 看最坏一次，p95 看频繁的中停顿，两个指标可以反向变化。

**重点**：调度器的核心不是多收请求，而是用 Token Budget 限制每轮 Prefill 强度并强制插入 Decode 轮。

**边界与常见误区**：预算按 padded 而非有效 Token 计，补齐的空白也耗预算；且无显存准入、抢占、优先级与 Prefill/Decode 分离，budget 2048 只是本次扫描的最佳档位，不是通用最优。[易错] "预算调小总是更公平"——过紧的预算引发超预算执行和额外轮次开销，吞吐反向下降（256 档实测 -6.8%）。

**产品对应**：vLLM/SGLang 的迭代级调度都内置 Decode 保护这类机制；调度器里的 `max_num_batched_tokens` 参数就是 Token Budget 的产品形态。

**方案视角**：有已开始会话被新请求打断的在线交互业务都需要它；纯批量离线任务无所谓。评估时先问两件事："单轮 Prefill 最长多久？""已有会话的最长卡顿（ITL max）是多少？"并看 Token Budget 默认值是否按你们的负载扫过档。

**讲给客户的一句话**：调度器的核心指标是"已开始的对话不被新请求卡住"——要看最大停顿，而不是平均。

**延伸阅读**：[NVIDIA: Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)（推理优化总览，含调度与 Prefill/Decode 干扰）

**掌握目标**：能画出 FCFS + Token Budget 的每轮选择逻辑（含超预算队头处理）；能答出 ITL max 与 ITL p95 何时反向变化及原因。

**深挖入口**：帖子「调度器」（8/31）· 教材 p183 · 仓库 chapter_06

### ch07 · Chunked Prefill

<!--fig:ch07-->

一个人上传了 2048 Token 的长文档，点下发送的瞬间，整个服务里所有正在打字的人同时卡住——因为调度器要一口气把 2048 个 Token 的 Prefill 做完，这期间其他人一次 Decode 都轮不上。ch06 的 Token Budget 本该拦住这种事，但"软"字露了馅：预算不够时只能放行整条 Prompt，预算形同虚设。本章把这件事补齐：单条 Prompt 也可以拆成小块，分多轮做完。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Chunked Prefill | 把一条 Prompt 按预算切块，跨多轮迭代逐步做完 Prefill |
| Chunk | 一轮里被处理的那段 Prompt 切片，含若干 Token |
| prefilling 状态 | 请求新状态：Prefill 做了一部分、还没做完 |
| prefill_cursor | 游标：记录这条 Prompt 已经算到第几个 Token |
| 硬预算 | 预算不足时切断 Prompt 而非放行，预算真正有约束力 |
| 全局因果偏移 | Chunk 内位置仍按整条 Prompt 的全局位置算注意力范围 |
| 因果性 | 每个 Token 只能看见它之前的 Token，不能偷看未来 |

它是什么：不是文本切分（不受 RAG chunking），而是引擎内部对同一条 Token 序列的增量计算。

**怎么工作（拆到计算步骤级）**：请求状态机从 `waiting → running` 扩成 `waiting → prefilling → running → finished`，机制拆成三步：**第一步切块**——请求进入 `prefilling`，`prefill_cursor` 从 0 开始，每轮调度器按硬 Token Budget 给它切一段连续区间 `[start, end)`，末尾几个 Token 挪进 `running` 后可与其余请求拼批（共享前缀长度的多请求 Chunk 分在同一轮计划）；**第二步中间块只写 KV**——每轮只把本段 K/V 按 Prompt 的全局逻辑位置追加进 Block Table，中间 Chunk **不产出任何 Token**；**第三步末块才出首 Token**——最后一块完成时整条 Prompt 的 KV 齐了，生成首 Token、转入 `running`。时序从"[整条 Prefill] → Decode"变成"[chunk 0] → Decode → [chunk 1] → … → [末块] → 首 Token"。一个必须做对的计算细节：Chunk 内 Attention 用**全局因果偏移**——历史长度为 P 时，局部位置 q 只能看见 `k ≤ P + q` 的 Key，不能按局部位置写 `k ≤ q`，否则会偷看或漏看历史。

**两阶段时间线**：回到"算几个位置 × 读多少历史"的底图。baseline（full）的那次长 Prefill 是"一轮并行算 2048 个位置"——算得多、一次做完，但把所有 Decode 轮堵在后面；chunked 把它切成 8 段，每段只算 256 个位置、读已有的 0–1792 份历史 K/V，每段之间正常插入 Decode 轮（每请求算 1 个新位置、读全部历史）。单次停顿从"容纳 2048"降到"容纳 256"，别人等的时间从一整份变成 1/8 份。

**数字**（3080 Ti、Qwen3-0.6B、budget 256、2048 Token 长 Prompt 干扰实验，README「3080 Ti 实测结果」）：最大单轮 Prefill 316.28→65.39 ms（-79.3%），既有请求最大 ITL 374.78→120.60 ms（降 67.8%），峰值显存降 16.8%。代价同样是再分配：长请求 TTFT 增加 141.6%（它自己排队更久了）、输出吞吐降 6.2%、ITL p95 反而升 88.8%。两个负对照很干净：短 Prompt（128 Token）没有可拆的工作，chunked 反而慢 6.4%——收益来自"拆超预算长 Prompt"本身，不是新增状态机；无并行 Decode 场景中没有要保护的 ITL，隔离收益按定义为零。

**算一算**：2048 Token 的 Prompt 在 budget 256 下拆成几块？`2048 ÷ 256 = 8` 轮 Prefill，中间穿插 Decode——这也解释了 TTFT 为何变差：同一份工作切成 8 段排队慢慢推进，但每段只有 1/8 长，别人只能等 1/8。

**重点**：Chunked Prefill 让单请求 Prefill 可拆分，Token Budget 才从软建议变成硬约束；本质是延迟在请求之间的再分配。

**边界与常见误区**：Chunk 切得越小停顿越短，但迭代、Kernel Launch、历史 K/V 读取与 Python 状态管理开销越多——budget 512 实测最大 ITL 降 60.4% 而吞吐只降 0.6%，说明档位要按负载扫；最大 ITL 改善不保证 ITL p95 与吞吐同时改善。[易错] 把它当无条件加速器——两个负对照说明收益的前提是"有要保护的 Decode + 有要拆的长 Prompt"；另外它和 RAG chunking、流式接收 Prompt 完全是两码事。

**产品对应**：vLLM/SGLang 均默认启用 Chunked Prefill；`max_num_batched_tokens` 等参数实际控制的就是每轮 Prefill 的 Chunk 粒度。

**方案视角**：负载里有长文档/Prompt 输入且输出流式交互（多人同时在打字）的业务必须开启；纯批量生成、没有并发交互的离线任务，收益有限还多付轮次开销。现场先问："你们的输入 Prompt 长度分布什么样？输出是不是流式多人并发？"落点指标：既有请求的 ITL max 与长请求的 TTFT p95。

**讲给客户的一句话**：Chunked Prefill 用"首 Token 稍慢一点"，换"其他用户不掉帧"——本质是延迟在用户之间的再分配。

**延伸阅读**：[Redis: Prefill vs Decode Explained](https://redis.io/blog/prefill-vs-decode/)（两阶段分工与互相干扰）· [NVIDIA: Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)（推理优化总览）

**掌握目标**：能画出 prefilling 状态下 Cursor 的推进与 Chunk 间穿插 Decode 的时序（含全局因果偏移 `k ≤ P + q`）；能判断某个负载该不该开 Chunked Prefill。

**深挖入口**：帖子「Chunked Prefill」（9/3）· 教材 p225 · 仓库 chapter_07

### ch08 · Prefix Cache

<!--player:ch08-->

你要问客服模型 8 个问题，每个问题前面都带着同一段 2048 Token 的公司制度说明。第一条请求把 2048 Token 的前缀算下来没有任何问题，第二条、第三条……每一条都要把这段一模一样的前缀重新 Prefill 一遍——算力花在计算"已知答案"上。系统提示、few-shot 示例、合同/代码模板：很多业务所有请求共用同一段长前缀。本章让后面这些请求直接复用第一条算好的 KV Block，不重算。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 前缀命中 | 新请求与已缓存的某条序列开头完全一致，可复用其 KV |
| 链式哈希 | 块哈希绑定父块摘要，同内容不同位置哈希不同 |
| 不可变 Block | 已填满并宣布只读的物理块，可安全多人共享 |
| 私有尾块 | 未填满的末块与 Decode 块不共享，故无需写时复制 |
| 引用计数 | 多少请求正在用某块；归零才允许回收 |
| leaf-LRU | 从链的叶子端开始淘汰最久未用的缓存块 |
| 热请求 | 命中缓存的那次服务调用，TTFT 显著变短 |

**怎么工作（拆到计算步骤级）**：baseline（disabled）每条请求重算完整 Prompt；variant（enabled）把 Prefill 拆成三步：**第一步，哈希匹配最长完整前缀**——把 Prompt 按 Block 切开，沿链式哈希 `SHA256(model_namespace || parent_hash || block_token_ids)` 逐块查找，取最长连续命中段；为什么必须带 parent：相同的 Token 块在不同历史上下文里算出的 K/V 不同，单独哈希每块会误判命中——自检专门验证了"相同第二个 Token 块位于不同历史上下文时不能误命中"，且哈希命中后还会确定性比较父摘要与 Token IDs。**第二步，命中块引用**——命中的块都是已填满并发布为不可变的物理块，新请求把它们的引用加进自己的 Block Table 即可（引用计数 +1），一个字节都不重算；未填满的尾块与 Decode 块始终私有，因此无需 Copy-on-Write。**第三步，只算未命中后缀**——Prefill 只做未命中的 Prompt 后缀；还有一条隐蔽边界：KV 不保存首 Token Logits，所以最多复用 `prompt_length − 1` 个 Token，至少要重算最后一个 Prompt Token 才能拿到完整 Logits。释放侧两本账分开管：请求结束只减请求引用；缓存索引仍持有的块不回空闲池，池满时按固定 LRU 顺序从叶子缓存块开始淘汰，只有缓存所有权已移除且引用计数为零的块才安全复用。

**两阶段时间线**：复用全部发生在 Prefill 阶段——热请求的 Prefill 从"并行算 2048 个位置"缩成"只算未命中的几十个位置"，这正是"算得少"的直接兑现；Decode 完全不变，每步照旧算 1 个新位置、读全部历史。

**数字**（3080 Ti、Qwen3-0.6B、8 请求共享 2048 Token 前缀，README「3080 Ti 实测结果」）：热请求 service TTFT p50 从 384.11 降到 52.64 ms（-86.30%），实际 Prefill Token 16896→2560（-84.85%，命中 14336 Token，即除首条外每条命中全部 2048）。但 Makespan 只降 11.33%——Decode 是大头且分毫不受益；enabled 还为缓存池多保留物理块，峰值显存增加 280 MiB。

**算一算**：为什么 Prefill Token 减 84.85%、TTFT 降 86.3%，Makespan 只降 11.33%？总执行时间 ≈ Prefill + Decode。设 Prefill 占 20%、Decode 占 80%：Prefill 省掉约 85% 只能省总量的 `0.85 × 0.20 ≈ 17%`——再被缓存管理开销摊薄，剩下约 11% 与实测吻合。收益上限由 Decode 占比决定，不是命中率。

**重点**：Prefix Cache 省的是 TTFT 和 Prefill 算力，不省 Decode 一秒；链式哈希保证"相同 Token 在不同上下文"不会被误共享。

**边界与常见误区**：本期不做部分块共享/COW、并发冷请求去重（in-flight dedup）、跨模型或跨进程共享（跨实例见 ch14）；缓存池多占的 280 MiB 受池容量与 PyTorch 分配方式影响，不能当作所有实现的固定开销。[易错] "命不命中决定一切"——命中率高但 Makespan 只小降，收益上限由 Decode 占比与显存代价共同决定。

**产品对应**：vLLM 的 Prefix Caching（按哈希组织缓存块）；SGLang 的 RadixAttention（Radix Tree 组织前缀）；跨进程/跨实例的分层缓存见 LMCache（ch14）。

**方案视角**：判断按两点：业务前缀是否整齐、有多长。统一系统提示/固定模板/RAG 长 context 的业务值得开；用户自由闲聊、前缀散乱的业务命中率低还白占显存。现场先问"system prompt 是否每轮重复、前缀是否整齐"，再用指标验证：实际执行 Prefill Token 相对总请求数的比值。

**讲给客户的一句话**：前缀缓存的价值取决于业务前缀是否整齐——命中省的是首 Token 延迟和 Prefill 算力，Decode 一秒都省不了。

**延伸阅读**：[vLLM PagedAttention (arXiv 2309.06180)](https://arxiv.org/abs/2309.06180)（块共享与引用计数的原始出处）· [Sebastian Raschka: What is a KV cache](https://sebastianraschka.com/faq/docs/kv-cache.html)（KV Cache 问答精讲）

**掌握目标**：能画出前缀 Block 的链式哈希结构与"命中→只算后缀→引用计数→LRU 淘汰"流程；能判断一笔业务的前缀命中率是否值得开启该功能。

**深挖入口**：帖子「Prefix Cache」（9/4）· 教材 p256 · 仓库 chapter_08

### ch09 · FlashAttention

<!--player:ch09-->

想把上下文窗口从 2048 放大到 8192，改完配置直接 OOM——不是 KV Cache 装不下，而是 Attention 这一步先把显存吃穿了。ch02 说过，Decode 的瓶颈在"读"：算得少、读得多，好的 Kernel 就是"快读"。但本章的账先落在 Prefill 上：朴素写法要把 `QK^T` 的整个 Score 矩阵连同 softmax 结果完整写进显存，中间矩阵随序列长度**平方**增长——4096 Token 增量峰值 2576 MiB，8192 直接 OOM。长上下文能不能上，先卡在 Attention 这个算子的写法上。本章用 Triton 亲手写一个：中间矩阵根本不落显存。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Score 矩阵 | Q 与 K 转置相乘的注意力打分矩阵，尺寸随序列平方 |
| 物化 | 把中间计算结果完整写进显存存放 |
| 分块计算 | 一次只装一小段 K/V 进算，算完丢弃搬下一块 |
| Online Softmax | 不等全量打分就边收边修正的流式 softmax |
| m / l / acc | 每行维护的运行最大值、归一化总和、加权累加器 |
| alpha 修正系数 | 出现更大 max 时用 `exp(m − m_new)` 把旧结果换算到新基准 |
| 微基准 | 单独测这个算子的受控实验，区别于完整模型 |
| Triton | 写 GPU Kernel 的高级语言，比手写 CUDA 简单 |

**怎么工作（拆到计算步骤级）**：baseline（eager）显式走 ch02 的注意力三步——出题、配对打分、取内容——但三步之间的 Score 和 Probability 张量按 `[seq, seq]` 尺寸真正写显存再读回来。variant（inference-only Triton 前向）把这套三步改成流式：**第一步装载**——Q 常驻，K/V 按 `BLOCK_M × BLOCK_N` 的 tile 分块从显存搬进片上，算完丢弃；**第二步 tile 打分**——对当前块算局部 Score，做因果 Mask，取行最大值更新 `m_new = max(m, rowmax(scores_tile))`，再用 `alpha = exp(m − m_new)` 把之前所有累加结果换算到新基准；**第三步取内容并累加**——`p = exp(scores_tile − m_new)`，`acc = alpha × acc + p @ value_tile`，`l = alpha × l + rowsum(p)`；**收尾归一**——全部块扫完后 `output = acc / l`。完整 Score 和 Probability 从头到尾不写 GPU 全局显存。两个必须记住的性质：数学结果与 eager 相同（Kernel 数值验证最大绝对误差 0.003906，真实模型 4 Token greedy 输出完全一致）；**FLOPs 仍是二次方**——省的是中间显存和显存搬运量，不是计算量。接入引擎时的细节：Chunked Prefill 必须用全局因果偏移——历史长度为 P 时，局部位置 q 只能读 `k ≤ P + q` 的 Key，不能按局部位置写 `k ≤ q`。

**数字**（3080 Ti、因果 Attention、head_dim 128、batch 1、heads 16 微基准，README 实测）：1024 Token 单步延迟 1.070→0.316 ms（`3.38×`），4096 Token 15.937→4.309 ms（`3.70×`）；增量峰值显存 2576.1 MiB 对 **16.0 MiB**；8192 Token 上 eager OOM、flash 15.79 ms 正常完成。但接入完整引擎后收益被稀释到极限：2048 Prompt + 16 输出 Token 端到端只有 1.08–1.10×（Makespan 1166.86→1075.65 ms）。原因有三：Chunked Prefill 把每轮 Query 限制在 256 个、Attention 只占整体时间一小部分、Projection/MLP/Paged KV 读取与调度继续占大头。**Kernel 加速比 ≠ 模型加速比**，这是全系列最重要的反面教材之一。

**算一算**：为什么增量显存从 2576 MiB 掉到 16 MiB？eager 要物化 `[seq, seq]` 的 Score 矩阵：4096 Token 一个副本就是 `4096 × 4096 × 2B ≈ 32 MiB`，再乘多头、多层与中间副本；flash 只需要几百 KB 的 tile 加每行 3 个状态量——平方级变常数级。8192 时平方项再翻四倍，eager 于是 OOM。注意这账治的是 Prefill 的中间矩阵：Decode 的 Query 只有 1 个位置，Score 只有 1 行，本来就不大。

**重点**：FlashAttention 省的是显存搬运而非计算量；它决定长序列能不能算，端到端收益幅度则由 Attention 占全模型时间的比例决定。

**边界与常见误区**：本期是 inference-only 教学实现：不支持 backward、任意 Mask，也不感知 Paged KV 的分块存储（K/V 仍先 gather 成稠密再进 Kernel），不能代表 PyTorch SDPA 或官方 flash-attn 库的性能。[易错] "接入 FlashAttention 引擎能快几倍"——微基准 3.70× 到端到端只剩 1.08×，瓶颈在别处；把 Kernel 基准宣传成产品加速能力，正是这一章要你反省的场景。

**产品对应**：FlashAttention 官方库（`flash-attn` Kernel）；vLLM/SGLang 内置融合的 FlashAttention/FlashInfer 后端。

**方案视角**：长上下文业务的前置条件——8K+ 上下文没有它基本不可行；但选型时不要信 Kernel 公认数字，要问"Attention 占你们端到端时间的多少"，并用集成后的端到端 TTFT/吞吐验证。短上下文、Decode 重负载的服务，FlashAttention 层面的升级感知很小。

**讲给客户的一句话**：FlashAttention 省的是显存搬运：序列越长收益越大、长上下文能不能开全靠它，但整体加速要看它占总时间的多少。

**延伸阅读**：[FlashAttention 论文 (arXiv 2205.14135)](https://arxiv.org/abs/2205.14135)（Online Softmax 原始出处）· [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（回看注意力三步）

**掌握目标**：能分步说出"装载→tile 打分→取内容累加→归一"的 Online Softmax 流程，并指出 `alpha = exp(m − m_new)` 修正的是什么；能算出给定序列长度下 eager Score 矩阵的字节数，并解释 Kernel 级加速为何在端到端被稀释。

**深挖入口**：帖子「手写 FlashAttention」（9/6）· 教材 p291 · 仓库 chapter_09

### ch10 · CUDA Graph

<!--fig:ch10-->

想象你在指挥一条流水线，每拧一颗螺丝都要喊一声"开始"——工人手速再快，也全耗在等你喊话上。小模型 Decode 就是这条流水线：回到 basics 的两阶段时间线，每步 Decode 只算 1 个新位置、读全部历史 K/V——计算很轻，但 CPU 要逐个提交上千个 Kernel（README 实测单步 1903 次 `cudaLaunchKernel`），GPU 大部分时间在等指令。ch02 说 Decode 瓶颈是带宽，本章暴露的是另一头：Host 提交开销把带宽瓶颈完全盖住了。CUDA Graph 的思路是"把整段指令提前录成一张施工图，之后每次只喊一声：照图重放"。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Kernel 提交 | CPU 把一个计算任务发给 GPU 执行的动作，每次都有固定开销 |
| Host Launch 开销 | CPU 端发起提交所花的时间，与 GPU 实际计算无关 |
| CUDA Graph | 把一整段 Kernel 序列提前录下，之后一次调用整体重放 |
| Capture | 录制阶段：跑一遍程序，把 Kernel 序列存成 Graph |
| Replay | 重放阶段：一次 `cudaGraphLaunch` 执行整张图 |
| 静态 Buffer | 地址和形状固定的预分配显存，Graph 只认固定地址 |
| Context Bucket | 预留的上下文长度档位，按档建 Graph |
| Block Table | 逻辑块→物理块映射表，Replay 前可更新内容 |

**怎么工作（拆到计算步骤级）**：baseline（`static_eager`）与 variant（`cuda_graph`）执行完全相同的 Decode Tensor 程序——相同模型、权重、KV Pool、greedy 设置——唯一变量是提交方式。variant 的流程固定四拍：**第一步静态化**——为每组 `(batch_capacity, context_bucket)` 预建独立 Runner，把 `input_ids`、`sequence_lengths`、固定宽度 Block Table、覆盖整个 Bucket 的物理 KV Block 全部预分配成固定地址的 Tensor；**第二步 Capture**——在所有 Slot inactive 的安全状态下完整跑一遍程序，把 Kernel 序列录成 Graph（真实 Prompt 之后才加载，避免 Capture 消耗正式请求状态）；**第三步更新**——每步 Replay 前改写输入 Token、Block Table 里的物理块 ID 和有效长度：**内容可换，地址和形状不许变**，无效位置靠 `active_mask` 屏蔽；**第四步重放**——一次 `cudaGraphLaunch` 执行整张图，argmax 和下一 Token 回填也搬进 Graph 内，避免每步 `.item()` 同步把 CPU 拽回来。

**数字**（3080 Ti、Qwen3-0.6B、bf16、1 次 warm-up + 3 次重复均值，README 实测）：Profiler 显示 Eager 单步 1903 次 `cudaLaunchKernel`，Graph 提交侧只有 1 次 `cudaGraphLaunch`。capacity 1、128 Token 下 Decode TPOT 37.90→5.22 ms（`7.26×`）；上下文和 Batch 变大后 GPU 计算变重，capacity 4、2048 Token 下只剩 `2.25×`——Graph 不减少矩阵计算和 K/V 读取，能消除的 Host 开销占比自然下降。端到端实验里 TPOT 提升 4.03×、完整请求只提升 2.53×：Prefill 保持 Eager、不受机制影响，把 Decode 收益稀释了。Capture 一次约 95–140 ms。

**算一算**：capacity 1、128 Token 时每步节省 37.90−5.22=32.68 ms，Capture 花 115.82 ms，115.82÷32.68≈3.5 步回本——前提是这组 Graph 能被后续请求复用（生产系统还要算 Graph Cache miss 与驱逐成本，README 明确此估算不含）。

**重点**：**CUDA Graph 消除的是 Host 提交开销，不是计算量；GPU 越忙（Batch 大、上下文长），剩余收益越小。**

**边界与常见误区**：可捕获性要用静态换——Bucket 尾部存在无效 K/V 读取和 Mask 计算，请求准入、Block 分配、取消仍在 Graph 外处理。[易错] "静态化导致显存暴涨"——实测 Graph 稳态峰值反而略低于 Eager（复用 Capture 期内存）；真正的显存代价是 KV Pool 必须按 Bucket 整段预留，随 Capacity 和 Bucket 线性增大（本实验 19.2→917.0 MiB）。加速比也不能外推到更大模型或动态流量。

**产品对应**：vLLM、SGLang 均对 Decode 路径启用 CUDA Graph，按 batch size 与上下文档位缓存多套 Graph 实例，命中档位才 Replay `[业界通识]`。

**方案视角**：小模型、低并发的服务 Decode 粒度细、Host 开销占比高，Graph 收益最大；大模型高并发下 GPU 本就吃饱，Graph 是标配但增益有限。现场先问：你们的加速比是 Decode TPOT 还是端到端？让客户把 TPOT 提升倍数和端到端提升倍数并排看——差距就是 Prefill 等不受机制影响部分的稀释。

**讲给客户的一句话**：CUDA Graph 解决的是"CPU 指挥慢于 GPU 计算"的问题；Batch 和上下文越大、GPU 越忙，剩余收益越小。

**延伸阅读**：[NVIDIA: Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)（推理优化总览中的 Decode 优化视角）

**掌握目标**：能画出"静态化→Capture→更新→Replay"四拍流程，并标出哪些状态必须静态化、哪些可在 Replay 前更新；能算出给定 Capture 时间与每步节省量时的回本步数。

**深挖入口**：帖子「CUDA Graph」（9/8）· 教材 p324 · 仓库 chapter_10

### ch11 · Greedy Speculative Decoding

<!--fig:ch11-->

你请教授写作文，但他每写一个字都亲自提笔——其实很多字是套路，可以让助教先猜几个候选，教授只做圈阅。Speculative Decoding 就是这个思路：小模型（Draft）连猜候选 Token，大模型（Target）一次批验证。听上去稳赚，本章实测却几乎全是负优化——这个反转比机制本身更值得记住。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Draft Model | 提候选的小模型，便宜但不保证和大模型答案一致 |
| Target Model | 做最终裁定的大模型，验证一批候选只跑一次前向 |
| Draft Length γ | 每轮让小模型连猜的候选个数 |
| 接受率 | 候选里与小模型外答案相符的比例，收益的决定因素 |
| 最长匹配前缀 | 一批候选里从头连续被接受的那段 |
| Correction Token | 首个被拒绝处由 Target 给出的纠正 Token |
| Bonus Token | 候选全对时 Target 顺带多出的下一个 Token |
| KV 回滚 | 候选被拒时把两套 KV Cache 截断回接受边界 |
| 块验证 | Target 对 γ 个候选一次并行前向打分 |

**为什么需要它（用两阶段时间线看 Decode 的串行问题）**：basics 的时间线说，Prefill 一次算多个位置所以快，Decode 每步只算 1 个位置——第 t 个字不算出来，第 t+1 个没资格开始，这是自回归的硬依赖，baseline 里大模型被**串行**调用 N 次。但还有一个被浪费的并行性：模型一次前向本来就能算**多个位置**的 Logits（Prefill 正是这么干的，"算得多"）。推测解码就是把这个并行性借回 Decode：Draft 先猜一段，Target 一次前向验证一整段——验证 γ 个候选是一次批量前向，像一次"小 Prefill：算 γ+1 个位置、读全部历史"，成本远低于 γ 次串行生成。只要接受率够高、Draft 足够便宜，每轮就能白拿多个 Token。本章实现 Greedy 版本：小模型猜的与大模型 greedy 一致才接受，数学上输出与只跑大模型完全相同。

**怎么工作（拆到计算步骤级）**：baseline（Target-only）逐 Token 串行生成。variant 每轮四拍：**第一步提候选**——Qwen3-0.6B 串行连猜 γ 个候选（一轮共 γ+1 次 Draft 前向，每个候选都要接回末尾才能猜下一个）；**第二步块验证**——Qwen3-4B 对 γ+1 个位置一次并行前向打分；**第三步裁定**——接受最长匹配前缀，拒绝处补 `Correction Token`，全对再加 `Bonus Token`，一轮最多产出 γ+1 个 Token；**第四步对账**——两套 KV Cache 按同一个接受边界提交或截断回滚，否则 Target 和 Draft 的 Cache 会漂移。正确性实验（FP16）显示输出与 Target-only 逐 Token 一致；但 BF16 下块验证与单 Token 前向选了不同 argmax——数学等价不等于任意 dtype 下的逐位保证，主实验因此固定 FP16。

**数字**（3080 Ti、FP16、Batch 1、生成 32 Token、EOS disabled、3 次重复均值，README 实测）：中文解释负载接受率最低，γ=8 时仅 0.173，TPOT 加速比 0.284×——越猜越错还越贵；只有接受率 1.000 的代码补全负载在 γ=8 拿到 1.023×，勉强回本。根因在组件成本：Draft 单步 42.51 ms 只比 Target 的 48.16 ms 快约 12%——参数量差 6.7 倍，但 Eager 教学栈里两者都执行大量小算子、都被 Host Launch 拖住（这正是 ch10 要治的病）；一轮还要多付 γ+1 次 Draft 前向，块验证省下的 Target 调用覆盖不了。

**算一算**：一轮 γ=8 的期望成本 = 8×42.51（Draft）+ 约 57（块验证），约 397 ms；期望产出 = 8×0.173+1≈2.4 个 Token，即约 165 ms/Token——比 Target 直接生成的 48.16 ms/Token 贵 3 倍多，接受率低时必亏。

**重点**：**投机解码是"用便宜计算赌接受率"；Draft 不够快或接受率不够高时是纯亏，且每轮都要多付 γ+1 次 Draft 前向。**

**边界与常见误区**：只实现 Greedy 路径（不含随机 Speculative Sampling）、Batch 1、无 MTP/EAGLE/Medusa；结果不能外推到 32B/70B 或融合执行栈。[易错] "候选越多收益越高"——γ 越大错得越远、Draft 成本线性涨；接受率 0.357–0.833 的组别在所有 γ 下加速比都低于 1。先看组件成本比，再谈接受率。

**产品对应**：vLLM/SGLang 内置 speculative decoding，支持 n-gram、draft model、MTP、EAGLE 等候选来源 `[业界通识]`；MTP 等更轻量 Speculator 是本机制成立的前提，但不是本章自变量。

**方案视角**：适合输出重复性强（代码、模板）、接受率高，且候选生成器明显比 Target 快一个量级（原生 MTP、融合 Kernel）的负载。现场先问：业务 Prompt 的输出重复性如何？候选生成器单步比 Target 快多少？——接受率和最佳 γ 都要按业务负载实测，不能凭"理论上快"上会。

**讲给客户的一句话**：投机解码是"用便宜计算赌接受率"，Draft 不够快或接受率低时是纯亏——上线前必须按业务负载实测。

**延伸阅读**：[NVIDIA: Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)（总览中的 Speculative Decoding 一节）

**掌握目标**：能画出"提候选→块验证→接受/补正→KV 对账"四拍流程，并说清块验证为什么像一次小 Prefill；能根据 Draft 单步成本与接受率算出期望每 Token 成本，判断是否值得上线。

**深挖入口**：帖子「Speculative Decoding」（9/10）· 教材 p362 · 仓库 chapter_11

### ch12 · KV Cache Offloading

<!--player:ch12-->

餐厅满座时来了新客人：要么门口挂"客满"牌子（拒绝请求），要么请一位吃得最慢的客人暂时去休息区，让出座位。推理引擎显存装不下所有并发请求的 KV 时也面临同样选择——被请出去的客人，随身行李是寄存（swap 到 CPU）还是扔了、回来重吃一遍（recompute）？这就是本章对比的两条抢占恢复路径。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 抢占 | 显存不足时暂停一个运行中请求，腾出它的 KV Block |
| 受害请求 | 被选中暂停的请求，默认是最后接纳的那个 |
| swap | 整请求 KV 经 PCIe 换出到 Pinned CPU 池，恢复时换回 |
| recompute | 直接丢弃 KV，恢复时对全部已有内容重新 Prefill |
| Pinned Memory | 锁页内存，CPU 不换页，可跑满 PCIe 带宽 |
| 乐观准入 | 先放请求进池，不够了再抢占，把池始终打满 |
| 保守准入 | 预留 prompt+最大输出，绝不抢占，池用不满 |
| Makespan | 一批请求从到达到全部完成的总时长 |
| ITL | 相邻两个输出 Token 的时间间隔 |

它是什么、为什么需要：显存装不下所有并发 KV 时，悲观预留把池"用不满"，拒绝请求又损失体验。乐观准入 + 抢占给出第三条路：池满了就让晚来的请求挤掉最后接纳的 running 请求，把"显存不足=失败"变成"慢一点但能跑"。换出单位是整请求：回到 basics 时间线——Decode 每步只算 1 个新位置，却要读全部历史 K/V，历史 K/V 一刻不能缺，"部分换出"不存在。

**怎么工作**（拆到步骤级）。两条路径共用相同的池容量、准入水位、受害者选择（最后接纳的 running 请求）和恢复顺序（swapped 队列 FCFS 优先于 waiting），只在"恢复"一步分岔：

- **swap 三拍**：①换出——把受害者整请求 KV 按块主序拼成 contiguous 数据，经 PCIe 拷进预分配的 Pinned CPU 池（连续布局保证一个物理块一次拷完）；②断点存档——CPU 侧记下该请求的 block 表，这就是"断点"；③恢复搬回——轮到它时按 block 表把 KV 拷回 GPU 新块、重建映射，从暂停处的下一个 Decode 步续算。
- **recompute 两拍**：①丢——受害请求的 KV Block 直接释放回池；②重 Prefill——恢复时把全部已有内容当成新 Prompt，并行重算所有位置、重建 Cache，再续接 Decode。

一个前置细节：Pinned Pool 必须预分配——现场申请 448 MiB pinned 要约 102 ms，比一次换出还贵，所以池在启动时建好。

**数字**（3080 Ti、PCIe Gen3 x16、Qwen3-0.6B、bf16、8 请求固定负载，README chapter_12 实测）：swap 的换出+换入合计约 49–63 ms（约 11.9 GB/s）；recompute 的额外 Prefill 暴露约 673–764 ms，重算明显贵于搬运。容量收益成立：swap 吞吐 37.0 vs 保守准入 35.7 token/s（池 230）、44.9 vs 40.7（池 320），逻辑并发从 2–3 升到 4–5；但都低于 664 块大池的 45.5——池充足时抢占收益为零。

**算一算**：Qwen3-0.6B 单 Token KV = 28 层 × 8 KV 头 × head_dim 128 × K/V 两份 × 2 字节 ≈ 112 KiB，Block Size 16 时每块 1.75 MiB；一个 2048 Token 的请求占 128 块 ≈ 224 MiB，按 11.9 GB/s 换出约需 19.7 ms——与 README 实测 2048 Token 换出 19.67 ms 吻合。

**重点**：**抢占把显存不足从失败语义变成延迟语义；代价由被抢占请求的秒级暂停承担，恢复方式的选择取决于带宽成本与重算成本的相对值。**

**边界与常见误区**：换出副本只用于恢复原请求，不做命中或复用（复用是 ch14）；换出不提升单请求上下文上限；拷贝在默认 Stream 上同步进行，期间 GPU 不推进其他工作（异步化是 ch13）。[易错] "swap 一定优于 recompute"——端到端受负载支配：池 230 中 recompute 吞吐 37.6 略胜 swap 37.0；也别把秒级 ITL max（约等于受害者暂停时长）归因于几十毫秒的 PCIe 拷贝。

**产品对应**：vLLM 的抢占式调度与 CPU Offload（swap-based preemption）`[业界通识]`；LMCache 等做 KV Cache 分层存储，其"复用"超出本章只换不复用的边界 `[业界通识]`。

**方案视角**：适合"不能拒绝请求、可容忍偶发秒级慢回复"的客户。现场先问：恢复延迟和重算算力哪个贵？判据落在两条曲线——KV 传输时间（≈上下文字节 ÷ PCIe 带宽）对重新 Prefill 时间（随上下文近线性、随模型变贵）：长上下文、大模型、PCIe 宽的平台偏 swap；短上下文、带宽差时 recompute 未必输。再问 SLA 对秒级尖刺的容忍度——本实验暂停 p95 达 4–10 s。

**讲给客户的一句话**：换出把"显存不够就失败"变成"慢一点但能跑"，代价由被抢占的请求承担，走搬运还是重算要按上下文长度和算力价格算。

**延伸阅读**：[Sebastian Raschka: What is a KV cache](https://sebastianraschka.com/faq/docs/kv-cache.html)（KV 到底存了什么、为什么换出的是它）· [NVIDIA: Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)（推理优化总览中的显存管理位置）

**掌握目标**：能画出抢占→暂停→换出/丢弃→恢复的两条路径并标出成本；能按传输时间与重算 Prefill 时间判断该选 swap 还是 recompute。

**深挖入口**：帖子「Offloading」（9/12）· 教材 p381 · 仓库 chapter_12

### ch13 · KV Cache 异步传输

<!--fig:ch13-->

上章的换出有个尾巴：拷贝发生时，GPU 亲自去搬 KV——搬那几十毫秒里自己在"停工"。可你想想快递站：收银员不必亲手把包裹扔上卡车，收完客人的钱继续服务下一位，扔包裹是装卸工的事。GPU 上也有这样一条"装卸通道"。本章只改一个变量：KV 在 GPU 和 Pinned CPU 池之间的搬运，从"计算流上同步搬"换成"独立通道上异步搬"，看显存压力下的调度器能不能趁机干点别的。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Copy Stream | 独立的拷贝通道，与计算流各走各的队列 |
| CUDA Event | 流上的"完成回执"，可查询传输做完没有 |
| Block 所有权 | 某块 KV 此刻归谁管的规则，未转移前不许动 |
| `query()` 轮询 | 只查 EVENT 状态、不原地干等的非阻塞问询 |
| 暴露等待 | 调度器被迫停下等传输的时间，优化的真正目标 |
| 容量压力 | 池不够导致抢占，传输与计算竞争资源的场景 |

它是什么、为什么需要：回到 basics 时间线——Decode 每步算 1 个新位置、读全部历史 K/V，而 ch12 的同步换出恰好卡在"读"上：拷贝排在计算流队列里，前后的 Decode 都排队等它。异步传输不减少搬运字节，只把"搬"从计算队列里摘出去，让"算"和"搬"并行。

**怎么工作**（拆到步骤级）。variant（async）每次换出/换入走三步：

1. **提交**：D2H/H2D 逐块提交到独立 Copy Stream，计算流立即返回——调度器马上安排下一个 Prefill 或 Decode。
2. **Event 同步**：在 Copy Stream 上记一个 CUDA Event 当"完成回执"，传输推进到它即算完成。
3. **所有权转移**：正确性全靠 Event 管块所有权——D2H 完成前 GPU 源块不能释放/改写；H2D 完成前目标块不能读、CPU 源块不能丢。状态机因此多了三个中间态：`running → swapping_out → swapped → swapping_in → running`。

调度层诀窍在"不白等"：`Event.query()` 只轮询回执状态、不原地干等；只有确实没有可执行工作、或下次 Decode 必须申请 Block 时，才去等对应 Event——有别的 Prefill/Decode 可推进就先推进。对照 baseline（sync）在默认计算流上逐块拷贝，拷完调度器才继续，期间 GPU 无任何推进。

**数字**（3080 Ti、PCIe Gen3 x16、Qwen3-0.6B、bf16、Block Size 16，README chapter_13 实测）：受控微基准证明本机确有重叠能力——传输与 BF16 GEMM 并发后，16 Block 一次 D2H 从串行 6.69 ms 压到 4.64 ms（1.44×），各方向加速 1.21–1.50×。但主实验不是单向正优化：池 230 组 async 略慢（39.31 vs sync 39.61 token/s），池 320 组较快（47.62 vs 45.55）。机制指标更干净：async 把暴露等待从 48.3→20.0 ms（池 230）、62.2→19.2 ms（池 320）——等待确实被藏掉了，端到端却还有别的东西在波动。

**算一算**：池 230 组 async 的设备传输 44.8 ms、暴露等待 20.0 ms，即约 55% 的传输时间已被独立计算覆盖；但 Makespan 差值看不出来（38.79 vs 39.07 s）——几十毫秒的传输窗口决定不了几十秒批次里真正的瓶颈。

**重点**：**异步传输的正确性靠 Event 管块所有权，不靠提交顺序；能赚多少取决于拷贝时有没有独立计算可做——判断优化用"暴露等待"，不是设备总传输时间。**

**边界与常见误区**：D2H 在紧急 Block 水位触发，完成前源块不能释放，重叠窗口可能大部分暴露；H2D 完成前占着 GPU Block 和运行名额，有别的请求时才可能被藏住；不实现 SSD、RDMA、传输优先级或预测性换出。[易错] "开了异步端到端就更快"——池 230 组反而略慢；Makespan 差异远大于传输窗口且波动明显，别把端到端差值全归因于隐藏 PCIe 传输；"异步提交"也不等于"传输已完成"。

**产品对应**：vLLM/SGLang 的 offloading 流水线、NIXL 等 KV 传输层设计中的异步 Stream 使用 `[业界通识]`。

**方案视角**：需要它的是"抢占频繁、且同一时刻总有别的请求可推进"的满载服务；不需要的是池充足、抢占为零的服务——大池无抢占负对照下收益本来就是零。现场先问"池还紧不紧、恢复时卡上有没有别的活在跑"，判据落到"暴露等待时间"，而不是"异步加速 1.4×"。

**讲给客户的一句话**：异步传输用独立通道和 Event 管住块所有权，卖出的是"搬运期间计算不停"；但要看它是不是真的帮你——请拿暴露等待时间验收，不是拿"开了异步"验收。

**延伸阅读**：[NVIDIA: Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)（推理优化总览）· [Redis: Prefill vs Decode Explained](https://redis.io/blog/prefill-vs-decode/)（两阶段各自瓶颈，理解"搬"能藏在哪一段）

**掌握目标**：能画出 Copy Stream 与 Event 化的块所有权转移时序（`swapping_out→swapped→swapping_in` 并标出"谁在什么时候允许动块"）；能区分"暴露等待"与"总传输时间"两个指标并答出为何前者才是优化目标。

**深挖入口**：帖子「异步传输」（9/14）· 教材 p406 · 仓库 chapter_13

### ch14 · 外部 KV Cache 与 LMCache

<!--fig:ch14-->

你重启了推理服务，客户那些三千字系统提示词的 KV 全部蒸发——接下来每个请求又从零 Prefill 一遍；隔壁实例也在重算同一份提示。问题是：KV 只活在单进程显存里，没身份证、出不去。给它一个跨进程都认的"身份证"，换台机器也能 Lookup 一下、加载过来、只算个尾巴——这就是本章。LMCache 之所以存在，就是为了这件事。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| Token Chunk | 以 256 个连续 Token 为单位存放的外部 KV 块 |
| CacheKey | 链式哈希：模型 + 父块摘要 + 本块 Token，当身份证 |
| Namespace | 打包模型/revision/dtype/拓扑等参数的隔离标记 |
| Cache Service | 独立进程，只管存不可变 KV 字节与索引淘汰 |
| Lookup/Load/Store | 查索引、取 Payload、写回，三步远端访问 |
| leaf-LRU | 只淘汰没有后继的叶子块，避免留孤儿前缀 |
| warm 命中 | 新进程查到旧 KV 并成功加载的状态 |

它是什么、为什么需要：KV 的价值全在"复用"，前提是"活着且找得到"。ch12 的换出副本只伺候原请求、用完即弃；这里要把 KV 变成跨进程资产：谁需要同一份前缀，谁来取——包括重启后的新进程。

**怎么工作**（拆到步骤级）。variant（external）把一次 warm 启动拆成三步：

1. **查最长连续外部块**：新 Prompt 切成 256 Token 的 Chunk，逐块用链式 CacheKey `SHA256(model namespace || parent chunk digest || chunk token ids)` 向独立 Cache Service 发 Lookup；从第 0 块数到第一个未命中的 Chunk 为止，命中前缀必须"最长连续"——中间断一块，后面全不算，缺任何一段 K/V，后面位置的注意力都配不上对。
2. **搬入本地块**：对命中部分发 Load，把远端 CPU KV 重建到新引擎申请的本地 GPU Block，重建本地 Block Table（外部对象不含本地 Block ID，映射必须重造）。
3. **补算末 Token**：外部 Chunk 只存到最后一个 Prompt Token 之前——KV 里没有首 Token 的 Logits，所以即便 512 个全命中，也要对最后 1 个 Prompt Token 做一次 Prefill（时间线上"算 1 个位置"那步）才能吐出回应；未命中的后缀再正常 Prefill。

协议细节两处：服务端按实际字节数限容，Store 校验通过、父 Chunk 在场才发布子 Chunk（子块 CacheKey 链着父块摘要，父丢子全作废，故淘汰用 leaf-LRU 只删叶子）；Lookup/Load 失败或校验不过，引擎记错误并安全回退完整重算——缓存坏了不能影响正确性。

**数字**（3080 Ti、Qwen3-0.6B、bf16、GPU/External Chunk 16/256 Token，README chapter_14 实测）：正确性端到端过关——两个独立进程，Producer 存完退出，Consumer 仍命中 512 Token、warm 侧只 Prefill 1 Token，输出与重算逐位一致。性能端却是全负优化：2048 Token 前缀 warm TTFT 2131.68 ms vs 重算 480.62 ms（0.23×）；Lookup 始终约 1 ms，瓶颈在 TCP Payload 传输 1831.88 ms 与逐 Chunk 重建，2048 Token 的 KV Payload 达 234,881,024 Bytes（约 235 MB）。

**算一算**：235 MB ÷ 1.83 s ≈ 128 MB/s 有效带宽，而重算 480 ms 里那次 Prefill 只动权重、带宽压力小得多——当"搬进来的字节多到超过链路能力"时，命中的贵过重算的，缓存就成了负资产。

**重点**：**给 KV 发跨进程身份证（Token Chunk + CacheKey）是架构上的关键一步；但缓存是否有价值取决于传输路径总成本是否低于重算——"已经命中"不等于"值得加载"。**

**边界与常见误区**：本期固定同步 TCP 加 SHA-256 校验的搬运，路径存在用户态复制，不是共享内存、CUDA IPC、GPUDirect 或 RDMA；Cache Service 无认证/多租户且退出即丢数据，不做多个冷请求的并行去重。[易错] "命中了就赚"——当传输成本超过重算成本，命中率越高亏得越多；负优化不是机制错误，是传输路径不达标，别把本教学节点的瓶颈外推到 LMCache 的高性能 Connector 或更大模型。

**产品对应**：LMCache 逐块对应本期（ExternalCacheClient↔Connector、Token Chunker↔Token Index、不可变 KV Payload↔Memory Object、CacheStore↔CPU Backend+LRU）；vLLM 通过 KV Connector 接 LMCache 作为外挂存储 `[业界通识]`。

**方案视角**：需要它的是多实例部署、频繁重启/滚动升级、请求间共享长系统提示或文档前缀的业务；不需要的是单实例、短上下文、前缀差异大的闲聊型业务。现场先问三个问题落到三个数字上："同一前缀会被打多少次（命中率）？Cache 放在哪层、加载走什么链路？搬一份前缀多少字节、加载 vs 重算的临界点在哪？"

**讲给客户的一句话**：KV 拿到跨进程身份证后，"重启不丢、实例之间能带飞"才有戏；但请务必算过"加载 vs 重算"这笔账再上——慢链路会把缓存做成越用越亏的负资产。

**延伸阅读**：[Sebastian Raschka: What is a KV cache](https://sebastianraschka.com/faq/docs/kv-cache.html)（复用的对象到底是什么）· [vLLM PagedAttention (arXiv 2309.06180)](https://arxiv.org/abs/2309.06180)（KV 按块管理、按块寻址的源头）

**掌握目标**：能画出 Lookup→Load→重建 Block Table→重算末 Token 的完整流程（含失败回退路径）；能按 Payload 字节数与链路带宽算出"加载 vs 重算"的成本临界点。

**深挖入口**：帖子「LMCache」（9/16）· 教材 p424 · 仓库 chapter_14

### ch15 · Prefill/Decode 分离

<!--player:ch15-->

你让餐厅里唯一的服务员既负责带新客从头念完十页菜单（Prefill，计算密集），又要随时给老客续杯（Decode，访存密集）——两种节奏不一样的活挤在一个人身上，互相拖累，想多雇人也说不清该按哪种活配大厨还是配传菜工。把两种角色拆开，就要有人把"旧客已经念到哪"交接下去：这份交接物就是请求的 KV，交接协议就是本章的 Handoff。

**名词卡**：

| 名词 | 白话解释 |
|---|---|
| 5 种角色 | Prefill Worker、Decode Worker、Handoff Service 三进程 |
| KV Handoff | 一份请求的 Prompt KV 从生产者交到消费者的过程 |
| 逻辑导出 | 按连续布局导出 KV，不传进程内物理 Block ID |
| ACK | 接收方"校验通过、装进显存了"的确认回执 |
| ACK-gated | 首 Token 等 ACK 之后才允许发给客户端 |
| fallback | 载荷损坏/不匹配时退回完整重算的保底路径 |

它是什么、为什么需要：basics 时间线里，Prefill 是"一次算完几十上百个位置"的并行大活，Decode 是"每步算 1 个新位置、读全部历史 K/V"的串行细活——一个吃算力，一个吃带宽。挤在同一张卡上，长 Prompt 的 Prefill 会瞬间占满算力，同卡 Decode 的 ITL 出现尖刺；想扩容也不知该按哪种负载买卡。拆开各自扩容，代价是每个请求都要经历一次"搬家"。

**怎么工作**（拆到步骤级）。variant（disaggregated）按 `Request ID + Attempt ID` 做一次性交接，分三段：

1. **Prefill worker 全程**：接单 Worker 只做 Prefill——并行算完整个 Prompt 的所有位置（时间线上"算得多"那步）、建好 KV、顺手产出首 Token，随后它对这个请求的任务结束。
2. **KV+首 Token 过同步通道**：把逻辑导出的 KV Payload 走 `GPU→CPU→TCP→CPU→GPU` 同步发给 Handoff Service 再到 Decode Worker；Payload 用"逻辑布局"（连续、按层），不暴露进程内物理 Block ID——物理块映射是各进程私有的，传过去也对不上。
3. **Decode worker 接管**：Decode 校验、分配本地 Block、H2D 导入后回 ACK；Producer 收到 ACK 才释放本地 KV——在那之前谁都不能动这份 KV。首 Token 也在 ACK 之后才允许外发（ACK-gated）：客户端收到首字时，接手方已确实装好 KV，后续 Decode 不断档。

故障路径：Payload 损坏或 Namespace 不匹配时，Decode 发 fallback ACK，双方再用原始 Prompt 完整重算——先保正确，再谈速度。注意它和 ch14 的差别：Handoff 不做前缀 Lookup、不保留副本、只有一次交接，Payload 用完即释放。

**数字**（3080 Ti、Qwen3-0.6B、bf16、ACK-gated split TTFT、2048 Token Prompt，README chapter_15 实测）：正确性三 Worker PID 互异、Logits 摘要与输出逐 Token 一致；但单卡串行路径是明确负优化——KV Payload 224 MiB，mono TTFT 433.39 ms vs split 2378.55 ms，5.49×。阶段中位数里，Export 约 433 ms、Publish 约 896 ms、Receive 约 358 ms、Import 约 253 ms——交接本身的搬运比 Prefill 本身还贵。端到端也别指望回本：mono 685.59 vs split 2648.27 ms。换成 1.7B 压力模型（同 28 层、8 KV Head，KV 字节不变），split/mono 仍 5.23×，负优化的根不在"模型太小"。

**算一算**：224 MiB 走两次 socket 传递，Publish + Receive ≈ 1254 ms，纯搬 KV 就吃掉约 1.25 秒——monolithic 里这份 KV 根本不用离开进程；这就是"角色分离要付出数据搬家税"的具体单价。

**重点**：**P/D 分离卖的是"角色拆开后各自扩容、互不干扰"，不是单请求更快；单请求路径上，它是比 monolithic 多一次昂贵 KV 交接的负优化。**

**边界与常见误区**：两个 Worker 共一张卡，没有真正的 GPU 隔离；Handoff Service 无认证/多租户/超时转移，Payload 走同步 CPU Bytes 和 TCP。[易错] "上 P/D 分离就加速"——本期所有长度全负优化；多 GPU 的并行吞吐收益不在范围，别拿本章数字论证产品效果。

**产品对应**：vLLM disaggregated prefill、DistServe、Mooncake 等多卡/多机 P/D 分离方案 `[业界通识]`；KV-aware 调度是其配套能力，超出本章范围 `[业界通识]`。

**方案视角**：需要它的是并发大到 Prefill 与 Decode 互相挤压、要按两种瓶颈独立配 GPU 数量的服务；不要的是单实例、低并发的单请求场景——拆开只会多一次搬家税。现场先问"规模是否大到有收益（并发量级、上下文长度、当前卡上两阶段是否互相拖）"，判据落到"两种角色的高峰并发量与资源占比"，而不是"新架构一定快"。

**讲给客户的一句话**：P/D 分离是给"大了"之后做角色分工与独立扩容用的，不会让单个请求变快——你先有多大的并发，再谈拆不拆。

**延伸阅读**：[NVIDIA: Mastering LLM Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)（推理优化总览，P/D 分离在其中的位置）· [Redis: Prefill vs Decode Explained](https://redis.io/blog/prefill-vs-decode/)（两阶段为什么该分开看）

**掌握目标**：能画出 ACK-gated 的 Handoff 时序（Publish/Receive/Import/ACK/双方释放）并标出六段组成 `split TTFT` 的每一项；能判断某客户的业务何时该考虑 P/D 分离、何时不该。

**深挖入口**：帖子「PD分离」（9/18）· 教材 p453 · 仓库 chapter_15
