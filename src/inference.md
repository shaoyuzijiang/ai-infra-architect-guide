# 推理篇（chapter_01–chapter_15）

> 事实来源：`miniinfra/chapter_NN/README.md`（唯一变量、实测表、可推广/不可推广结论）。教案不复制原版，只讲机制、产品对应与方案判断。

### ch01 · 手写 Qwen3 推理

**一句话定位**：不用任何推理框架，手写 Qwen3-0.6B 的 Tokenizer 与单请求自回归生成。

**解决什么问题**：一切推理加速的话术都建立在一个前提上——知道"大模型推理到底在算什么"。上万人同时用着传播的加速比时，谁都说不清一条请求从 Prompt 进来到文本出来经历了什么。本章把这件事拆到最底层：只用 Python 和 PyTorch 跑通生成循环，为全系列建立 baseline。

**原理图解**：<!--fig:ch01-->

**原理要点**：
- baseline 是"裸生成"：读取配置与 Safetensors 权重 → 手写 Byte-level BPE 与 non-thinking Chat Template → 逐层手写 Qwen3 前向 → greedy decoding 逐 Token 生成；本期唯一变量是"能否自己跑通"，不引入任何优化。
- 对照物是 Transformers（只出现在对照程序里，不进推理路径）：相同输入、greedy 条件下，三组 Prompt 的生成 Token 逐个一致，Logits 最大误差均为 0。
- 成本侧裸露问题：Prefill 与 Decode 是两种延迟结构——19 输入 + 32 输出在 3080 Ti 上 Prefill 34.43 ms，平均 Decode 33.40 ms/token，端到端模型计算 1069.80 ms，峰值显存 1173.80 MiB。
- 每个 Decode 步都重新计算完整上下文（无 KV Cache）：历史越长浪费越大，这正是 ch02 的第一个优化目标。

**重点**：**推理循环 = Prefill 一次算完整段 Prompt，再逐 Token 自回归 Decode；而 Decode 每步重算全部历史的浪费，是所有推理优化的起点。**

**边界与常见误区**：仅支持固定 revision 的 0.6B dense 模型、batch size 1、greedy、关闭 thinking 的纯文本 system/user 模板，不支持工具调用与多模态。[易错] "手写实现不如 Transformers 准"——同为对照路径下两者逐 Token 一致，Transformers 只是正确性参照；且单次测量不能外推。

**产品对应**：任何推理引擎的最小可运行内核 `[业界通识]`；Transformers/HF 在本章只作正确性对照，不进入执行路径。

**方案视角**：评估任何推理优化方案之前，先要一份可复现 baseline。向供应商问两个问题："你们的加速比以什么为分母？""在什么模型、硬件、dtype 上测得？"没有 baseline 的加速比无从审计。

**讲给客户的一句话**：谈加速之前，先建立一份可复现的 baseline——否则"优化"无从谈起。

**掌握目标**：能画出单请求自回归生成的循环（Prefill + 逐步 Decode）；能答出为什么 Decode 每步成本随历史长度增长。

**深挖入口**：帖子「手写Qwen3推理」（8/24）· 教材 p1 · 仓库 chapter_01

### ch02 · KV Cache

**一句话定位**：把每层历史 K/V 存下来复用，Decode 只算新 Token——第一个推理优化。

**解决什么问题**：上万人同时用聊天模型时，每次生成一个字都要把整段历史重跑一遍模型层——上下文越长浪费越大。KV Cache 是对这个浪费的直接回应：历史 Token 的 Key/Value 算过一次就够，存下来给后面所有步复用。

**原理图解**：<!--fig:ch02-->

**原理要点**：
- baseline 每 Decode 步对完整上下文重跑所有模型层；variant 在 Prefill 时保留每层历史 K/V，此后每步只算新 Token 的 Q/K/V 并追加进 Cache——机制差异只在"历史是否重算"。
- Cache 是普通 Tensor，每层一对 `[batch, num_kv_heads, seq_len, head_dim]`，用 `torch.cat` 增长；Key 在 QK Norm 与 RoPE 之后写入，位置编号从 Cache 长度续算。
- 数据量公式 `2 × layers × num_kv_heads × head_dim × seq_len × batch × dtype_bytes` 线性增长，1024 Token Prompt 实测 115.39 MiB，与公式一致。
- 实测揭示关键边界：16 Token 短 Prompt 上 KV Cache 反而**负优化**（端到端 1191 vs 1012 ms）——Qwen3-0.6B 单步算量极小，Python 调度、Kernel Launch 与 `torch.cat` 再分配把省下的计算吃掉了；到 1024 Token 才有约 `1.85×` 加速（Decode 均值 60.43→32.64 ms）。

**重点**：**KV Cache 只优化 Decode、不优化 Prefill；收益随上下文变长才显现——短上下文上管理开销可能反超节省的计算。**

**边界与常见误区**：KV Cache 不减少 Prefill 计算；Query 仍要读全部历史 K/V，Decode 不是与上下文长度无关的常数复杂度；`torch.cat` 动态增长的教学实现远谈不上高效。[易错] "开了 KV Cache 一定更快"——16 Token 组实测更慢，结论必须按负载给。

**产品对应**：所有推理引擎的核心组件；vLLM 等引擎的 Cache 预分配与分页管理（ch05 起）远优于动态 `torch.cat`。

**方案视角**：判断按 Token 计价或访谈时上下文收益的问题：客户业务 Prompt 通常多长？上下文越长，KV Cache 与其显存管理越关键；全是几十 Token 的短对话，重点反而落在调度与其他开销上。

**讲给客户的一句话**：KV Cache 的收益随上下文变长才显现，短 Prompt 小模型上管理开销可能吃掉全部节省。

**掌握目标**：能算出给定模型层数/KV 头数/head_dim/长度下 KV Cache 的字节数；能判断"开了 KV Cache 是否更快"取决于什么条件。

**深挖入口**：帖子「kv cache」（8/25）· 教材 p45 · 仓库 chapter_02

### ch03 · Static Batching

**一句话定位**：把多个请求组成成员固定的 Batch 共享一次前向，提高 GPU 并行度。

**解决什么问题**：单请求跑模型 GPU 利用率很低——矩阵太小。把多个请求拼进同一个 Batch、共享一次模型前向，是最朴素的"凑大批"。但请求长短不齐时，padding 和"完成不能退出"会让省下来的算力又漏回去。

**原理图解**：<!--fig:ch03-->

**原理要点**：
- baseline 是多个带 KV Cache 的请求逐个串行执行；variant 组成成员固定的 Batch：异长 Prompt 左 Padding + Attention Mask，一次 Batched Prefill，然后按槽位同步 Batched Decode。
- Batch 开始后不进新请求，完成请求的槽位空转到最长请求结束——输出预算 `[8,16,32,64]` 的批次运行到 64 轮，有效槽位利用率仅 `46.88%`，有效吞吐 57.63 token/s。
- 异长 Prompt Padding 有效率 46.88% 时有效 Prompt 吞吐减半（6880.90 vs 14641.75 token/s），而 Prefill 延迟与峰值显存几乎不变——浪费来自无效 Token 也占满计算，不是"短 Prompt 拖慢 Batch"。
- 收益侧：Batch Size 16 相对串行 makespan 约 `13.98×`；但 Batch Size 192 吞吐 2981.83 token/s 高于 196 的 2931.97——接近显存边界后延迟上升，最大可跑 Batch ≠ 吞吐最优 Batch（197 起在 Prefill LM Head 处 OOM）。

**重点**：**固定 Batch 的吞吐收益来自并行度，而 padding 浪费与空转槽位会把它抵消——拼批的关键是请求长度整齐，不是越大越好。**

**边界与常见误区**：float32 下无请求串扰；bfloat16 下 Batch Shape 改变矩阵 Kernel 的归约顺序，候选 Logits 接近时可能分叉——这是数值现象，不是"Batch 降低准确率"。离线结果不含排队与网络，不代表在线 SLO。[易错] 把 `46.88%` 的浪费理解成"短 Prompt 拖慢别人"——真实原因是矩形 Tensor 按最长长度执行。

**产品对应**：离线批量任务（评测打分、数据标注）仍用固定 Batch 思路；在线服务已被 Continuous Batching 取代。

**方案视角**：面向离线批量客户时问"请求长度分布是否整齐？"整齐可用静态拼批吃满并行度；在线交互业务直接看下一章。

**讲给客户的一句话**：拼批能提吞吐，但请求长短不齐时 padding 空转可能浪费一半算力。

**掌握目标**：能算出给定长度分布下 Batch 的 Padding 有效率；能区分"批越大越好"与"吞吐最优 Batch"两个命题。

**深挖入口**：帖子「Static Batching」（8/26）· 教材 p72 · 仓库 chapter_03

### ch04 · Continuous Batching

**一句话定位**：让已完成请求的 Batch 槽位在下一次迭代被等待请求复用。

**解决什么问题**：固定 Batch 里先完成的请求不退出，后面的请求只能等整批结束——在线场景下请求到达和输出长度的随机性让槽位常年空着。Continuous Batching 允许这些名额在迭代边界被"见缝插针"地补满。

**原理图解**：<!--fig:ch04-->

**原理要点**：
- baseline 成员固定、完成槽位空转；variant 在每次模型迭代前执行"回收完成槽位 → FCFS 填补 → 有新请求做 Batched Prefill，否则对剩余槽位 Decode"——唯一变量是"完成名额能否复用"。
- 异长输出场景：makespan `1.169×`、吞吐 +16.8%、执行槽位有效率 62.50%→74.07%；等长输出负对照两条路径几乎无差（约 0.4%）——没有可提前复用的槽位就没有收益，机制成立的前提是负载不整齐。
- 固定到达轨迹（每 150 ms 一请求）：makespan `1.486×`、TTFT p95 1511→205 ms，但 ITL p95 36.50→71.48 ms；干扰实验里 512 Token 新 Prompt 在中途到达使 ITL 最大值 35.07→72.71 ms——完整 Prefill 会阻塞已有的 Decode。
- KV Cache 本期仍是 dense 固定容量的 `[max_running_requests, heads, physical_length, head_dim]` 槽位入库；新请求更长时全批左扩容搬移，这是留给 ch05 的浪费。

**重点**：**Continuous Batching 在迭代级复用完成槽位，但新请求的完整 Prefill 会卡住已有请求 Decode——这个矛盾要等 Chunked Prefill 才能解决。**

**边界与常见误区**：本期 Prefill 优先、FCFS 策略是为了可复现而非最优调度的声明；CPU 调度与状态管理仍有约 2.3% 开销；dense Cache 扩容须搬移。[易错] 把 ITL p95 恶化读成机制失效——是 Prefill 阻塞 Extend 其他请求的确认代价，不是复用机制的错误。

**产品对应**：vLLM/SGLang 等在线引擎的标准调度方式，源自 Orca 论文的 iteration-level scheduling。

**方案视角**：评估在线引擎时的必选能力；观察指标是" TTFT p95 和 ITL p95 是否同时可接受"——只看一项会被 Prefill 干扰欺骗。

**讲给客户的一句话**：Continuous Batching 让新请求见缝插针，但 Prefill 与 Decode 会互相干扰，延迟分位要一起看。

**掌握目标**：能画出请求在 waiting→running→finished 状态机中的槽位复用流程；能判断什么负载下 Continuous Batching 无收益。

**深挖入口**：帖子「连续批处理」（8/27）· 教材 p102 · 仓库 chapter_04

### ch05 · Paged KV Cache

**一句话定位**：把 KV 显存切成固定物理 Block，用 Block Table 把逻辑序列映射到物理块。

**解决什么问题**：dense KV Cache 按全局最长请求预留物理宽度：一个 4096 Token 的请求会让所有槽位在 4096 级别上跑 masked Attention，长度不均时显存浪费严重、并发上不去——而"上万人同时用"恰恰意味着长度分布杂乱。

**原理图解**：<!--fig:ch05-->

**原理要点**：
- baseline 沿用 dense 槽位，长请求加入时所有层所有槽位一起左扩容搬移、完成后宽度不回收；variant（variant 路径）借鉴操作系统虚拟内存分页：显存切成物理 Block，请求只为实际长度申请 Block，尾块浪费限制在最后一个 Block，完成后物理块直接归还复用，不动其他请求的 K/V。
- 机制示意：一个物理 Block 携带全部层 K/V（`[num_layers, 2, kv_heads, block_size, head_dim]`），每请求只存 Block Table 与逻辑长度；逻辑连续不要求物理连续。
- 成硅收益随长度不均显著增大：3072 Token 长请求下 KV 读取减少 94.1%、Cache Pool 减少 82.6%；但同并发吞吐反而降 36–45%——PyTorch reference 的 Block gather、分组 Attention 与额外 Kernel Launch 吃掉了计算节省。
- 真正的收益在容量：12GB 卡上 dense 最大成功并发 8（12 起 OOM），paged 跑到并发 16，吞吐 157.80 token/s 高于 dense 最优约 0.9%、峰值显存低约 43.0%。

**重点**：**分页把显存占用从"按最长请求预留"变成"按需分配"——收益主要在并发容量，同并发下的单步速度要靠专用 Kernel 而非分页本身。**

**边界与常见误区**：减少 K/V 访问 ≠ 端到端更快，Block gather 的 Kernel 效率同样决定成败；Block Pool 按已知上界预分配、无动态准入。[易错] "上了分页吞吐就涨"——本期同并发吞吐是负的，价值在容量换吞吐。

**产品对应**：vLLM 的 PagedAttention；vLLM/SGLang 的专用 CUDA/Triton Kernel 直接读 Block Table，无本期 PyTorch reference 的开销。

**方案视角**：先问客户两个问题："你们的 Prompt 长度分布有多离散？""GPU 显存余量有多紧？"长度越离散、显存越紧张，分页收益越明显；再算容量账：同一张卡能多接多少并发。

**讲给客户的一句话**：分页把显存从"按最长请求预留"变成"按需分配"，同一张卡能多接一倍请求。

**掌握目标**：能画出 Block 的逻辑 Token 位置→物理 Block 的映射；能判断"减少 KV 读取"与"端到端变快"何时不是一回事。

**深挖入口**：帖子「Paged KV Cache」（8/28）· 教材 p144 · 仓库 chapter_05

### ch06 · 迭代级推理请求调度器

**一句话定位**：在迭代边界用 Token Budget 约束每轮 Prefill 强度，保护已有 Decode。

**解决什么问题**：ch05 解决了显存分配，但调度本身没有"算力预算"的概念：Prefill 突发涌入时，调度器可以连续多轮做 Prefill，把已在生成中的用户长时间晾着——界面上的"系统停了"多半是这个原因。

**原理图解**：<!--fig:ch06-->

**原理要点**：
- baseline 是纯 FCFS：有空闲槽位就一次 Prefill 所有可加入的请求；variant 是软 Token Budget：按 padded cost `batch_size × max(prompt_length)` 扣预算、预算不足即停，并且已有 Decode 请求时两个 Prefill iteration 之间至少插一次 Decode iteration。
- "软"的含义：单条 Prompt 超预算时不拆分（本期无 Chunked Prefill），允许队头超预算执行，避免 FCFS 队列永久阻塞。
- 突发 12 条长短混合 Prompt 实测，budget 2048 相对 baseline：吞吐 +13.1%、既有请求最大 ITL 降 58.5%、突发 TTFT p95 降 15.8%、Padding 减少 53.0%。
- 但 ITL p95 从 98.52 升到 207.37 ms：baseline 把 Prefill 集中成少数长停顿，budgeted 拆成多次中等停顿——最大值改善，p95 改善仍不成立；budget 256 过紧，反而因 6 次超预算请求吞吐 -6.8%。

**重点**：**调度器的核心不是"多收请求"，而是限制每轮 Prefill 的预算并强制插入 Decode 轮，保护已开始的会话不被卡住。**

**边界与常见误区**：预算按 padded 而非有效 Token 计，Padding 本身也耗预算；无显存准入、抢占、优先级与 P/D 分离。[易错] "预算调小总是更公平"——过紧的预算引发超预算执行和更多轮次开销，吞吐反向下降。

**产品对应**：vLLM/SGLang 的迭代级调度与 Decode 保护；`max_num_batched_tokens` 类参数即 Token Budget。

**方案视角**：评估调度策略先问两件事："单轮 Prefill 的最大时长是多少？""已有会话的最长卡顿（ITL max）是多少？"部署时 Token Budget 默认非最优，需按负载扫档。

**讲给客户的一句话**：调度器的核心指标是"已开始的对话不被新请求卡住"，要看最大停顿而非平均。

**掌握目标**：能画出 FCFS + Token Budget 的每轮选择逻辑；能判断 ITL max 与 ITL p95 何时反向变化的原因。

**深挖入口**：帖子「调度器」（8/31）· 教材 p183 · 仓库 chapter_06

### ch07 · Chunked Prefill

**一句话定位**：把长 Prompt 按预算切成多轮增量 Prefill，在 Chunk 间穿插 Decode。

**解决什么问题**：软预算下长 Prompt 必须整条执行——一条 2048 Token 的 Prefill 一次就叫同批 Decode 停数百毫秒，预算形同虚设。只有单条 Prompt 可拆，Token Budget 才能从"软建议"变成"硬约束"。

**原理图解**：<!--fig:ch07-->

**原理要点**：
- baseline（full）沿用 ch06 软预算，长 Prompt 一次完整 Prefill、可以超预算；variant（chunked）给请求新增 `prefilling` 状态与 `prefill_cursor`，Prompt 按 Token Budget 跨多轮推进，每轮只把本段 K/V 追加进 Block Table，中间 Chunk 不出 Token，最后一块完成才生成首 Token。
- Chunk 内 Attention 用全局因果偏移：逻辑位置 `q` 只能看见 `k ≤ q`，共享同前缀长度的多请求 Chunk 由调度器分在同一轮计划。
- 长 Prompt 干扰实验（budget 256，2048 Token）：最大单轮 Prefill 316.28→65.39 ms（-79.3%）、既有请求最大 ITL 降 67.8%、峰值显存降 16.8%；代价是长请求 TTFT +141.6%、吞吐 -6.2%——本质是延迟在请求间再分配。
- 负对照很干净：短 Prompt（128 Token）场景没有可拆工作，chunked 反而慢 6.4%；无并行 Decode 场景中 ITL 隔离收益按定义为零。

**重点**：**Chunked Prefill 用长请求首 Token 稍晚，换取其他请求不被一次长 Prefill 卡住——预算有硬约束的前提、单请求 Prefill 可拆分。**

**边界与常见误区**：这是引擎内部对同一 Token 序列的增量计算，不是 RAG chunking 或流式接收；Chunk 切得越小停顿越短但迭代与 Kernel Launch 开销越多，ITL max 改善不保证 ITL p95 与吞吐同时改善。[易错] 把 Chunked Prefill 当作无条件加速器——两个负对照都证明了收益需要"有要保护的 Decode + 有要拆的长 Prompt"。

**产品对应**：vLLM/SGLang 均默认启用 Chunked Prefill，`chunks`/`max_prefill_tokens` 类参数控制粒度。

**方案视角**：负载里有长文档/Prompt 输入 + 流式输出响应交互时必开；纯批量生成（无并发交互）时收益有限，反而多吃轮次开销。

**讲给客户的一句话**：Chunked Prefill 用"首 Token 稍慢一点"，换"其他用户不掉帧"——本质是延迟在用户之间的再分配。

**掌握目标**：能画出 prefilling→running 状态下 Prefill Cursor 的推进与穿插 Decode 的时序；能判断某负载该不该开 chunked。

**深挖入口**：帖子「Chunked Prefill」（9/3）· 教材 p225 · 仓库 chapter_07

### ch08 · Prefix Cache

**一句话定位**：块级缓存共享前缀的 KV Block，多请求命中同前缀时免重算。

**解决什么问题**：系统提示、few-shot、合同/代码模板——很多业务所有请求带着同一段长前缀。每条请求都把前缀重新 Prefill 一遍，是纯浪费。块级 Prefix Cache 让后续请求直接复用已算好的完整前缀 KV Block。

**原理图解**：<!--fig:ch08-->

**原理要点**：
- baseline 每条请求重算完整 Prompt；variant 匹配最长连续完整前缀，只 Prefill 未命中的后缀——机制差异只在"前缀 KV 是否重算"。
- 缓存键是链式的：`SHA256(model_namespace || parent_hash || block_token_ids)`；相同 Token 块在不同历史上下文里 KV 不同，不能对单块单独哈希误判断命中。
- 只共享已填满并发布为不可变的 Block，未填满尾块与 Decode Block 私有——因此无需 Copy-on-Write；KV 不含首 Token Logits，最多复用 `prompt_length - 1`，至少重算最后一个 Prompt Token。
- 主实验（8 请求共享 2048 前缀）：热请求 service TTFT p50 降 86.30%、Prefill Token 减少 84.85%（16896→2560）；但 Makespan 只降 11.33%——Decode 不受益，且缓存池多占 280 MiB。

**重点**：**Prefix Cache 省的是 TTFT 和 Prefill 算力，不省 Decode；生命周期靠"请求引用计数 + 缓存所有权 + leaf-LRU"管理。**

**边界与常见误区**：不做部分块共享/COW、并发冷请求去重（in-flight dedup）、跨模型或跨进程共享（后者是 ch14）。[易错] "命中率决定一切"——readme 表明命中多但 Makespan 只小降；收益上限由 Decode 占比决定，缓存池还占用显存。

**产品对应**：vLLM 的 Prefix Caching、SGLang 的 RadixAttention（Radix Tree 存储）；跨进程跨实例共享见 LMCache（ch14）。

**方案视角**：先问"业务前缀是否整齐、多长"：高命中率业务（统一系统提示/RAG 长 context）值得开，随机对话业务收益小且占显存；有前缀的业务用指标：Prefill Token 实际数 vs 请求数。

**讲给客户的一句话**：前缀缓存的价值取决于前缀是否整齐——命中省的是首 Token 延迟和 Prefill 算力，Decode 一秒都省不了。

**掌握目标**：能画出前缀 Block 的链式哈希结构与复用流程；能判断一笔业务的前缀命中率是否值得开启该功能。

**深挖入口**：帖子「Prefix Cache」（9/4）· 教材 p256 · 仓库 chapter_08

### ch09 · FlashAttention

**一句话定位**：Triton 分块计算 + Online Softmax，Attention 中间矩阵不落全局显存。

**解决什么问题**：Eager Attention 显式物化 `QK^T → softmax → PV` 的完整 Score/Probability 矩阵，中间显存随序列长度平方增长——4096 Token 增量峰值 2576 MiB，8192 直接 OOM。长上下文生不能升，先卡在 Attention 的写法上。

**原理图解**：<!--fig:ch09-->

**原理要点**：
- baseline（eager）：显式物化全量 Score 矩阵再 softmax；variant（flash，手写 Triton inference-only 前向）：分块读入 K/V，每个 Query 行维护最大值 `m`、归一化总和 `l`、累加器 `acc`，用 Online Softmax 逐块修正（`alpha = exp(m − m_new)`）——完整中间矩阵不写显存。
- 数学结果相同、计算量仍是二次方：降的是显存读写规模，不是 FLOPs。
- 微基准（因果 Attention、head_dim 128）：1024 Token `3.38×`、4096 达 `3.70×` 且增量峰值仅 16 MiB，8192 Token 上 eager OOM 而 flash 可用。
- 接入完整引擎温效果为零：端到端只有 `1.08–1.10×`——Chunked Prefill 把 Query 限制在 256，Projection/MLP/Paged KV 读取/调度稀释了 Kernel 收益。证明面积为"Kernel 加速比 ≠ 模型加速比"。`[关键细节]` Chunked Prefill 要用全局因果偏移 `k ≤ P + q`，不能按局部位置算。

**重点**：**FlashAttention 省的是显存搬运而非计算量；它让长序列能算，收益幅度则由 Attention 占全模型时间的比例决定。**

**边界与常见误区**：本轮不支持 backward/任意 Mask/paging-Aware 的 Paged Attention（K/V 仍先 gather 成稠密再进 Kernel）[INV]; 教学手写实现不代表 PyTorch SDPA 或官方 flash-attn。[易错] "接入 flashattention 能带来一帧几倍"——微基准 3.7× 到引擎层只剩 1.08×，瓶颈在别处，别把 Kernel 基准宣传成产品加速能力。

**产品对应**：FlashAttention 官方库、vLLM/SGLang 内置融合 Attention。

**方案视角**：长上下文业务的前置条件（没有它 8K+ 都困难）；选型时看"Attention 占整体时间多少"而不是 Kernel 公认数字；集成后用端到端 TTFT/吞吐验证。

**讲给客户的一句话**：FlashAttention 省的是显存搬运，序列越长收益越大，但整体加速要看它占总时间的多少。

**掌握目标**：能画出 Online Softmax 的与其借助 m/l/acc 修约式流程；能判断 Kernel 级加速何以在端到端被稀释。

**深挖入口**：帖子「手写FlashAttention」（9/6）· 教材 p291 · 仓库 chapter_09

### ch10 · CUDA Graph

**一句话定位**：把一轮 Decode 提前捕获成 CUDA Graph，重放时一次 Host 提交替代逐个 Launch。

**解决什么问题**：Decode 每步的计算量很小，而 Eager 路径单步约 1903 次 `cudaLaunchKernel`——GPU 在等 CPU 发号施令。小模型 Decode 的瓶颈不是算不出来，是"指挥慢于干活"。

**原理图解**：<!--fig:ch10-->

**原理要点**：
- baseline（static_eager）每轮逐个提交 Kernel（实测 1903 次 `cudaLaunchKernel` + 4 次 `cudaMemcpyAsync`）；variant 把同一程序提前捕获成 Graph，之后更新输入缓冲、一次 `cudaGraphLaunch` 重放——Host 提交从上千次变成 1 次。
- 代价是静态化：每个 `(batch_capacity, context_bucket)` 建独立 Runner，预分配全部 Buffer 和覆盖整个 Bucket 的 KV Block，地址与形状不变，无效位置用 Mask 屏蔽；Replay 前可更新 Block Table 内容与输入 Token。
- 实测：capacity 1、128 Token 下 Decode TPOT 37.90→5.22 ms（`7.26×`）；上下文与 Batch 变大后 GPU 工作变重，capacity 4、2048 Token 下只剩 `2.25×`——Graph 不减少计算，能消除的 Host 开销占比递降。
- Capture 一次 95–140 ms，按每步节省估算 3–5 个 Decode 步即可摊销；Prefill 仍走出 Eager，所以 2048 Token 请求 TPOT 提升 `4.03×` 而端到端只 `2.53×`。

**重点**：**CUDA Graph 消除的是 Host 提交开销，不是计算量；Batch/上下文越大、GPU 越忙，剩余收益越小。**

**边界与常见误区**：需要 Bucket/Padding 换取可捕获性（Bucket 尾部有无效读取）；请求准入、Block 分配仍在 Graph 外；逐 Token `.item()` 同步会抹掉稳态收益。[易错] "Graph 静态化所以显存暴涨"——本期稳态峰值反而略低（复用 Capture 期内存），但 KV Pool 要按 Bucket 整段预留，这才是池扩展的代价所在。

**产品对应**：vLLM/SGLang 均对 Decode 使用 CUDA Graph，按 batch/context 档位缓存 Graph 实例。

**方案视角**：小模型 + 低并发（Decode 粒细）服务收益最大；大模型高并发下 Host 开销占比低，Graph 是标配但增益有限；观察点：Decode TPOT 提升 vs 端到端提升的差距。

**讲给客户的一句话**：CUDA Graph 解决的是"CPU 指挥慢于 GPU 计算"的问题，Batch 和上下文越大、剩余收益越小。

**掌握目标**：能画出 Capture 与 Replay 两段流程；能答出为什么加速比随负载变重而下降。

**深挖入口**：帖子「CUDA Graph」（9/8）· 教材 p324 · 仓库 chapter_10

### ch11 · Greedy Speculative Decoding

**一句话定位**：小 Draft Model 连提候选 Token，大 Target Model 一次批验证，减少串行调用。

**解决什么问题**：Decode 最大的浪费是"每出一个字都要请大模型跑一次"。Speculative Decoding 让便宜的小模型连续猜几个候选，大模型一次批验证：接受最长匹配前缀、拒绝处补 Token——数学上 Greedy 输出与只跑大模型完全一致。

**原理图解**：<!--fig:ch11-->

**原理要点**：
- baseline（Target-only）逐 Token 串行生成；variant：Qwen3-0.6B 作为 Draft 连提 γ 个候选，Qwen3-4B 作为 Target 一次块验证——接受最长匹配前缀、拒绝处补 Correction Token、全对补 Bonus Token，收益完全取决于接受率与两个组件的成本。
- 两套 KV Cache 按同一接受边界提交或截断回滚——输出提交与 KV 回滚必须在同一个边界，否则两套 Cache 会漂移。
- FP16 下与 Target-only 逐 Token 等价的确定性成立；但实测在 3080 Ti 上几乎全是负优化：中文解释负载 γ=8 接受率仅 0.173，TPOT 加速比 0.284×；只有接受率 1.0 的代码样例在 γ=8 得到 1.023×。
- 根因在组件成本：Draft 单步 42.51 ms 只比 Target 的 48.16 ms 快约 12%——Eager 教学栈让两者都被小算子和 Host Launch 拖住，参数量 6.7 倍差距没转化为速度差；而一轮推测仍要付 γ+1 次 Draft 前向。

**重点**：**投机解码是"用便宜计算赌接受率"；Draft 不够快或接受率不够高时是纯亏，且每轮都要多付 γ+1 次 Draft 前向。**

**边界与常见误区**：只做 Greedy（非随机 Speculative Sampling）、Batch 1、无 MTP/EAGLE/Medusa。[易错] "Draft 候选生成器越 الاقتص也多收益越高"——γ 大则错得更远；接受率 0.357–0.833 的组别在 γ=1/2/4/8 的加速比始终低于 1——组件成本比越长才是第一关。

**产品对应**：vLLM/SGLang 内置 speculative decoding；MTP/EAGLE 是更轻的候选生成器，信号是"候选生成尤其便宜"场景的替代物种。

**方案视角**：评估模型：固有数字"是否"？"输出是否重复性强（代码/模板）"（接受率高）且"Draft 是否明显比 Target 快一个量级"（原生 MTP 或融合 Kernel）；要按业务 Prompt 分布实测，不能听开发者"理论上快"。

**讲给客户的一句话**：投机解码是"用便宜计算赌接受率"，Draft 不够快或接受率低时是纯亏——上线前要按业务负载实测。

**掌握目标**：能画出候选提出→批验证→接受/补正的流程；能按 Draft 单步加速比与接受率，判断实测是否值得上线。

**深挖入口**：帖子「Speculative Decoding」（9/10）· 教材 p362 · 仓库 chapter_11

### ch12 · KV Cache Offloading

**一句话定位**：GPU Block 不足时，把受害请求的 KV 换到 CPU 或弃算，用延迟换准入。

**解决什么问题**：GPU 显存装不下所有并发请求的 KV，新请求进不来。乐观准入 + 抢占给出第三条路：池满了就让晚来的请求挤掉一个受害请求——受害者在哪暂停？其 KV 是搬到 CPU（swap）还是扔掉重算（recompute），这是本章的上半部。

**原理图解**：<!--fig:ch12-->

**原理要点**：
- 两个抢占路径对比：`swap` 把整请求 KV 经 PCIe 换出到 Pinned CPU 池，恢复时换回断点续算；`recompute` 直接丢弃 KV，恢复时对 prompt + 已生成部分重新 Prefill（另有 conservative 保守准入与 664 块大池两类对照）。
- 为什么是整请求：Decode 每步 Attention 要读全部历史 KV，"运行中部分换出"不存在；受害单位是整请求，被暂停请求的 KV 才是"暂时无用"的部分。
- 本环境下 swap 机制成本更低：换出+换入双向传输合计约 49–63 ms（实测约 11.9 GB/s，正值，仅限 PCIe Gen3 x16）；recompute 的重新 Prefill 暴露 673–764 ms，差距远超需带宽补偿幅度。
- 容量收益成立：同池下 swap 吞吐 37.0 vs 保守准入 35.7 token/s（池 230）、44.9 vs 40.7（池 320），逻辑并发从 2–3 升到 4–5；但都低于 664 块大池的 45.5——池充足时抢占收益为零。

**重点**：**抢占把"显存不足=失败/拒绝"变成"慢一点但能跑"；代价由被抢占请求的秒级暂停承担，而恢复方式的选择由带宽和重算成本的相对值决定。**

**边界与常见误区**：换出副本只用于恢复原请求，不做复用；单请求上下文上限仍由 GPU 池决定；所有传输在默认 Stream 上同步、拷贝期间 GPU 不推进其他工作。[易错] "swap 一定优于 recompute"——端到端均值受负载支配（池 230 中 recompute 略胜），机制计时才能区分；ITL max 约等于受害请求暂停时长（秒级），不能把秒级暂停归因于几十毫秒的 PCIe 拷贝。

**产品对应**：vLLM 的 swap-based 抢占与 CPU Offload；Pinned Memory 池用于预分配而非临时申请。

**方案视角**：适用于"不能拒绝请求但可容忍偶发慢回复"的客户；池大小决策：宁可保守预留还是乐观准入+抢占？看 pipeline SLA 对秒级延迟尖刺的容忍度。

**讲给客户的一句话**：换出把"显存不够就失败"变成"慢一点但能跑"，代价由被抢占请求的延迟承担。

**掌握目标**：能画出抢占→暂停→换出→恢复的两条路径；能判断 swap 与 recompute 何者更优的判据（带宽成本 vs 重算成本）。

**深挖入口**：帖子「Offloading」（9/12）· 教材 p381 · 仓库 chapter_12

### ch13 · KV Cache 异步传输

**一句话定位**：把 KV 换出/换入放到独立 Copy Stream，用 Event 管理块所有权以隐藏传输。

**解决什么问题**：ch12 的换出在默认计算 Stream 上同步拷贝——拷贝那几十毫秒里，GPU 只能站着等。独立 Copy Stream 让数据和计算各走各的通道，用 CUDA Event 保证"谁在搬、谁能用"的正确性。

**原理图解**：<!--fig:ch13-->

**原理要点**：
- baseline（sync）在计算 Stream 逐块传输，完成后调度器才继续；variant（async）在独立 Copy Stream 提交 D2H/H2D，用 CUDA Event 延迟转移 Block 所有权——D2H 完成前 GPU 源块不能释放，H2D 完成前目标块不能读；调度器只在无工作可做或下一步必须申请 Block 时才等 Event。
- 请求新增 `swapping_out / swapping_in` 中间状态；`Event.query()` 只轮询、不等待；正确性由"所有权未转移就不动块"保证，而不是由"先拷贝完"的同步直觉保证。
- 受控微基准证明本机具备重叠能力：传输与 BF16 GEMM 并发 1.21–1.50×；但主实验不是单向正优化：池 230 组 async 略慢（39.31 vs 39.61 token/s），池 320 组较快（47.62 vs 45.55）。
- 机制结论看专门指标：async 把暴露等待 48.3→20.0 ms（池 230）、62.2→19.2 ms（池 320）；M 所以 D2H 在紧急水位触发、完成前源块不能释放，重叠窗口可能大部分暴露。

**重点**：**异步传输的正确性靠 Event 管块所有权，不是靠提交顺序；能赚多少取决于"拷贝时有没有别的计算可做"——不是开了就快。**

**边界与常见误区**："异步提交"不等于"传输已完成"；D2H 水位触发过晚时窗口大部分暴露；不实现 SSD/RDMA/预测性换出。[易错] 认为端到端应能稳定超越 sync——主实验的差异多于几十毫秒传输窗口，且单次运行波动明显，机制结论必须看分阶段 Event 指标。

**产品对应**：vLLM/SGLang 的 offloading 流水线、NIXL 等 KV 传输层的异步 Stream 设计。

**方案视角**：当抢占频繁且同时有多个请求可推进时，异步传输才显价值；评估指标是"暴露等待时间"，不是设备传输总时间。

**讲给客户的一句话**：异步传输用 Event 管住块所有权，收益取决于端到端是否被迫等待——不是开了就快。

**掌握目标**：能画出 Copy Stream 与 Event 化的块所有权转移时序；能判断"暴露等待"与"总传输时间"哪个才是优化目标。

**深挖入口**：帖子「异步传输」（9/14）· 教材 p406 · 仓库 chapter_13

### ch14 · 外部 KV Cache 与 LMCache

**一句话定位**：用进程无关的 Token Chunk 标识 KV，使其可跨进程、跨请求、跨实例被复用。

**解决什么问题**：ch08 的 Prefix Cache 只在同一进程内共享物理块；引擎重启、多实例部署时，其他请求或进程命中的前缀都得重算。如果 KV 以 Token 序列为身份存在服务端，任何进程都能"Lookup 命中→加载→只算尾巴"。

**原理图解**：<!--fig:ch14-->

**原理要点**：
- baseline（recompute）新请求完整重算；variant（external）用链式 CacheKey `SHA256(model namespace || parent digest || token ids)` 标识 256 Token 的外部 Chunk，向独立 Cache Service 发 Lookup/Load/Store，命中的 CPU KV 重建到新申请的本地 GPU Block（因外部对象不含本地 Block ID，Block Table 必须重建），只 Prefill 未命中后缀。
- 与 ch08 的关键差异：外部 Chunk 只存 `prompt_length - 1` 范围的完整 Block，KV 不含首 Token Logits，至少要重算最后一个 Prompt Token 才能出首 Token。
- 正确性完美：跨进程命中 512 Token、warm Prefill 只算 1 Token、输出与重算逐位一致、Producer 退出后 Consumer 仍能命中。
- 但性能全负优化：2048 Token 前缀 warm TTFT 2131.68 ms vs 重算 480.62 ms（0.23×）——Lookup 只要约 1 ms，成本在 TCP 传输（Payload 约 235 MB）与逐 Chunk 重建；"已经命中"不等于"值得加载"。

**重点**：**KV 变成可跨进程命中的"外部对象"是架构上的关键一步；但加载成本必须在数据路径上被压到比重算更便宜，缓存才不会变负资产。**

**边界与常见误区**：本期是同步 TCP + 用户态复制，非共享内存/CUDA IPC/GPUDirect/RDMA；Cache Service 无持久化、无认证。[易错] "命中了就赚"——当传输成本 > 重算成本时，命中反而在损失；负优化的存在不是机制错误，是传输路径不达标。

**产品对应**：LMCache（ExternalCacheClient↔Connector、Token Chunker↔Token Index、原始 KV Payload↔Memory Object、CacheStore↔CPU Backend + LRU）；vLLM 的 KV Connector 接口。

**方案视角**：多实例部署或服务重启频繁的业务才需要；评估按"命中率 × 复用范围 × 传输速度"三项问：Lookup 便宜吗？加载走什么路径？按业务带宽算出"加载 vs 重算"的成本临界点。

**讲给客户的一句话**：外部 KV Cache 的价值在"命中率 × 复用范围"，但必须保证加载比重算快——慢速传输会让缓存变负资产。

**掌握目标**：能画出 Lookup→Load→重建 Block Table→算尾 Token 的流程；能算出一笔前缀的"加载 vs 重算"成本临界线。

**深挖入口**：帖子「LMCache」（9/16）· 教材 p424 · 仓库 chapter_14

### ch15 · Prefill/Decode 分离

**一句话定位**：Prefill 与 Decode 拆进不同 Worker，经 KV Handoff（含 ACK）实现所有权转移。

**解决什么问题**：Prefill 是计算密集、Decode 是访存密集，混在同一个 Worker 里互相干扰；扩容时也难以按各自瓶颈独立扩。P/D 分离的极简版：Prefill Worker 算完 Prompt KV 与首 Token，把 KV 移交给 Decode Worker 继续生成——那时才能谈"Prefill 与 Decode 独立扩容"。

**原理图解**：<!--fig:ch15-->

**原理要点**：
- baseline（monolithic）同一进程完成 Prefill、首 Token 与 Decode；variant（disaggregated）跨进程 Handoff：Prefill Worker 导出逻辑 KV Payload，经同步 GPU→CPU→TCP→CPU→GPU 路径发布，Decode Worker 重建本地 Paged KV 后返回 ACK。
- ACK-gated 语义：确认所有权转移后 Decode 侧续算、首 Token 才可发给客户端；Payload 损坏或 Namespace 不匹配时回退（fallback），Decode 侧完整重算保证正确性——split TTFT = Prefill + Export + Publish + Receive + Import + ACK。
- 与 ch14 的差异：Handoff 是同一 `Request ID + Attempt ID` 的一次性所有权交接，不做前缀 Lookup、不保留副本。
- 正确性达标（三 Worker PID 不同、首 Token Logits 摘要一致、输出与 mono 逐 Token 一致），但单卡串行路径明确负优化：2048 Token 的 KV Payload 224 MiB，split TTFT 是 mono 的 `5.49×`（2378 vs 433 ms），其中 Publish 约 896 ms、Export 约 433 ms；换 1.7B 压力模型倍数只缩到 `5.23×`——"负优化是模型太小"不成立，是单请求上 split 永远比 mono 多一次 KV 交接。

**重点**：**P/D 分离的机制价值在"角色分离后的独立扩展与干扰隔离"，而非单请求更快——单请求路径上它只是多一次昂贵的 KV 交接。**

**边界与常见误区**：同卡串行不具备 GPU 资源隔离； HANDOFF Service 无认证/TLS。[易错] "上 P/D 分离就加速"——本期单卡实测全负优化；多 GPU 的并行吞吐收益不在范围，部署时真实收益来自多实例调度与干扰隔离，不能拿本章数字直接推产品效果。

**产品对应**：vLLM disaggregated prefill、DistServe、Mooncake 等多卡/多机 P/D 分离方案；KV-aware 调度是后续主题。

**方案视角**：大规模并发下的资源配比问题：Prefill（计算密集）与 Decode（访存密集）能否按各自瓶颈分别扩容、互不干扰？没有多卡资源隔离和 KV-aware 调度的单卡业务，P/D 分离只是"多了一次数据搬家"。

**讲给客户的一句话**：P/D 分离卖的是大规模下的资源配比和干扰隔离，不是单请求更快——单卡串行演示里它只会更慢。

**掌握目标**：能画出 ACK-gated 的 Handoff 时序（Publish/Receive/Import/ACK/释放）；能判断某业务何时该考虑 P/D 分离、何时不该。

**深挖入口**：帖子「PD分离」（9/18）· 教材 p453 · 仓库 chapter_15
