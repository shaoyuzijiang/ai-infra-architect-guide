# 产品与方案速查（售前地图）

机制学完要落到"客户拿什么产品、打什么场景"。这张地图把课程机制 ↔ 业界产品 ↔ 常见方案对应起来（除标明外均为 `[业界通识]`）。

## 推理侧：产品 ↔ 机制

| 产品/框架 | 机制对应 | 一句话场景 |
|---|---|---|
| vLLM | `PagedAttention`(ch05) · 连续批处理(ch04) · `Chunked Prefill`(ch07) · Prefix Caching(ch08) · speculative decoding(ch11) | 当前开源推理服务的事实标准之一，生产 API 服务默认起点 |
| SGLang | `RadixAttention`(ch08 的前缀树增强版) · overlap 调度 | 多轮对话/Agent 场景前缀复用更激进 |
| TensorRT-LLM | fused kernel、in-flight batching、paged KV | NVIDIA 生态闭源栈，追求单卡极致吞吐 |
| LMCache | 外部 KV Cache 分层缓存(ch14) | 跨实例/跨会话复用 KV，多轮+长 system prompt 场景 |
| Mooncake | KV 为中心的 PD 分离架构(ch15) | 超大规模服务的 KV 池化与调度 [业界通识] |
| Transformers / TGI | ch01 的"参考实现"与基线推理 | 原型与质量对照；生产高并发一般不选它 |
| llama.cpp / Ollama | 单机量化推理（量化不在本课范围） | 端侧/笔记本本地跑 |

## 训练侧：产品 ↔ 机制

| 产品/框架 | 机制对应 | 一句话场景 |
|---|---|---|
| PyTorch FSDP | ZeRO-3 参数分片生命周期(ch34) | PyTorch 原生大模型训练，选 FSDP 而非第三方栈 |
| DeepSpeed ZeRO-1/2/3 | 优化器状态/梯度/参数逐级分片(ch32–34) | 显存不够的老牌解法；配套 offload 另有章节 |
| Megatron-LM | TP/SP/Pipeline(ch35–41) | 预训练集群的事实标准并行范式 |
| NCCL | AllReduce/Reduce-Scatter/AllGather 通信库(ch29–38 的底层) | 选型时先看它对目标硬件的成熟度 |
| `torch.compile` | Inductor 图编译(ch28) | 训练步体免费的 10–30% 量级优化（量级视负载） |
| `torch.distributed.checkpoint` | 分片 checkpoint+manifest(ch43) | 大集群断点续训的官方路径（README 直接引用） |
| HuggingFace TRL/Accelerate | SFT/LoRA 封装(ch16) | 私有化 SFT 项目的快速落地层 |

## 方案视角：常见单子怎么套

| 客户场景 | 课程机制组合 | 现场要先问的问题 |
|---|---|---|
| 企业知识库/RAG 问答 | Prefix Cache(ch08) + 外部 KV(ch14) | system prompt/检索文档是否每轮重复出现（前缀命中率决定收益） |
| 多轮 Agent 平台 | Prefix + KV Offloading(ch12) + 容量按"并发×上下文×驻留时长"算 | 会话平均驻留多长？工具结果要不要进上下文？（数百倍存储放大） |
| 高并发在线 API | 连续批(ch04) + Token Budget(ch06) + Chunked Prefill(ch07)；规模大再上 PD 分离(ch15) | 日活没有意义，要 QPS 与上下文长度分布 |
| 离线批量分析/数据蒸馏 | 固定批(ch03)也够 + Chunked Prefill | 在线 SLO 指标不适用离线吞吐 |
| 私有化 SFT 训练 | 数据工程(ch19–22) + BF16/累积/CKPT(ch23–25) + ZeRO 选型(ch32–34) | 卡型/卡数/网络（决定 ZeRO 级别与是否上 TP） |
| 预训练/大集群 | 三维并行(ch42) + 分布式 Checkpoint(ch43) | 拓扑与故障率（恢复时间目标决定 checkpoint 频率） |
| 国产化/信创替代 | 全课机制与具体 GPU 解耦 | 生态三件套：算子覆盖、通信库、推理引擎适配 |

> 使用建议：报价/方案文档里引用本课机制时，写"机制名+章节号"（如 Paged KV Cache，ch05），别写绝对加速比——README 的实测只属于当时的软硬件组合，现场要重新口径。
