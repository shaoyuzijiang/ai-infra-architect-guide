# 原课程索引（保留原件）

课程主线 = GitHub 仓库 [SVDX-lab/miniinfra](https://github.com/SVDX-lab/miniinfra) 43 章。作者：小红书 **@因式分解X**（主页：[xiaohongshu.com/user/profile/6a6234040000000010000401](https://www.xiaohongshu.com/user/profile/6a6234040000000010000401)）。推理篇另有全文教材（479 页 PDF，各章起始页见下表）。

> 更新状态（截至 2026-10-17）：仓库已发布到 ch43；帖子发到训练篇 14（ch29）。ch30–43 的帖子尚未发布——按原作者节奏（10 月约每日一更）会陆续跟上。

## 推理篇（ch01–15）

| 章 | 主题 | 帖子（日期） | 教材 |
|---|---|---|---|
| ch01 | 手写 Qwen3-0.6B 推理 | 手写Qwen3推理(一)(二)+配套源码（8/24） | p1 |
| ch02 | KV Cache——第一个推理优化 | 超轻量AI Infra入门教程02 kv cache(1)(2)+源码（8/25） | p45 |
| ch03 | 固定批处理 | Static Batching(1)(2)（8/26）+第三期-固定批处理的代码（8/27） | p72 |
| ch04 | Continuous Batching | 超轻量AIInfra04 连续批处理(1)(2)（8/27） | p102 |
| ch05 | Paged KV Cache | 从零实现Paged KV Cache（8/28） | p144 |
| ch06 | 推理请求调度器 | MiniInfra系列之推理请求调度器（8/31） | p183 |
| ch07 | Chunked Prefill | 从零实现Chunked Prefill（9/3） | p225 |
| ch08 | Prefix Cache | 从零实现Prefix Cache-infra系列教程8（9/4） | p256 |
| ch09 | 高性能 Attention | 手写FlashAttention（9/6） | p291 |
| ch10 | CUDA Graph 优化 Decode | CUDA Graph（9/8） | p324 |
| ch11 | Greedy Speculative Decoding | Greedy Speculative Decoding（9/10） | p362 |
| ch12 | KV Cache Offloading | KV Cache Offloading（9/12） | p381 |
| ch13 | KV Cache 异步传输 | KV Cache 异步传输（9/14） | p406 |
| ch14 | 外部 KV Cache 与 LMCache | 外部KV Cache和LM Cache（9/16） | p424 |
| ch15 | Prefill/Decode 分离 | 手写PD分离（9/18） | p453 |

## 训练篇（ch16–43）

| 章 | 训练篇 | 主题 | 帖子（日期） |
|---|---|---|---|
| ch16 | 01 | 手写 Qwen3-0.6B 全参数 SFT | 9/20 |
| ch17 | 02 | Assistant-only Loss | 9/21 |
| ch18 | 03 | 固定 Mini-batch | 9/23 |
| ch19 | 04 | 按长度分桶 | 9/28 |
| ch20 | 05 | Sequence Packing | Sequence Packing（9/30） |
| ch21 | 06 | 离线 Token Shard | 10/1 |
| ch22 | 07 | DataLoader 预取与异步 H2D | 10/3 |
| ch23 | 08 | BF16 混合精度 | 10/5 |
| ch24 | 09 | 梯度累积 | 10/10 |
| ch25 | 10 | Activation Checkpointing | 10/10 |
| ch26 | 11 | 训练版 FlashAttention | 10/12 |
| ch27 | 12 | Fused AdamW | 10/13 |
| ch28 | 13 | torch.compile | 10/13 |
| ch29 | 14 | DDP | 10/14 |
| ch30–43 | —（未发帖） | Bucket 重叠/延迟同步/ZeRO 三部/TP 三部/SP/Pipeline 三部/混合并行/分布式恢复 | 读仓库 README |

## 时评支线（非课程主线，穿插读）

**算力供需/成本**：存储的需求侧分析（agent 800× 内存消耗）· agent 对存储消耗的详细推导（8/10）· 光模块需求分析 ×2（8/11）· 自建服务器卖 token 的详细计算 · 国产模型性价比（8/12）· KV Cache 命中/未命中成本极限（8/17）——更早的 7/28–8/9 还有 16 篇同主题（10 亿 Token 成本、GLM-5.2 拆解、ASIC、Ilya 等）。
**行业观察/点评**：从 Google Brain 到 OpenAI（8/13）· 大模型的评估难度（8/16）· DeepSeek-V4-Pro-0813 效果分析（8/17）· Alec Radford 与生成时代（8/20）· 智谱上 infra agent（8/28）· 置顶《AI Infra 行业思考与入门建议》（9/1）· GLM-5.3-Flash 国产化大集群推理技术分析（9/2）· 程序员真正该焦虑什么（9/4）· AI 越来越会写算子了（9/15）· AI 行业观察（一）（10/9）· DeepSeek-V4.1-Flash-1017（10/17）。

## 学法备注

- 原课程设计是"一章一个变量"的受控对照实验课；本站把它转写成架构视角。想动手时随时可回到仓库（每章独立、CPU smoke test 即可起步）。
- 原作者的结论分层习惯值得继承：**机制性结论**（可推广）与**测量性结论**（只属于当时的软硬件组合）分开记。
