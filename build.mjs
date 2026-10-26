// 构建脚本：把 src/*.md 拼装成单页 index.html（零外部依赖的成品页面）
// 用法：node build.mjs（需要 ../node_modules/marked，见 /Users/yzj/Work/package.json）
import { marked } from "../node_modules/marked/lib/marked.esm.js";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(here, "src");
const OUT = path.join(here, "index.html");

const sections = [
  { id: "learn", title: "怎么学", group: "起点", file: "intro.md", svg: null, exit: null },
  { id: "basics", title: "基础概念篇", group: "起点", file: "basics.md", svg: null, exit: null },
  { id: "inference", title: "推理篇 ch01–15", group: "推理篇", file: "inference.md", svg: "inference-arch.svg", exit: "exit-inference.md" },
  { id: "training", title: "训练篇上 ch16–29", group: "训练篇", file: "training.md", svg: "training-arch.svg", exit: null },
  { id: "training-b", title: "训练篇下 ch30–43", group: "训练篇", file: "training-b.md", svg: null, exit: "exit-training.md" },
  { id: "commentary", title: "时评阅读示范", group: "支线", file: "commentary.md", svg: null, exit: null },
  { id: "capacity", title: "容量与成本工程", group: "工具箱", file: "capacity.md", svg: null, exit: null },
  { id: "decisions", title: "决策速查表", group: "工具箱", file: "decisions.md", svg: null, exit: null },
  { id: "products", title: "产品与方案速查", group: "工具箱", file: "products.md", svg: null, exit: null },
  { id: "examples", title: "选型算例库", group: "工具箱", file: "examples.md", svg: null, exit: null },
  { id: "faq", title: "售前/答辩 FAQ", group: "工具箱", file: "faq.md", svg: null, exit: null },
  { id: "index", title: "原课程索引", group: "附录", file: "course-index.md", svg: null, exit: null },
];

function inlineSvg(rel, wrap = true) {
  let s = fs.readFileSync(path.join(SRC, rel), "utf8");
  s = s.replace(/<\?xml[^>]*\?>/g, "").trim();
  s = s.replace(/\n\s*\n/g, "\n");                       // 压缩空行：HTML block 以空行为界，svg 内不能有
  s = s.replace(/<svg /, '<svg class="arch" ');
  return wrap ? `<figure class="archwrap">${s}</figure>` : s;
}

function body(file) {
  let md = fs.readFileSync(path.join(SRC, file), "utf8");
  md = md.replace(/^# .*\n/, "");                       // 章文件自带的一级标题由 section 承担
  md = md.replace(/\]\(#容量与成本工程\)/g, "](#capacity)");
  md = md.replace(/`<!--((?:svg|fig):[a-z0-9-]+)-->`/g, "<!--$1-->");
  md = md.replace(/<!--(?:svg|fig):([a-z0-9-]+)-->/g, (m, name) => {
    const fig = inlineSvg(/^ch\d+$/.test(name) ? `fig/${name}.svg` : `${name}.svg`);
    return `\n\n${fig}\n\n`;                            // 独立成块：行内上下文会让 marked 截断 svg
  });
  return md;
}

// —— 字段组件化：把 **字段**：段落 变成带样式的卡片 ——
const FIELD_CSS = {
  "名词卡": "terms", "算一算": "calc", "数字": "nums", "边界与常见误区": "warn",
  "产品对应": "prod", "方案视角": "plan", "讲给客户的一句话": "say",
  "掌握目标": "goal", "延伸阅读": "read", "深挖入口": "dig", "重点": "key",
};
// —— 各章步骤播放器配置：g = 图上要点亮的编组 id 列表 ——
const STEP_CONFIGS = {
  ch01: [
    { t: "读图纸 config.json", d: "模型的规格清单：28 层、32 个 Q 头、8 个 KV 头、头维 128——后面所有结构都从这里读出来。", g: ["g1"] },
    { t: "读权重 Safetensors", d: "把 6 亿个参数的数值从文件载入显存（bf16）。参数是模型学到的全部知识。", g: ["g2"] },
    { t: "BPE 分词", d: "文字 ⇄ 编号互转：你的输入被切成 19 个 Token（词表里的编号），模型只认编号。", g: ["g3"] },
    { t: "套 Chat Template", d: "把对话装进模型出厂规定的格式信纸：system/user 各就位，模型靠格式区分角色。", g: ["g4"] },
    { t: "Prefill 读完", d: "19 个位置并行过 28 层——一次算完整段输入，顺手产出第一个新 Token。实测 34.43ms。", g: ["g5", "g8"] },
    { t: "Logits 与采样", d: "输出 151,936 个分数（词表逐条打分），greedy = argmax 取最高分。温度/top-p 是别的采样口味。", g: ["g6"] },
    { t: "接回末尾循环", d: "把新 Token 接到句子末尾，回到第 3 步——这就是自回归。", g: ["g7"] },
    { t: "两阶段延迟账", d: "怪现象：读 19 个字只要 34ms，写 1 个字却要 33ms——每步都把全部历史重算一遍（本章没有 Cache）。这是 ch02 的靶子。", g: ["g9"] },
    { t: "正确性对照", d: "Transformers 只当标准答案：同输入同 greedy 下逐 Token 一致、误差 0——手写没错，它不进推理路径。", g: ["g10"] },
    { t: "显存四座山", d: "峰值 1173.80 MiB：权重常驻 + 激活临时 + CUDA 上下文。KV Cache 本章还没有——下一章加。", g: ["g11"] },
  ],
  ch05: [
    { t: "先映射", d: "逻辑块连续、物理块零散：Block Table 记 0→7 1→2 2→11，gather 按表到 Pool 里把对应物理块收上来——逻辑连续不要求物理连续。", g: ["g1"] },
    { t: "两种计价方式", d: "dense 按最长请求整块预留（128 Token 的短请求也在 4096 宽度跑 masked Attention）；paged 只付实际长度，碎片限在各请求自己的尾块。", g: ["g2"] },
    { t: "① 申请", d: "Prefill 时按 ceil(prompt_len/16) 从 Block Pool 取空闲块——258 Token 的 Prompt 取 17 块。", g: ["g3"] },
    { t: "② 写尾块", d: "Decode 逐 Token 往尾块追加——每个请求最后那个未装满的块就是尾块。", g: ["g4"] },
    { t: "③ 满了再取", d: "尾块满了去 Pool 取新空闲块接着装；Pool 按已知上界一次预分配，没有动态准入。", g: ["g5"] },
    { t: "④ gather 读", d: "Attention 按页表收起散落物理块来读；请求完成把块还回 Pool——还块就是释放并发额度。", g: ["g6"] },
    { t: "读账", d: "3072 长请求场景：KV 读取 −94.1%、Pool −82.6%；但同并发吞吐 −36~45%——PyTorch reference 的 gather 与 Kernel 开销吃掉节省。", g: ["g7"] },
    { t: "容量账", d: "12GB 卡固定负载：dense 并发 8 封顶（12 起 OOM），paged 跑到 16 还略高 0.9%，峰值显存 −43.0%——分页的主要收益是容量换吞吐。", g: ["g8"] },
  ],
  ch40: [
    { t: "GPipe 的形状", d: "baseline：连做全部 Forward（Warm-up）再排全 Backward（Drain）——激活生命周期 = 整个 Step。", g: ["g1"] },
    { t: "代价", d: "所有 M 份 micro 激活图须驻留到 Step 结束：M 大时这是显存灾难。", g: ["g2"] },
    { t: "1F1B · Warm-up", d: "开局连做 min(S−1−s, M) 个 Forward：越靠前的 Stage 能提前做的越多；micro 不够时吃下全部，退化成 Fill-Drain 形状。", g: ["g3"] },
    { t: "Steady State", d: "一次 Forward 配一次 Backward 交替 M−warmup 轮。「先收后发」是死锁关键：先发激活再收梯度，gloo 严格阻塞下相邻 Stage 会互等成环。", g: ["g4"] },
    { t: "Cool-down", d: "主循环结束，排空最后 warmup 个 Backward——收尾与更新和 ch39 完全一致，调度只重排时间。", g: ["g5"] },
    { t: "两本账", d: "每 Stage 在飞激活从 GPipe 的 M 份降到 min(S−s, M) 份；Bubble 上界不变——1F1B 先解决「放得下」，其次才是速度。", g: ["g6"] },
  ],
  ch02: [
    { t: "注意力三步", d: "一层注意力只做三件事：出题——用当前 Token 算出 Query；配对打分——Q 和历史每个 K 点积、softmax 归一成权重；取内容——按权重加权 V。K 是「标签」，V 是「内容」，Q 是检索词。", g: ["g1"] },
    { t: "没有 Cache 的账", d: "手写推理写第 21 个字时，前 20 个位置的 Q/K/V 全部重算（28 层 × 20 位置）。但历史 Token 的 K/V 只依赖「过去」、不随「现在写谁」变——重复计算它们是纯浪费。", g: ["g2"] },
    { t: "Prefill 一次读完", d: "第一步 Prefill：19 个输入位置并行过 28 层，只需 34.43 ms（3080 Ti、bf16 实测）。这一次前向顺手把每层每头的 K/V 存进 Cache，并产出首 Token。", g: ["g3"] },
    { t: "逐 Token 写/读 KV", d: "之后每步 Decode 只算 1 个新位置的 Q/K/V：K/V 追加进 Cache，Q 拿去和全部历史 K 打分、按权重取 V——历史不再重算，Cache 形状 [batch, 8, seq_len, 128]。", g: ["g4", "g5"] },
    { t: "两阶段延迟差", d: "读 19 个字只要 34.43 ms（并行），「看一遍历史再写一个字」平均 33.40 ms（串行读全部历史）；32 字端到端 1069.80 ms。算得少、读得多——Decode 瓶颈从算力转向显存带宽。", g: ["g3", "g4", "g5", "g6"] },
    { t: "显存账本", d: "Cache 数据量 = 2 × 28 层 × 8 KV 头 × 128 头维 × 2 字节（bf16）≈ 0.11 MB/Token（K 与 V 各一份）；1024 Token Prompt 生成到 1055 长度实测 115.39 MiB，与公式吻合。", g: ["g7"] },
  ],
  ch08: [
    { t: "① 链式哈希匹配", d: "Prompt 按 Block 切开后，沿链式哈希 SHA256(model_namespace‖父块摘要‖块 Token IDs) 逐块查找，取最长连续命中段。哈希必须绑定父块：相同 Token 块在不同历史上下文算出的 K/V 不同，单独哈希会误判命中。", g: ["g1", "g4"] },
    { t: "② 命中块引用", d: "命中的都是已填满并宣布不可变的物理块（ch05 Block Pool 里那批），新请求把引用加进自己的 Block Table、引用计数 +1 即可，一个字节不重算；未填满尾块与 Decode 块始终私有，无需 Copy-on-Write。", g: ["g2"] },
    { t: "③ 只算后缀", d: "Prefill 只做未命中的 Prompt 后缀。隐蔽边界：KV 不保存首 Token Logits，最多复用 prompt_length−1 个 Token，至少重算最后一个 Prompt Token 才能拿到完整 Logits。", g: ["g3"] },
    { t: "释放两本账", d: "请求结束只减请求引用；缓存索引仍持有的块不回空闲池。池满时按 leaf-LRU 从链的叶子端最久未用块开始淘汰，只有缓存所有权已移除且引用计数归零的块才安全复用。", g: ["g5"] },
    { t: "对照：全量重算", d: "disabled 基线：8 条请求并行各算完整 2048 个位置，Prefill Token 共 16896，热请求 TTFT p50 高达 384.11ms——算力全花在重复计算同一段已知前缀上。", g: ["g6"] },
    { t: "首条冷建缓存", d: "enabled 下第一条请求没有可命中的缓存，仍整条算 2048 个位置；算完的满块按哈希入缓存索引并宣布不可变，供后续请求复用——缓存是首条请求顺手建起来的。", g: ["g7"] },
    { t: "热请求的账", d: "请求②~⑧ 命中全部 2048 位置、引用复用，只算 64 个后缀位置：Prefill Token 16896→2560（−84.85%），TTFT p50 384.11→52.64ms（−86.3%）。但 Makespan 只降 11.33%——Decode 分毫不省，缓存池还多占 280 MiB 显存。", g: ["g8", "g9"] },
  ],
  ch09: [
    { t: "装载 tile", d: "Q 常驻片上不回显存，K/V 按 BLOCK_M×BLOCK_N 的 tile 分块从显存搬进来算、算完丢弃。中间矩阵不落显存，省的就是这一来一回的搬运。", g: ["g1"] },
    { t: "tile 局部打分", d: "对当前块算局部 Score 矩阵并套因果 Mask——这个 [seq,seq] 级别的矩阵只在片上活一小块，全程不写显存，平方级中间量就地消失。", g: ["g2", "g5"] },
    { t: "在线修正 m/l/acc", d: "取行最大值更新 m_new = max(m, rowmax(scores_tile))；出现更大 max 时用 alpha = exp(m − m_new) 把之前的累加结果换算到新基准——这就是 Online Softmax 的边收边修正。", g: ["g3"] },
    { t: "累加与归一", d: "p = exp(scores_tile − m_new)，acc = alpha×acc + p@V，l 同步修正累加；K/V 块全部扫完后 acc/l 归一输出——每行只留 m、l、acc 三个状态量。", g: ["g4"] },
    { t: "eager 的显存账", d: "朴素写法把 Score 和 Probability 按 [seq,seq] 完整写显存再读回：4096 Token 增量峰值 2576.1 MiB，8192 Token 直接 OOM——中间矩阵随序列平方增长。", g: ["g6"] },
    { t: "flash 的显存账", d: "只需要几百 KB 的 tile 加每行 3 个状态量：4096 Token 峰值 16.0 MiB，8192 Token 15.79ms 正常完成。FLOPs 一次没少，省的全是搬运。", g: ["g7"] },
  ],
  ch12: [
    { t: "抢占：暂停受害请求", d: "乐观准入把池打满，显存不够时暂停最后接纳的 running 请求当 victim，腾出它的 KV Block。池容量、准入水位、受害者选择两条路径完全相同，只在恢复一步分岔。", g: ["g1"] },
    { t: "recompute：丢弃重算", d: "直接把受害请求的 KV Block 释放回池，恢复时把全部已有内容当成新 Prompt 重新 Prefill。3080 Ti 实测额外 Prefill 暴露约 673–764 ms，明显贵于搬运。", g: ["g2"] },
    { t: "swap：整请求换出", d: "把受害者整请求 KV 按块主序拼成 contiguous 数据，经 PCIe 拷进预分配的 Pinned CPU 池。Qwen3-0.6B 单 Token KV 约 112 KiB，2048 Token 换出实测 19.67 ms。", g: ["g3"] },
    { t: "Pinned 池：启动预分配", d: "副本放锁页内存，CPU 不换页才能跑满 PCIe（实测约 11.9 GB/s）。池必须启动时建好——现场申请 448 MiB pinned 要约 102 ms，比一次换出还贵。", g: ["g4"] },
    { t: "恢复：断点续算", d: "swapped 队列 FCFS 优先于 waiting；轮到时按 block 表把 KV 拷回 GPU 新块、重建映射，从暂停处的下一 Decode 步续算。池 320 时 swap 吞吐 44.9 vs 保守准入 40.7。", g: ["g5"] },
  ],
  ch15: [
    { t: "拆分依据：两种负载", d: "Prefill 是一次并行算完几十上百个位置的大活、吃算力；Decode 是每步算 1 个新位置、读全部历史 K/V 的串行细活、吃带宽。挤在同一张卡上，长 Prompt 的 Prefill 会占满算力，同卡 Decode 的 ITL 出现尖刺，扩容也说不清该按哪种负载买卡。", g: ["g1"] },
    { t: "P 实例：只做 Prefill", d: "接单 Worker 只做 Prefill：并行算完整个 Prompt 的所有位置、建好 KV、顺手产出首 Token，随后它对这个请求的任务结束。但交接有单价——这个请求的 KV Payload 有 224 MiB，搬运比 Prefill 本身还贵。", g: ["g2"] },
    { t: "KV 走同步通道交接", d: "按 Request ID + Attempt ID 一次性交接：KV 用逻辑布局（连续、按层）导出，不暴露进程内物理 Block ID，走 GPU→CPU→TCP→CPU→GPU 发到 Decode Worker。实测阶段中位数 Export 约 433 ms、Publish 约 896 ms、Receive 约 358 ms、Import 约 253 ms。", g: ["g3"] },
    { t: "D 实例：ACK 后接管", d: "Decode 校验 Payload、分配本地 Block、H2D 导入后回 ACK；Producer 收到 ACK 才能释放本地 KV，首 Token 也等 ACK 之后才外发（ACK-gated）。Payload 损坏或 Namespace 不匹配时发 fallback ACK，退回完整重算先保正确。", g: ["g4"] },
    { t: "算总账：单请求负优化", d: "3080 Ti、Qwen3-0.6B、2048 Token Prompt 实测：KV 224 MiB 走两次 socket，Publish+Receive 约 1.25 秒；mono TTFT 433.39 ms vs split 2378.55 ms（5.49×），端到端 685.59 vs 2648.27 ms。P/D 分离卖的是各自扩容互不干扰，不是单请求更快。", g: ["g1", "g2", "g3", "g4"] },
  ],
  ch23: [
    { t: "主权重 FP32", d: "0.6B 模型的主权重约 2.4 GB，以 FP32 常驻显存、原件从不被 autocast 改写——它是优化器更新与数值稳定性的锚。", g: ["g1"] },
    { t: "造 BF16 副本", d: "进入 torch.autocast(bfloat16) 区间后，算子用到的参数被现造一份 BF16 副本——只是给 GEMM 用的临时拷贝，不是原地降精度。", g: ["g2"] },
    { t: "GEMM 走 Tensor Core", d: "QKV、O 投影、FFN 这些 GEMM 的输入自动转 BF16，吃 Tensor Core 的低精度算力红利；区间内激活和注意力输出也是 BF16。", g: ["g3"] },
    { t: "反向梯度 cast 回", d: "反向在区间内算出的梯度是 BF16，出区间即 cast 回 FP32 累积——dtype 普查用 forward hook 逐模块记录，「变了没变」是测出来的。", g: ["g4"] },
    { t: "AdamW 更新 FP32", d: "FP32 梯度喂给 AdamW：动量 m、v 两份状态全 FP32，小步更新不会被 BF16 舍入吃掉——优化器的数学一个字没动。", g: ["g5"] },
    { t: "无需 GradScaler", d: "BF16 的指数位与 FP32 同多、动态范围一样大，不会像 FP16 那样小梯度下溢——GradScaler 是 FP16 专属，这里不需要。", g: ["g6"] },
    { t: "算一算读账", d: "主权重 2.4 + 梯度 2.4 + 两份动量 4.8 ≈ 9.6 GB 状态纹丝不动；省下的是激活显存约减半——混合精度省的是「算」不是「存」。", g: ["g7"] },
  ],
  ch24: [
    { t: "baseline：一次大批", d: "baseline 一次大 Batch 完成 Forward/Backward 后立刻 Step，Loss 用 token_mean——全部监督 Token 等权平均。这是拆成 K 个微批后仍要对齐的口径。", g: ["g1"] },
    { t: "① 切片", d: "make_step_plan 用同一 shuffle 把有效 Batch 按 16 分组、再按微批尺寸切片，各臂吃到完全相同的样本序列；每次显存只按微批算——容量问题变成时间问题。", g: ["g2"] },
    { t: "② 逐微批加权", d: "第 i 个微批的 Loss 按 token_sum_i/total_supervised 缩放后 Backward：两个微批监督 Token 100 和 300、总 400，就分别乘 0.25 和 0.75——监督多的权重大，才与联合 token_mean 同口径。", g: ["g3"] },
    { t: "累积期：不 Step", d: "PyTorch 的梯度默认累加：未满 K 次时每个微批的 Backward 只往 .grad 上加，不更新参数、不清零——图中虚线回路就是这条微批循环。", g: ["g5"] },
    { t: "③ 一步优化器", d: "第 K 个微批算完才执行一次全局裁剪 + 一次 AdamW Step——累积梯度与一次大 Batch 严格同口径（FP32 下只差求和顺序噪声）；Step 后梯度清零，回到下一轮累积。", g: ["g4"] },
    { t: "代价与收益", d: "K 次小 GEMM 的时间加总，换来微批级显存上限：大批 OOM 的配置能跑。反例已验证「每个微批取均值再简单平均」在监督 Token 不均时会偏离联合目标。", g: ["g6"] },
  ],
  ch29: [
    { t: "单卡基线", d: "baseline 一张卡吃下全局 Batch，前向、反向、更新一气呵成，无任何卡间通信。它就是 variant 必须逐位对齐的正确性参照：对照要求 8 步轨迹与跨 Rank 逐位一致。", g: ["g1"] },
    { t: "持全量 + 分片", d: "每张卡持一份完整的模型+梯度+优化器状态副本——所以 DDP 只治吞吐、治不了装不下。全局批次 4 条按 rank::world_size 切片，两卡各拿 2 条。", g: ["g2"] },
    { t: "前向 + 反向", d: "各卡对本地分片正常 Forward/Backward，先不碰对方。此时两边梯度已经不同——参数更新会分叉，唯一解法是把梯度合并成同一个再更新。", g: ["g2"] },
    { t: "AllReduce SUM", d: "Backward 结束后一次 AllReduce SUM，人人拿到同一份求和结果。Ring 两阶段各 W−1 步，通信量 2(W−1)/W·D；W=2 时系数为 1，每卡发出约 2.4 GB 走 PCIe PHB，16 GB/s 下理想也要约 150 ms——通信全额暴露。", g: ["g3"] },
    { t: "除均 + 更新", d: "本地损失按 supervised_local/supervised_global 占比缩放后，求和结果与单卡大 Batch 的 token_mean 严格等价；各卡拿同一份梯度执行 optimizer step，跨 Rank 逐位一致，训练语义不变。", g: ["g4"] },
  ],
  ch34: [
    { t: "起点：ZeRO-2", d: "baseline 每卡全量参数常驻，Forward/Backward 照常算，只在步末一次 AllGather 刷新；每卡常驻模型状态 2N+14N/W 字节（FP32 口径）——参数这份 2N 在两步之间整份占着显存。", g: ["g1"] },
    { t: "参数分片常驻", d: "ZeRO-3/FSDP 让训练参数平时也只以 Shard 形式躺在每 Rank，用的时候再临时拼全量；参数、梯度、状态全按 1/W 等分，每卡常驻模型状态降到 16N/W 字节。", g: ["g2"] },
    { t: "forward 前 AllGather", d: "forward 需要参数时，在 autograd 不可见的 no-grad 区里 AllGather 把各卡分片临时拼成全量权重；这次通信从「每步一次」搬进了热路径，排队就在主流程里。", g: ["g3"] },
    { t: "算完即 Reshard", d: "本单元的前向计算结束后真正释放全量存储、退回分片状态——不是「标记可回收」；显存里参数重新只剩本 Rank 的 1/W。", g: ["g4"] },
    { t: "backward 前再 AllGather", d: "权重梯度要用全量权重，按需再拼一次；算出的权重梯度当场 Reduce-Scatter，只有本 Rank 的分片落进 .grad。", g: ["g5"] },
    { t: "分片 Optimizer Step", d: "optimizer 与 ZeRO-1/2 相同，只持本分片；下一个 forward 的 AllGather 本身就是参数刷新，post-step AllGather 被取消。W=4、N=0.6B 时每卡约 2.4GB，对照 ZeRO-2 的约 3.3GB。", g: ["g6"] },
  ],
  ch39: [
    { t: "serial：接力等待", d: "不调度时一个 micro 走完所有 Stage 才轮到下一个，任一时刻只有一张卡在算，S 张卡利用率只有 1/S——模型按层放下了，流水却白切。", g: ["g1"] },
    { t: "Fill：连推 Forward", d: "micro 1 的分段激活沿 Stage 0→1→…→S−1 的 P2P 流下去；Stage 0 不等它走完全流水，紧接着发 micro 2 的分段——流水被逐段错峰填满。", g: ["g2"] },
    { t: "激活驻留代价", d: "serial 每 Stage 同一时刻只驻留 1 个 micro 的激活图，GPipe 要驻留 M 份；README 把这条峰值显存差列为机制性结论——压 Bubble 的显存代价先在这里记账。", g: ["g4"] },
    { t: "Drain：梯度逆流排空", d: "末 Stage 从 loss 反传；其余 Stage 收到自己输出激活的输入梯度后 autograd.backward(out, g) 启动本段反传，把输入激活的 .grad 经 P2P 传回上一 Stage。", g: ["g3"] },
    { t: "Bubble：micro 换效率", d: "填充-排空模型给出 Bubble 上界 (S−1)/(M+S−1)：PP=2、M=8 约 1/9≈9%，M=2 却是 1/3——micro 开大压 Bubble，代价正是驻留 M 份激活。", g: ["g5"] },
  ],
};

const built = sections.map(sec => {
  let md = body(sec.file);
  if (sec.svg) md = `${inlineSvg(sec.svg)}\n\n` + md;
  if (sec.exit) md += `\n\n<div class="exit">\n\n${body(sec.exit)}\n\n</div>\n`;
  let html = marked.parse(md);
  html = html.replace(/<h3>(ch\d+[^<]*)<\/h3>/g, (m, t) => {
    const id = (t.match(/ch\d+/) || [t])[0];
    return `<h3 id="${sec.id}-${id}">${t}</h3>`;
  });
  html = fields(html);
  html = players(html);
  return { ...sec, html };
});

function fields(html) {
  // 名词卡 = 标签段 + 紧随的表格
  html = html.replace(/<p><strong>名词卡<\/strong>：?\s*<\/p>\s*(<table>[\s\S]*?<\/table>)/g,
    (m, tbl) => `<section class="f fld-terms"><span class="fhead">名词卡</span>${tbl}</section>`);
  const re = /<p><strong>(算一算|数字|边界与常见误区|产品对应|方案视角|讲给客户的一句话|掌握目标|延伸阅读|深挖入口|重点)<\/strong>：([\s\S]*?)<\/p>/g;
  html = html.replace(re, (m, label, rest) => {
    const cls = FIELD_CSS[label];
    return `<section class="f fld-${cls}"><span class="fhead">${label}</span>${rest}</section>`;
  });
  return html;
}

// —— 步骤播放器：<!--player:chNN--> 换成 带按钮的图解播放组件 ——
function players(html) {
  return html.replace(/<!--player:(ch\d+)-->/g, (m, ch) => {
    const steps = STEP_CONFIGS[ch];
    if (!steps) return "";
    const fig = inlineSvg(`fig/${ch}.svg`, false);
    const btns = steps.map((s, i) => `<button class="pbtn" data-step="${i}">${i + 1}</button>`).join("");
    return `<div class="player" id="player-${ch}">
<figure class="archwrap">${fig}</figure>
<div class="pbtns">${btns}<button class="pbtn pplay" data-all="1">▶ 全流程</button></div>
<div class="pdesc" id="pd-${ch}">点上面的数字按钮，图上对应环节会亮起；或点「▶ 全流程」自动播放。</div>
</div>`;
  });
}

// 侧边栏：按 group 分组；章节芯片指向各节 h3
const groups = [];
for (const sec of built) {
  let g = groups.find(x => x.name === sec.group);
  if (!g) { g = { name: sec.group, items: [] }; groups.push(g); }
  const chapters = [...sec.html.matchAll(/<h3 id="([^"]+)">([^<]+)<\/h3>/g)]
    .map(m => ({ id: m[1], n: (m[2].match(/ch\d+/) || [m[2]])[0] }));
  g.items.push({ sec, chapters });
}

const chip = (c) => `<a class="chip" href="#${c.id}" title="${c.n}"><span>${c.n}</span></a>`;

const sidebar = groups.map(g => `
  <div class="group"><div class="gtitle">${g.name}</div>
  ${g.items.map(it => `
    <a href="#${it.sec.id}" class="top" data-sec="${it.sec.id}">${it.sec.title}</a>
    ${it.chapters.length ? `<div class="chips">${it.chapters.map(chip).join("")}</div>` : ""}
  `).join("")}
  </div>`).join("\n");

const main = built.map(sec => `<section id="${sec.id}">${sec.html}</section>`).join("\n<hr/>\n");

const htmlDoc = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Infra 架构师学习指南 · miniinfra 43 章导读</title>
<meta name="description" content="以架构师/售前视角导读 miniinfra 43 章：基础概念、推理与训练系统全景、容量测算、选型算例、售前话术。"/>
<style>
:root{ --bg:#fafaf9; --surface:#ffffff; --fg:#1c1917; --muted:#78716c; --border:#e7e5e4; --accent:#0f766e; --accent-soft:#ccfbf1; --warning:#b45309; --code:#292524; }
@media (prefers-color-scheme: dark){ :root{ --bg:#17150f; --surface:#211e18; --fg:#e7e5e4; --muted:#a8a29e; --border:#3f3a33; --accent:#2dd4bf; --accent-soft:#134e4a; --warning:#fbbf24; --code:#0c0a09; } }
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font:15.5px/1.78 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
#progress{position:fixed;top:0;left:0;height:2.5px;background:var(--accent);width:0;z-index:50;transition:width .15s linear}
.layout{display:grid;grid-template-columns:272px 1fr;max-width:1260px;margin:0 auto}
aside{position:sticky;top:0;height:100vh;overflow-y:auto;padding:26px 16px 40px;border-right:1px solid var(--border);font-size:13.5px;scrollbar-width:thin}
aside .brand{font-weight:750;font-size:15.5px;margin-bottom:2px}
aside .brandsub{color:var(--muted);font-size:11.5px;margin-bottom:20px}
aside .group{margin-bottom:18px}
aside .gtitle{font-size:11px;letter-spacing:.12em;color:var(--muted);margin:0 10px 6px;display:flex;align-items:center;gap:6px}
aside .gtitle::before{content:"";width:7px;height:7px;border-radius:2px;background:var(--accent);opacity:.85}
aside ul{list-style:none;margin:0;padding:0}
aside a{color:var(--muted);text-decoration:none;display:block;padding:6px 10px;border-radius:8px;transition:background .15s,color .15s}
aside a.top{color:var(--fg);font-weight:620}
aside a:hover{background:var(--accent-soft)}
aside a.active{background:var(--accent-soft);color:var(--accent);font-weight:650}
aside .chips{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin:4px 6px 8px}
aside .chips a{padding:3px 0;text-align:center;font-size:11px;border:1px solid var(--border);border-radius:6px;color:var(--muted);background:var(--surface)}
aside .chips a:hover{border-color:var(--accent);color:var(--accent)}
aside .chips a.active{background:var(--accent);border-color:var(--accent);color:var(--bg)}
aside::-webkit-scrollbar{width:6px}
aside::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
main{padding:44px 44px 90px;min-width:0}
section{max-width:880px}
hr{border:none;border-top:1px solid var(--border);margin:60px auto;max-width:880px}
h2{font-size:27px;line-height:1.35;margin:0 0 18px;letter-spacing:.2px}
h3{font-size:20px;margin:46px 0 12px;padding-top:22px;border-top:1px solid var(--border)}
h4{font-size:16.5px;margin:26px 0 8px}
.exit{background:var(--accent-soft);border-radius:12px;padding:6px 24px 18px;margin:36px 0}
.exit h1{font-size:18px}
p{margin:10px 0}
li{margin:4px 0}
blockquote{margin:14px 0;padding:10px 16px;border-left:3px solid var(--accent);background:var(--surface);border-radius:0 10px 10px 0;color:var(--muted)}
blockquote p{margin:4px 0}
code{font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--surface);border:1px solid var(--border);padding:1.5px 6px;border-radius:6px}
pre{background:var(--code);color:#e7e5e4;padding:14px 18px;border-radius:12px;overflow-x:auto}
pre code{background:none;border:none;color:inherit;font-size:13px;line-height:1.6}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13.5px;background:var(--surface)}
th,td{border:1px solid var(--border);padding:7px 11px;text-align:left;vertical-align:top}
th{background:var(--accent-soft);font-weight:650;white-space:nowrap}
figure.archwrap{margin:18px 0 22px;padding:16px 18px;background:var(--surface);border:1px solid var(--border);border-radius:14px}
svg.arch{width:100%;height:auto;display:block}
a{color:var(--accent)}
/* —— 字段卡片 —— */
.f{margin:16px 0;padding:12px 16px;border-radius:12px;background:var(--surface);border:1px solid var(--border)}
.f p{margin:6px 0 2px}
.fhead{display:inline-block;font-size:11.5px;font-weight:700;letter-spacing:.06em;padding:2px 10px;border-radius:999px;background:var(--accent-soft);color:var(--accent);margin-bottom:6px}
.fld-terms{border-left:3px solid var(--accent)}
.fld-terms table{margin:8px 0 2px}
.fld-calc{border-left:3px solid var(--accent)}
.fld-calc code{background:var(--accent-soft);border-color:transparent}
.fld-nums{border-left:3px solid var(--muted)}
.fld-warn{border-left:3px solid var(--warning)}
.fld-warn .fhead{background:transparent;color:var(--warning);border:1px solid var(--warning)}
.fld-key{border-left:3px solid var(--accent);background:var(--accent-soft)}
.fld-prod,.fld-plan{border-left:3px solid var(--accent)}
.fld-say{border-left:3px solid var(--accent);background:var(--accent-soft)}
.fld-say{border-radius:0 14px 14px 14px;font-weight:550}
.fld-goal .fhead{background:var(--accent);color:var(--bg)}
.fld-read a{color:var(--accent)}
.fld-read .fhead{background:var(--surface);border:1px solid var(--border);color:var(--muted)}
.fld-dig{border-left:3px solid var(--border);color:var(--muted);font-size:13px;background:transparent}
.fld-dig .fhead{background:transparent;border:1px solid var(--border);color:var(--muted)}
/* —— 步骤播放器 —— */
.player{border:1px solid var(--border);border-radius:14px;padding:12px 12px 6px;background:var(--surface);margin:18px 0 22px}
.player figure.archwrap{margin:0 0 10px;border:none;background:transparent}
.pbtns{display:flex;flex-wrap:wrap;gap:6px}
.pbtn{font:600 12px ui-monospace,Menlo,monospace;padding:5px 12px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--muted);cursor:pointer}
.pbtn:hover{border-color:var(--accent);color:var(--accent)}
.pbtn.on{background:var(--accent);border-color:var(--accent);color:var(--bg)}
.pdesc{margin:10px 2px 8px;font-size:13.5px;color:var(--fg);min-height:40px;line-height:1.6}
@media (max-width:960px){ .layout{grid-template-columns:1fr} aside{position:static;height:auto;border-right:none;border-bottom:1px solid var(--border);padding:14px} main{padding:24px 16px 70px} hr{margin:44px auto} aside .chips{grid-template-columns:repeat(6,1fr)} }
</style>
</head>
<body>
<div id="progress"></div>
<div class="layout">
<aside>
  <div class="brand">AI Infra 架构师学习指南</div>
  <div class="brandsub">miniinfra 43 章导读 · 架构/售前视角</div>
${sidebar}
</aside>
<main>
${main}
<footer style="margin-top:70px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--border);padding-top:16px">
章节骨架与机制事实来自 SVDX-lab/miniinfra（原作者：小红书 @因式分解X）；讲解层为独立撰写，短引用已注明。原课程版权归原作者所有。生成于 2026-10-18。
</footer>
</main>
</div>
<script>window.PLAYERS=${JSON.stringify(STEP_CONFIGS)}</script>
<script>
// 顶部进度条
const bar = document.getElementById("progress");
addEventListener("scroll", () => {
  const h = document.documentElement;
  bar.style.width = (h.scrollTop / (h.scrollHeight - h.clientHeight) * 100) + "%";
}, { passive: true });
// 深链接兜底：带 hash 打开时若浏览器未完成滚动，手动滚到锚点
window.addEventListener("load", () => {
  if (!location.hash) return;
  const el = document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (el) setTimeout(() => el.scrollIntoView({ behavior: "instant", block: "start" }), 50);
});
// 滚动高亮：当前 section 的侧边入口点亮
const secs = [...document.querySelectorAll("main section[id]")];
const links = [...document.querySelectorAll("aside a[data-sec]")];
const spy = new IntersectionObserver((es) => {
  es.forEach(e => { if (e.isIntersecting) {
    links.forEach(a => a.classList.toggle("active", a.dataset.sec === e.target.id));
  }});
}, { rootMargin: "-20% 0px -60% 0px" });
secs.forEach(s => spy.observe(s));
// 步骤播放器：通用版——所有 .player 共用，步骤数据在 window.PLAYERS（构建期注入）
(function () {
  document.querySelectorAll(".player").forEach(function (player) {
    const ch = player.id.replace("player-", "");
    const svg = player.querySelector("svg");
    const grps = [...svg.querySelectorAll("g[id]")];
    const desc = document.getElementById("pd-" + ch);
    const DEFAULT = "点上面的数字按钮，图上对应环节会亮起；或点「▶ 全流程」自动播放。";
    const STEPS = window.PLAYERS[ch] || [];
    let timer = null;
  function show(i) {
    const on = new Set(STEPS[i].g);
    grps.forEach(g => {
      g.classList.toggle("off", !on.has(g.id));
      g.classList.toggle("on", on.has(g.id));
    });
    desc.innerHTML = "<b>" + (i + 1) + ". " + STEPS[i].t + "</b> — " + STEPS[i].d;
    player.querySelectorAll(".pbtn[data-step]").forEach(b =>
      b.classList.toggle("on", +b.dataset.step === i));
  }
  function clear() {
    grps.forEach(g => { g.classList.remove("off", "on"); });
    desc.textContent = DEFAULT;
    player.querySelectorAll(".pbtn[data-step]").forEach(b => b.classList.remove("on"));
  }
  player.querySelectorAll(".pbtn[data-step]").forEach(b =>
    b.addEventListener("click", () => show(+b.dataset.step)));
  const play = player.querySelector(".pplay");
  play.addEventListener("click", () => {
    if (timer) { clearInterval(timer); timer = null; play.classList.remove("on"); clear(); return; }
    play.classList.add("on");
    let i = 0; show(0);
    timer = setInterval(() => { i = (i + 1) % STEPS.length; show(i); }, 2600);
  });
  });
})();
</script>
</body>
</html>`;

fs.writeFileSync(OUT, htmlDoc);
console.log("built:", OUT, `${htmlDoc.length} bytes`);
console.log("h3 cards:", (htmlDoc.match(/<h3 id=/g) || []).length,
  "| arch figures:", (htmlDoc.match(/<figure class="archwrap">/g) || []).length,
  "| tables:", (htmlDoc.match(/<table>/g) || []).length,
  "| sidebar groups:", groups.length);
