# 训练篇出口（白板题）

1. 白板写出一个 SFT step 的完整数据流：Chat Template → Tokenize → labels/-100 掩码 → Forward → Shifted Cross Entropy → Backward → 裁剪 → AdamW step → Checkpoint（ch16–18）。
2. 推导梯度累积的 loss 缩放：为什么微批 i 必须按 `token_sum_i / total_supervised` 加权（ch24）？
3. 一句话区分 ZeRO-1/2/3，并写出常驻显存从 16 字节/参数到 16N/W 量级的账（ch32–34）。
4. 给定 7B 模型、8×80GB：推导各组件显存，决定 ZeRO 级别，说明 TP/PP 还差什么（表 B + ch35–42）。
5. 训练中断恢复需要哪些状态？分片保存 + manifest 的恢复语义强在哪（ch43）？
