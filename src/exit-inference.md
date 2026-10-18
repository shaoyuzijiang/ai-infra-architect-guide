# 推理篇出口（白板题）

不看资料，能画/能讲，本站算过关：

1. 画出一条请求的完整数据流：到达 → 调度 → 组批 → Cache 分配 → Attention → 首 Token → Decode，标出 ch02–ch15 每个优化作用在哪一段。
2. 讲清 PagedAttention 的动机与机制（ch05）：它消灭的是哪种浪费？教学实现为什么反而吞吐变低、生产价值在哪里？
3. 解释 ch02 的实测现象：16-token prompt 开 KV Cache 反而更慢、1024-token 才 1.85×——管理开销与计算节省怎么博弈？
4. 说明 ch11：为什么 greedy speculative decoding 在数学上不改变输出？Bonus Token 是什么？
