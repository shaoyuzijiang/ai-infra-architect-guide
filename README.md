# AI Infra 架构师学习指南

以**架构师/售前视角**导读 SVDX-lab/miniinfra（小红书 @因式分解X 的《超轻量 AI Infra 教程》）43 章：不需要跑实验，目标是机制心智模型、容量测算、选型决策与讲清楚的能力。

线上阅读（腾讯云 CloudBase，国内直连快）：`https://review-cloud-app-d7ebbb0b5df8d25-1307675462.tcloudbaseapp.com/`
（首次访问出现「页面访问提示」属 CloudBase 测试域名提示页，点「确定访问」即可；源码仓库：https://github.com/shaoyuzijiang/ai-infra-architect-guide ，GitHub Pages 镜像：https://shaoyuzijiang.github.io/ai-infra-architect-guide/ ）

## 结构

- 怎么学（材料三层分工：本站导读 / 原课程图文与教材 / 代码仓库）
- 推理篇 ch01–15：架构全景图 + 每章卡片（问题 / 机制通俗版 / 架构位置 / 业界对应 / 讲给客户的一句话）+ 出口白板题
- 训练篇 ch16–43：同上（ch30–43 帖子未发布，读仓库）
- 容量与成本工程：显存三账、KV 公式、token 成本、开刀顺序
- 决策速查表：swap vs recompute、ZeRO 分级、TP/PP 选择等四张表
- 售前/答辩 FAQ：14 条机制版答案
- 原课程索引：43 章 ↔ 帖子日期 ↔ 教材页码，跟更入口

## 来源与版权

章节骨架与机制事实来自 [SVDX-lab/miniinfra](https://github.com/SVDX-lab/miniinfra) 的公开 README 与原作者公开内容，短引用已注明出处；讲解层为独立撰写。原课程与教材版权归原作者所有，本仓库不复制教材原文。

## 重建

`src/` 是 Markdown 源稿与架构图 SVG；`index.html` 由 `marked` 拼装生成（见 src 同级的 build 说明，或直接以 `src/` 内容为准手工重建）。
