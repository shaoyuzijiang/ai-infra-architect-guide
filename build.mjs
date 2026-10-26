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
