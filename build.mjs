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
  { id: "capacity", title: "容量与成本工程", group: "工具箱", file: "capacity.md", svg: null, exit: null },
  { id: "decisions", title: "决策速查表", group: "工具箱", file: "decisions.md", svg: null, exit: null },
  { id: "products", title: "产品与方案速查", group: "工具箱", file: "products.md", svg: null, exit: null },
  { id: "examples", title: "选型算例库", group: "工具箱", file: "examples.md", svg: null, exit: null },
  { id: "faq", title: "售前/答辩 FAQ", group: "工具箱", file: "faq.md", svg: null, exit: null },
  { id: "index", title: "原课程索引", group: "附录", file: "course-index.md", svg: null, exit: null },
];

function inlineSvg(rel) {
  let s = fs.readFileSync(path.join(SRC, rel), "utf8");
  s = s.replace(/<\?xml[^>]*\?>/g, "").trim();
  s = s.replace(/\n\s*\n/g, "\n");                       // 压缩空行：HTML block 以空行为界，svg 内不能有
  s = s.replace(/<svg /, '<svg class="arch" ');
  return `<figure class="archwrap">${s}</figure>`;
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

const built = sections.map(sec => {
  let md = body(sec.file);
  if (sec.svg) md = `${inlineSvg(sec.svg)}\n\n` + md;
  if (sec.exit) md += `\n\n<div class="exit">\n\n${body(sec.exit)}\n\n</div>\n`;
  let html = marked.parse(md);
  html = html.replace(/<h3>(ch\d+[^<]*)<\/h3>/g, (m, t) => {
    const id = (t.match(/ch\d+/) || [t])[0];
    return `<h3 id="${sec.id}-${id}">${t}</h3>`;
  });
  return { ...sec, html };
});

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
</script>
</body>
</html>`;

fs.writeFileSync(OUT, htmlDoc);
console.log("built:", OUT, `${htmlDoc.length} bytes`);
console.log("h3 cards:", (htmlDoc.match(/<h3 id=/g) || []).length,
  "| arch figures:", (htmlDoc.match(/<figure class="archwrap">/g) || []).length,
  "| tables:", (htmlDoc.match(/<table>/g) || []).length,
  "| sidebar groups:", groups.length);
