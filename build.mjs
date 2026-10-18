// 构建脚本：把 src/*.md 拼装成单页 index.html（零外部依赖的成品页面）
// 用法：在 omp eval JS 内核里 `await import("<本文件绝对路径>")`，或任何支持 ESM 的 Bun/Node ≥18 环境：
//   bun run build.mjs   （需要 node_modules/marked，见 package.json）
import { marked } from "../node_modules/marked/lib/marked.esm.js";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(here, "src");
const OUT = path.join(here, "index.html");

const sections = [
  { id: "learn", title: "怎么学", file: "intro.md", svg: null, exit: null },
  { id: "inference", title: "推理篇 ch01–15", file: "inference.md", svg: "inference-arch.svg", exit: "exit-inference.md" },
  { id: "training", title: "训练篇上 ch16–29", file: "training.md", svg: "training-arch.svg", exit: null },
  { id: "training-b", title: "训练篇下 ch30–43", file: "training-b.md", svg: null, exit: "exit-training.md" },
  { id: "capacity", title: "容量与成本工程", file: "capacity.md", svg: null, exit: null },
  { id: "decisions", title: "决策速查表", file: "decisions.md", svg: null, exit: null },
  { id: "products", title: "产品与方案速查", file: "products.md", svg: null, exit: null },
  { id: "faq", title: "售前/答辩 FAQ", file: "faq.md", svg: null, exit: null },
  { id: "index", title: "原课程索引", file: "course-index.md", svg: null, exit: null },
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

const navItems = built.map(sec => ({
  id: sec.id, title: sec.title,
  chapters: [...sec.html.matchAll(/<h3 id="([^"]+)">([^<]+)<\/h3>/g)].map(m => ({ id: m[1], t: m[2] })),
}));

const sidebar = navItems.map(n => {
  const sub = n.chapters.length
    ? `<ul class="subnav">${n.chapters.map(c => `<li><a href="#${c.id}">${c.t.replace(/ · .*/, "")}</a></li>`).join("")}</ul>`
    : "";
  return `<li><a href="#${n.id}" class="top">${n.title}</a>${sub}</li>`;
}).join("\n");

const main = built.map(sec => `<section id="${sec.id}">${sec.html}</section>`).join("\n<hr/>\n");

const htmlDoc = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Infra 架构师学习指南 · miniinfra 43 章导读</title>
<meta name="description" content="以架构师/售前视角导读 miniinfra 43 章：推理与训练系统全景、容量测算、选型决策、售前话术。"/>
<style>
:root{ --bg:#fafaf9; --surface:#ffffff; --fg:#1c1917; --muted:#78716c; --border:#e7e5e4; --accent:#0f766e; --accent-soft:#ccfbf1; --warning:#b45309; --code:#292524; }
@media (prefers-color-scheme: dark){ :root{ --bg:#17150f; --surface:#211e18; --fg:#e7e5e4; --muted:#a8a29e; --border:#3f3a33; --accent:#2dd4bf; --accent-soft:#134e4a; --warning:#fbbf24; --code:#0c0a09; } }
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font:15.5px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
.layout{display:grid;grid-template-columns:264px 1fr;max-width:1240px;margin:0 auto}
aside{position:sticky;top:0;height:100vh;overflow-y:auto;padding:28px 18px;border-right:1px solid var(--border);font-size:13.5px}
aside .brand{font-weight:700;font-size:15px;margin-bottom:4px}
aside .brandsub{color:var(--muted);font-size:12px;margin-bottom:18px}
aside ul{list-style:none;margin:0;padding:0}
aside li{margin:2px 0}
aside a{color:var(--muted);text-decoration:none;display:block;padding:5px 10px;border-radius:7px}
aside a.top{color:var(--fg);font-weight:600}
aside a:hover{background:var(--accent-soft)}
aside ul.subnav{margin-left:10px;border-left:1px solid var(--border);padding-left:6px}
aside ul.subnav a{padding:2.5px 8px;font-size:12.5px;color:var(--muted)}
main{padding:44px 44px 90px;min-width:0}
section{max-width:860px}
hr{border:none;border-top:1px solid var(--border);margin:60px auto;max-width:860px}
h2{font-size:26px;line-height:1.35;margin:0 0 18px}
h3{font-size:19px;margin:44px 0 12px;padding-top:20px;border-top:1px solid var(--border)}
.exit{background:var(--accent-soft);border-radius:12px;padding:4px 22px 18px;margin:36px 0}
.exit h1{font-size:18px}
p{margin:10px 0}
blockquote{margin:14px 0;padding:10px 16px;border-left:3px solid var(--accent);background:var(--surface);border-radius:0 10px 10px 0;color:var(--muted)}
blockquote p{margin:4px 0}
code{font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--surface);border:1px solid var(--border);padding:1.5px 6px;border-radius:6px}
pre{background:var(--code);color:#e7e5e4;padding:14px 18px;border-radius:12px;overflow-x:auto}
pre code{background:none;border:none;color:inherit;font-size:13px;line-height:1.6}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13.5px;background:var(--surface)}
th,td{border:1px solid var(--border);padding:7px 11px;text-align:left;vertical-align:top}
th{background:var(--accent-soft);font-weight:650;white-space:nowrap}
figure.archwrap{margin:20px 0;padding:18px;background:var(--surface);border:1px solid var(--border);border-radius:14px}
svg.arch{width:100%;height:auto;display:block}
a{color:var(--accent)}
@media (max-width:960px){ .layout{grid-template-columns:1fr} aside{position:static;height:auto;border-right:none;border-bottom:1px solid var(--border);padding:16px} aside ul.subnav{display:none} main{padding:26px 18px 70px} hr{margin:44px auto} }
</style>
</head>
<body>
<div class="layout">
<aside>
  <div class="brand">AI Infra 架构师学习指南</div>
  <div class="brandsub">miniinfra 43 章导读 · 架构/售前视角</div>
  <ul>
${sidebar}
  </ul>
</aside>
<main>
${main}
<footer style="margin-top:70px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--border);padding-top:16px">
章节骨架与机制事实来自 SVDX-lab/miniinfra（原作者：小红书 @因式分解X）；讲解层为独立撰写，短引用已注明。原课程版权归原作者所有。生成于 2026-10-18。
</footer>
</main>
</div>
<script>
// 深链接兜底：带 hash 打开时若浏览器未完成滚动（长文档首帧布局未定），手动滚到锚点
window.addEventListener("load", () => {
  if (!location.hash) return;
  const el = document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (el) setTimeout(() => el.scrollIntoView({ behavior: "instant", block: "start" }), 50);
});
</script>
</body>
</html>`;

fs.writeFileSync(OUT, htmlDoc);
console.log("built:", OUT, `${htmlDoc.length} bytes`);
console.log("h3 cards:", (htmlDoc.match(/<h3 id=/g) || []).length,
  "| arch figures:", (htmlDoc.match(/<figure class="archwrap">/g) || []).length,
  "| tables:", (htmlDoc.match(/<table>/g) || []).length);
