// Copyright (c) 2026 Jiejing Zhang.
//
// The page GET / serves: ask about an image, or keep an eye on a camera or
// screen until a condition you describe comes true.
//
// It is one string in the binary rather than a resource bundle because the
// CLI ships as a single executable (Homebrew installs one file), and it is
// served from the API's own origin so its calls need no CORS -- which this
// server deliberately does not send (OpenAIServer.route, GET /).
//
// The watch mode is Eyes On's pipeline (thinkspread/eyes-on, Watchpoint.swift) with
// the app taken out:
//
//   1 fps frame -> pixel check (identical screen: nothing to do) ->
//   POST /v1/vision/frame (tower only, never the LLM) -> cosine gate ->
//   settle -> "which of these rules holds?" -> that rule's action
//
// A rule is "when it sees X, say Y, and beep and notify / just log".
// For a camera the thresholds are Eyes On's measured ones: 0.025 to fire,
// 0.010 to call the scene settled; screens and video files get lower ones
// (GATES, below).  A 60 s backstop catches changes the gate cannot see
// (a progress bar, steam).  The page shows how many frames reached the
// model, because that ratio is the whole point of having a gate.

enum WebPage {
    /// Reach this origin and nothing else.  `worker-src blob:` is the tick
    /// worker (timers in a hidden tab's main thread are throttled to one a
    /// minute; a worker's are not).
    static let contentSecurityPolicy =
        "default-src 'none'; script-src 'unsafe-inline'; "
        + "style-src 'unsafe-inline'; img-src data: blob:; "
        + "media-src blob: mediastream:; connect-src 'self'; "
        + "worker-src blob:; base-uri 'none'; form-action 'none'; "
        + "frame-ancestors 'none'"

    static let html = #"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tempo9</title>
<style>
:root { --bg:#fafaf9; --fg:#1c1917; --mut:#78716c; --line:#e7e5e4; --card:#fff;
        --acc:#2563eb; --yes:#dc2626; --ok:#15803d; --soft:#f5f5f4; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0c0a09; --fg:#f5f5f4; --mut:#a8a29e; --line:#292524; --card:#1c1917;
          --acc:#60a5fa; --yes:#f87171; --ok:#4ade80; --soft:#292524; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
header { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
         padding: 12px 20px; border-bottom: 1px solid var(--line); }
header .name { font-weight: 650; font-size: 17px; margin-right: 6px; }
.pill { font-size: 12px; padding: 2px 10px; border-radius: 999px;
        border: 1px solid var(--line); color: var(--mut); white-space: nowrap; }
.pill.good { color: var(--ok); border-color: var(--ok); }
.pill.bad { color: var(--yes); border-color: var(--yes); }
nav { margin-left: auto; display: flex; gap: 6px; }
button { font: inherit; color: var(--fg); background: var(--soft); cursor: pointer;
         border: 1px solid var(--line); border-radius: 8px; padding: 6px 14px; }
button:hover { border-color: var(--mut); }
button.primary { background: var(--acc); border-color: var(--acc); color: #fff; }
button.on { background: var(--fg); color: var(--bg); border-color: var(--fg); }
button:disabled { opacity: .45; cursor: default; }
main { max-width: 1180px; margin: 0 auto; padding: 20px; display: grid; gap: 20px;
       grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
@media (max-width: 860px) { main { grid-template-columns: 1fr; padding: 16px; } }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
        padding: 16px; min-width: 0; }
.card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em;
           color: var(--mut); margin: 0 0 10px; font-weight: 600; }
.row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 10px 0; }
textarea, input[type=number] { font: inherit; color: var(--fg); background: var(--bg);
           border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
textarea { width: 100%; resize: vertical; min-height: 64px; }
input[type=number] { width: 84px; }
.drop { border: 1.5px dashed var(--line); border-radius: 10px; min-height: 220px;
        display: grid; place-items: center; text-align: center; color: var(--mut);
        overflow: hidden; background: var(--bg); }
.drop.over { border-color: var(--acc); }
.drop img, video { max-width: 100%; max-height: 420px; display: block; border-radius: 8px; }
video { width: 100%; background: #000; }
.answer { white-space: pre-wrap; min-height: 120px; font-size: 16px; }
.stats { font: 12px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--mut); }
.big { font: 600 28px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; }
.kv { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 8px 0 12px; }
.kv div { background: var(--soft); border-radius: 8px; padding: 8px 10px; }
.kv span { display: block; font-size: 12px; color: var(--mut); }
canvas { width: 100%; height: 150px; display: block; background: var(--bg);
         border: 1px solid var(--line); border-radius: 8px; }
.log { list-style: none; margin: 10px 0 0; padding: 0; max-height: 420px; overflow: auto; }
.log li { display: grid; grid-template-columns: 96px minmax(0, 1fr); gap: 10px;
          padding: 8px 0; border-top: 1px solid var(--line); }
.log img { width: 96px; border-radius: 6px; display: block; }
.log .yes { color: var(--yes); font-weight: 650; }
.log .why { font-size: 12px; color: var(--mut); }
.err { color: var(--yes); }
.hide { display: none !important; }
.rule { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 6px;
        margin: 8px 0; padding-bottom: 8px; border-bottom: 1px solid var(--line); }
.rule .when { grid-row: 1; grid-column: 1 / 3; }
.rule button { grid-row: 1; grid-column: 3; }
.rule .say { grid-row: 2; grid-column: 1; }
.rule .fx { grid-row: 2; grid-column: 2 / 4; }
.rule input, select { font: inherit; color: var(--fg); background: var(--bg); min-width: 0;
        border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px; }
.rule button { padding: 6px 10px; }
label { color: var(--mut); font-size: 13px; }
footer { max-width: 1180px; margin: 0 auto; padding: 0 20px 24px; color: var(--mut); font-size: 12px; }
</style>
</head>
<body>
<header>
  <span class="name">Tempo9</span>
  <span class="pill" id="model">…</span>
  <span class="pill" id="vision">vision ?</span>
  <span class="pill" id="net"></span>
  <nav><button id="tabAsk" class="on" data-t="tabAsk">Ask</button><button id="tabWatch" data-t="tabWatch">Keep an eye on</button>
       <button id="lang" title="中文 / English">中文</button></nav>
</header>

<main id="ask">
  <section class="card">
    <h2 data-t="hImage">Image</h2>
    <div class="drop" id="drop" data-t="drop">Paste a screenshot (⌘V), drop an image, or take one below</div>
    <div class="row">
      <button id="pick" data-t="pick">Choose file…</button>
      <button id="snapCam" data-t="camera">Camera</button>
      <button id="snapScreen" data-t="screen">Screen</button>
      <input type="file" id="file" accept="image/*" class="hide">
    </div>
    <textarea id="q">What is in this picture? Read any text in it exactly.</textarea>
    <div class="row"><button class="primary" id="go" disabled data-t="ask">Ask</button>
      <span class="stats" id="askNote"></span></div>
  </section>
  <section class="card">
    <h2 data-t="hAnswer">Answer</h2>
    <div class="answer" id="answer"></div>
    <div class="kv">
      <div><span data-t="kTtft">first token</span><b id="sTtft">–</b></div>
      <div><span data-t="kTps">decode</span><b id="sTps">–</b></div>
      <div><span data-t="kPrompt">prompt tokens</span><b id="sPrompt">–</b></div>
      <div><span data-t="kImg">image</span><b id="sImg">–</b></div>
    </div>
  </section>
</main>

<main id="watch" class="hide">
  <section class="card">
    <h2 data-t="hSource">Source</h2>
    <div class="row">
      <button id="srcCam" data-t="camera">Camera</button>
      <button id="srcScreen" data-t="screenOrWindow">Screen or window</button>
      <button id="srcFile" data-t="videoFile">Video file…</button>
      <input type="file" id="vfile" accept="video/*" class="hide">
    </div>
    <video id="vid" muted playsinline autoplay></video>
    <h2 style="margin-top:14px" data-t="hRules">When it sees… then</h2>
    <div id="rules"></div>
    <div class="row">
      <button id="addRule" data-t="addRule">+ Rule</button>
      <select id="preset">
        <option value="" data-t="presets">Presets…</option>
        <option value="build" data-t="pBuild">A build finishes or fails</option>
        <option value="door" data-t="pDoor">Someone comes to the door</option>
      </select>
      <label><input type="checkbox" id="fast"> <span data-t="fast">react at the first sign of change</span></label>
    </div>
    <div class="row">
      <button class="primary" id="start" disabled data-t="start">Start</button>
      <button id="stop" disabled data-t="stop">Stop</button>
    </div>
    <div class="row">
      <label><span data-t="fire">fire</span> <input type="number" id="fire" value="0.025" step="0.005" min="0"></label>
      <label><span data-t="settle">settle</span> <input type="number" id="settle" value="0.010" step="0.002" min="0"></label>
      <label><span data-t="backstop">backstop s</span> <input type="number" id="backstop" value="60" step="10" min="10"></label>
    </div>
  </section>
  <section class="card">
    <h2 data-t="hGate">Gate</h2>
    <div class="kv">
      <div><span data-t="kFrames">frames</span><b id="cFrames">0</b></div>
      <div><span data-t="kTower">tower</span><b id="cTower">0</b></div>
      <div><span data-t="kAsked">asked the model</span><b id="cAsked">0</b></div>
      <div><span data-t="kDist">distance</span><b id="cDist">–</b></div>
    </div>
    <canvas id="chart" width="1000" height="300"></canvas>
    <div class="stats" id="gateState" data-t="idle">idle</div>
    <ul class="log" id="log"></ul>
  </section>
</main>
<footer data-t="footer">Everything runs on this Mac. This page may only talk to the server that served it
(Content-Security-Policy connect-src 'self'), so nothing you show it leaves the machine.</footer>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
let model = null;

// ---------- language ----------
// English or Chinese, from the browser unless the viewer picked one.  Fixed
// text carries data-t; everything the script writes goes through t().
const STR = {
  en: { tabAsk: "Ask", tabWatch: "Keep an eye on", hImage: "Image",
    drop: "Paste a screenshot (⌘V), drop an image, or take one below", pick: "Choose file…",
    camera: "Camera", screen: "Screen", ask: "Ask", hAnswer: "Answer", kTtft: "first token",
    kTps: "decode", kPrompt: "prompt tokens", kImg: "image", hSource: "Source",
    screenOrWindow: "Screen or window", videoFile: "Video file…",
    hRules: "When it sees… then", addRule: "+ Rule", presets: "Presets…",
    pBuild: "A build finishes or fails", pDoor: "Someone comes to the door",
    fast: "react at the first sign of change", start: "Start", stop: "Stop", fire: "fire",
    settle: "settle", backstop: "backstop s", hGate: "Gate", kFrames: "frames", kTower: "tower",
    kAsked: "asked the model", kDist: "distance", idle: "idle",
    footer: "Everything runs on this Mac. This page may only talk to the server that served it " +
            "(Content-Security-Policy connect-src 'self'), so nothing you show it leaves the machine.",
    qDefault: "What is in this picture? Read any text in it exactly.",
    connecting: "connecting…", unreachable: "server not reachable", visionOn: "vision on",
    noVision: "no vision encoder", needTower: "Start the server with --tower auto to ask about images.",
    offline: "offline", online: "network on", total: " s total", camErr: "camera: ", screenErr: "screen: ",
    quiet: "quiet — the model is not running", changing: "changing — waiting for it to settle",
    asking: "asking the model ({why})…", needRule: "add a rule first", stopped: "stopped",
    modelErr: "model: ", towerErr: "tower: ", first: "first look", change: "scene changed",
    backstopWhy: "backstop", rest: "came to rest", nothing: "nothing",
    holding: "now: “{rule}” — acts again only after this stops being true (e.g. they leave the view)", whenPh: "when it sees… (e.g. the kettle is boiling)",
    sayPh: "say… (optional)", fxAlert: "beep + notify", fxNone: "log only",
    remove: "remove", answerLang: "" },
  zh: { tabAsk: "提问", tabWatch: "盯着", hImage: "图片",
    drop: "粘贴截图（⌘V）、拖入图片，或用下面的按钮拍一张", pick: "选择文件…",
    camera: "摄像头", screen: "屏幕", ask: "提问", hAnswer: "回答", kTtft: "首个 token",
    kTps: "解码", kPrompt: "提示词 token", kImg: "图片", hSource: "来源",
    screenOrWindow: "屏幕或窗口", videoFile: "视频文件…",
    hRules: "看到…就…", addRule: "+ 规则", presets: "预设…",
    pBuild: "构建完成或失败", pDoor: "有人来到门口",
    fast: "一有变化就反应", start: "开始", stop: "停止", fire: "触发",
    settle: "稳定", backstop: "兜底（秒）", hGate: "门控", kFrames: "帧", kTower: "编码器",
    kAsked: "问模型", kDist: "距离", idle: "空闲",
    footer: "全部在这台 Mac 上运行。这个页面只能和提供它的服务通信" +
            "（Content-Security-Policy connect-src 'self'），你给它看的任何东西都不会离开本机。",
    qDefault: "图里有什么？如果有文字，请原样抄下来。",
    connecting: "连接中…", unreachable: "连不上服务", visionOn: "能看图",
    noVision: "没有视觉编码器", needTower: "启动服务时加上 --tower auto 才能问图片。",
    offline: "已断网", online: "联网中", total: " 秒", camErr: "摄像头：", screenErr: "屏幕：",
    quiet: "画面没变——模型没有运行", changing: "画面在变——等它稳定下来",
    asking: "正在问模型（{why}）…", needRule: "先加一条规则", stopped: "已停止",
    modelErr: "模型：", towerErr: "编码器：", first: "第一眼", change: "画面变了",
    backstopWhy: "兜底检查", rest: "画面停下来了", nothing: "都不成立",
    holding: "当前：「{rule}」——要等它不再成立（比如人离开画面）才会再次触发", whenPh: "看到…（例如：水壶开了）",
    sayPh: "说…（可选）", fxAlert: "提示音 + 通知", fxNone: "只记录",
    remove: "删除", answerLang: " Write the second line in Chinese." },
};
let lang = "en";
try { lang = localStorage.getItem("tempo9.lang") || ""; } catch (_) { lang = ""; }
if (!STR[lang]) lang = (navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en";
const t = (k) => (STR[lang][k] !== undefined ? STR[lang][k] : STR.en[k]);
function applyLang() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  for (const el of document.querySelectorAll("[data-t]")) el.textContent = t(el.dataset.t);
  $("lang").textContent = lang === "zh" ? "EN" : "中文";
  const q = $("q");
  if (!q.value || q.value === STR.en.qDefault || q.value === STR.zh.qDefault) q.value = t("qDefault");
}

applyLang();

// ---------- header ----------
let visionOn = null;
function visionPill() {
  if (visionOn === null) return;
  $("vision").textContent = visionOn ? t("visionOn") : t("noVision");
  $("vision").className = "pill " + (visionOn ? "good" : "bad");
}
function netPill() {
  const off = !navigator.onLine;
  $("net").textContent = off ? t("offline") : t("online");
  $("net").className = "pill" + (off ? " good" : "");
}
addEventListener("online", netPill); addEventListener("offline", netPill); netPill();

async function hello() {
  $("model").textContent = t("connecting");
  try {
    const m = await (await fetch("/v1/models")).json();
    model = m.data[0].id;
    $("model").textContent = model;
    const s = await (await fetch("/api/show", { method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model }) })).json();
    const vision = (s.capabilities || []).includes("vision");
    visionOn = vision; visionPill();
    if (!vision) {
      $("askNote").textContent = t("needTower");
    }
  } catch (e) {
    $("model").textContent = t("unreachable"); $("model").className = "pill bad";
  }
}
hello();

for (const [btn, pane] of [["tabAsk", "ask"], ["tabWatch", "watch"]]) {
  $(btn).onclick = () => {
    for (const [b, p] of [["tabAsk", "ask"], ["tabWatch", "watch"]]) {
      $(b).classList.toggle("on", b === btn); $(p).classList.toggle("hide", p !== pane);
    }
  };
}

// ---------- images ----------
// Long side capped: past ~1 MP the tower scales down anyway, and a larger
// upload only costs time.  Gate frames are 640 px wide, Eyes On's gate size.
function frameToJpeg(src, maxSide, quality) {
  const w = src.videoWidth || src.naturalWidth || src.width;
  const h = src.videoHeight || src.naturalHeight || src.height;
  const k = Math.min(1, maxSide / Math.max(w, h));
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(w * k)); c.height = Math.max(1, Math.round(h * k));
  c.getContext("2d").drawImage(src, 0, 0, c.width, c.height);
  return { url: c.toDataURL("image/jpeg", quality), w: c.width, h: c.height };
}
function loadImage(url) {
  return new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(i); i.onerror = bad; i.src = url; });
}
async function grabStream(constraintsFn) {
  const stream = await constraintsFn();
  const v = document.createElement("video");
  v.muted = true; v.playsInline = true; v.srcObject = stream; await v.play();
  await new Promise((r) => setTimeout(r, 350));   // let exposure settle
  const f = frameToJpeg(v, 1440, 0.9);
  stream.getTracks().forEach((t) => t.stop());
  return f.url;
}

// ---------- chat ----------
async function askModel(imageURL, text, maxTokens, onText) {
  const body = { model, stream: true, stream_options: { include_usage: true },
    temperature: 0, top_k: 1, max_tokens: maxTokens,
    chat_template_kwargs: { enable_thinking: false },
    messages: [{ role: "user", content: [
      { type: "image_url", image_url: { url: imageURL } }, { type: "text", text }] }] };
  const t0 = performance.now();
  const r = await fetch("/v1/chat/completions", { method: "POST",
    headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) {
    let msg = r.status + "";
    try { const j = await r.json(); msg = (j.error && (j.error.message || j.error)) || msg; } catch (_) {}
    throw new Error(msg);
  }
  const reader = r.body.getReader(); const dec = new TextDecoder();
  let buf = "", out = "", first = null, usage = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const ev = buf.slice(0, i); buf = buf.slice(i + 2);
      for (const line of ev.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const d = line.slice(6);
        if (d === "[DONE]") continue;
        const j = JSON.parse(d);
        if (j.usage) usage = j.usage;
        const c = j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content;
        if (c) { if (first === null) first = performance.now(); out += c; if (onText) onText(out); }
      }
    }
  }
  const end = performance.now();
  const ttft = first === null ? null : (first - t0) / 1000;
  const gen = usage && usage.completion_tokens;
  const tps = gen && first !== null && end > first ? (gen - 1) / ((end - first) / 1000) : null;
  return { text: out, usage, ttft, tps, total: (end - t0) / 1000 };
}

// ---------- ask ----------
let askURL = null, askSize = "";
async function setAskImage(url) {
  const img = await loadImage(url);
  const f = frameToJpeg(img, 1440, 0.9);
  askURL = f.url; askSize = f.w + "×" + f.h;
  $("drop").innerHTML = ""; const shown = new Image(); shown.src = f.url; $("drop").appendChild(shown);
  $("go").disabled = false; $("sImg").textContent = askSize;
}
function readFile(file) {
  return new Promise((ok) => { const r = new FileReader(); r.onload = () => ok(r.result); r.readAsDataURL(file); });
}
$("pick").onclick = () => $("file").click();
$("file").onchange = async (e) => { if (e.target.files[0]) setAskImage(await readFile(e.target.files[0])); };
$("drop").ondragover = (e) => { e.preventDefault(); $("drop").classList.add("over"); };
$("drop").ondragleave = () => $("drop").classList.remove("over");
$("drop").ondrop = async (e) => {
  e.preventDefault(); $("drop").classList.remove("over");
  const f = [...e.dataTransfer.files].find((x) => x.type.startsWith("image/"));
  if (f) setAskImage(await readFile(f));
};
addEventListener("paste", async (e) => {
  if ($("ask").classList.contains("hide")) return;
  const item = [...e.clipboardData.items].find((x) => x.type.startsWith("image/"));
  if (item) { e.preventDefault(); setAskImage(await readFile(item.getAsFile())); }
});
$("snapCam").onclick = async () => {
  try { setAskImage(await grabStream(() => navigator.mediaDevices.getUserMedia({ video: { width: 1280 } }))); }
  catch (e) { $("askNote").textContent = t("camErr") + e.message; }
};
$("snapScreen").onclick = async () => {
  try { setAskImage(await grabStream(() => navigator.mediaDevices.getDisplayMedia({ video: true }))); }
  catch (e) { $("askNote").textContent = t("screenErr") + e.message; }
};
$("go").onclick = async () => {
  if (!askURL) return;
  $("go").disabled = true; $("answer").textContent = ""; $("askNote").textContent = "";
  for (const id of ["sTtft", "sTps", "sPrompt"]) $(id).textContent = "…";
  try {
    const r = await askModel(askURL, $("q").value, 512, (t) => { $("answer").textContent = t; });
    $("sTtft").textContent = r.ttft === null ? "–" : r.ttft.toFixed(2) + " s";
    $("sTps").textContent = r.tps ? r.tps.toFixed(1) + " tok/s" : "–";
    $("sPrompt").textContent = r.usage ? r.usage.prompt_tokens : "–";
    $("askNote").textContent = r.total.toFixed(1) + t("total");
  } catch (e) {
    $("answer").innerHTML = ""; const s = document.createElement("span");
    s.className = "err"; s.textContent = e.message; $("answer").appendChild(s);
    for (const id of ["sTtft", "sTps", "sPrompt"]) $(id).textContent = "–";
  }
  $("go").disabled = false;
};

// ---------- watch ----------
const vid = $("vid");
let stream = null, running = false, busy = false, ticker = null;
let audio = null, lastMatch = 0, srcKind = "camera";
const W = { anchor: null, prev: null, pending: false, followUp: false, lastAsk: 0, tiny: null,
            towerAt: 0, dAnchor: 1, frames: 0, tower: 0, asked: 0, series: [],
            regional: false, noise: null };

// Thresholds by source.  A camera's are Eyes On's measured ones (sensor
// noise sits well under 0.025).  A screen or a video file has no sensor:
// an unchanged frame measures ~5e-7, and a "BUILD SUCCEEDED" line appearing
// in a dark terminal only 0.017 -- under the camera threshold, so the
// change waited a full minute for the backstop.  Hence lower gates there,
// and a finer pixel check in front of them.
//
// Those fixed numbers are for the whole-frame vector, which dilutes a local
// change by its area: a visitor filling under a third of a camera frame
// stayed under 0.025 (centre 10% replaced: 0.009; 30%: 0.023).  When the
// server sends region vectors (4 x 3) the gate takes the region that changed
// most, and the thresholds follow the scene's own noise the way Eyes On's
// do: fire at 3.5x an EMA of the quiet frame-to-frame distance, clamped
// (rLo..rHi), settle at 2x (sLo..sHi).  Measured on the Qwen3.5-9B tower,
// max-region distance: sensor-noise proxy 0.013-0.023, +4% exposure 0.007,
// a 2% patch 0.035-0.088, 5% 0.076-0.096, 10% 0.11-0.12.
const GATES = { camera: { fire: 0.025, settle: 0.010, pixel: 0.004,
                          rLo: 0.030, rHi: 0.060, sLo: 0.020, sHi: 0.040, n0: 0.012 },
                screen: { fire: 0.008, settle: 0.004, pixel: 0.0015,
                          rLo: 0.010, rHi: 0.040, sLo: 0.006, sHi: 0.020, n0: 0.002 },
                video:  { fire: 0.008, settle: 0.004, pixel: 0.0015,
                          rLo: 0.010, rHi: 0.040, sLo: 0.006, sHi: 0.020, n0: 0.002 } };
function setKind(kind) {
  srcKind = kind; W.noise = null;
  $("fire").value = GATES[kind].fire; $("settle").value = GATES[kind].settle;
}
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
// The thresholds in force: adaptive with region vectors (shown in the two
// boxes, read-only), the boxes' own values without them.
function gates() {
  if (!W.regional) return { fire: +$("fire").value, settle: +$("settle").value };
  const G = GATES[srcKind], n = W.noise == null ? G.n0 : W.noise;
  const g = { fire: clamp(3.5 * n, G.rLo, G.rHi), settle: clamp(2 * n, G.sLo, G.sHi) };
  $("fire").value = g.fire.toFixed(3); $("settle").value = g.settle.toFixed(3);
  $("fire").readOnly = $("settle").readOnly = true;
  return g;
}

function stopSource() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null; vid.srcObject = null;
  if (vid.src) { URL.revokeObjectURL(vid.src); vid.removeAttribute("src"); vid.load(); }
}
async function useStream(kind, fn) {
  stopSource();
  try {
    stream = await fn(); vid.srcObject = stream; await vid.play();
    setKind(kind); $("start").disabled = false;
  } catch (e) { $("gateState").textContent = e.message; }
}
const camera = () => navigator.mediaDevices.getUserMedia({ video: { width: 1280 } });
$("srcCam").onclick = () => useStream("camera", camera);
$("srcScreen").onclick = () => useStream("screen", () => navigator.mediaDevices.getDisplayMedia({ video: true }));
$("srcFile").onclick = () => $("vfile").click();
$("vfile").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  stopSource(); vid.src = URL.createObjectURL(f); vid.loop = false;
  await vid.play().catch(() => {}); setKind("video"); $("start").disabled = false;
};

// ---------- rules: "when it sees X, say Y and do Z" ----------
// The model is shown the numbered conditions and answers with the first
// one that holds, or 0.  Order is priority: put the specific ones first.
const PRESETS = { en: {
  build: { fast: false, rules: [
    { when: "a build or test run has finished successfully", say: "The build is done.", fx: "alert" },
    { when: "a build or test run has failed with an error", say: "The build failed.", fx: "alert" } ] },
  door: { fast: false, rules: [
    { when: "a person is standing at the door", say: "Someone is at the door.", fx: "alert" } ] },
  first: { fast: false, rules: [{ when: "the progress bar has finished", say: "", fx: "alert" }] },
}, zh: {
  build: { fast: false, rules: [
    { when: "构建或测试已经成功完成", say: "构建完成了。", fx: "alert" },
    { when: "构建或测试失败并报错", say: "构建失败了。", fx: "alert" } ] },
  door: { fast: false, rules: [
    { when: "有人站在门口", say: "门口有人。", fx: "alert" } ] },
  first: { fast: false, rules: [{ when: "进度条走完了", say: "", fx: "alert" }] },
} };
const FX = [["alert", "fxAlert"], ["none", "fxNone"]];

function ruleRow(r) {
  const row = document.createElement("div"); row.className = "rule";
  const when = Object.assign(document.createElement("input"), { className: "when", value: r.when || "",
                 placeholder: t("whenPh") });
  const say = Object.assign(document.createElement("input"), { className: "say", value: r.say || "",
                 placeholder: t("sayPh") });
  const fx = document.createElement("select"); fx.className = "fx";
  for (const [v, key] of FX) fx.add(new Option(t(key), v, false, v === (r.fx || "alert")));
  const del = Object.assign(document.createElement("button"), { textContent: "×", title: t("remove") });
  del.onclick = () => { row.remove(); saveRules(); };
  for (const el of [when, say, fx]) el.onchange = saveRules;
  row.append(when, say, fx, del);
  return row;
}
function readRules() {
  return [...document.querySelectorAll("#rules .rule")].map((row) => ({
    when: row.querySelector(".when").value.trim(), say: row.querySelector(".say").value.trim(),
    fx: row.querySelector(".fx").value })).filter((r) => r.when);
}
function setRules(rules, fast) {
  $("rules").innerHTML = ""; for (const r of rules) $("rules").appendChild(ruleRow(r));
  if (fast !== undefined) $("fast").checked = fast;
  saveRules();
}
// A per-viewer convenience: the rules you wrote survive a reload.  Nothing
// depends on it, and a browser that refuses storage just starts fresh.
function saveRules() {
  try { localStorage.setItem("tempo9.rules", JSON.stringify({ rules: readRules(), fast: $("fast").checked })); } catch (_) {}
}
(function loadRules() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem("tempo9.rules") || "null"); } catch (_) {}
  if (saved && saved.rules && saved.rules.length) setRules(saved.rules, !!saved.fast);
  else setRules(PRESETS[lang].first.rules, false);
})();
$("addRule").onclick = () => { $("rules").appendChild(ruleRow({})); };
$("lang").onclick = () => {
  lang = lang === "zh" ? "en" : "zh";
  try { localStorage.setItem("tempo9.lang", lang); } catch (_) {}
  applyLang(); netPill(); visionPill();
  if (!model) $("model").textContent = t($("model").classList.contains("bad") ? "unreachable" : "connecting");
  if (!running) $("gateState").textContent = t("idle");
  setRules(readRules(), $("fast").checked);   // placeholders and action names
};
$("fast").onchange = saveRules;
$("preset").onchange = () => {
  const p = PRESETS[lang][$("preset").value]; $("preset").value = "";
  if (p) setRules(p.rules, p.fast);
};

function tinyGray(src) {
  const c = tinyGray.c || (tinyGray.c = Object.assign(document.createElement("canvas"), { width: 32, height: 18 }));
  const x = c.getContext("2d", { willReadFrequently: true });
  x.drawImage(src, 0, 0, 32, 18);
  const d = x.getImageData(0, 0, 32, 18).data, g = new Float32Array(576);
  for (let i = 0; i < 576; i++) g[i] = (d[4 * i] * .299 + d[4 * i + 1] * .587 + d[4 * i + 2] * .114) / 255;
  return g;
}
function meanAbs(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length; }
function f32(b64) {
  const s = atob(b64), u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return new Float32Array(u.buffer);
}
function dist(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return 1 - s; }
// Gate distance between two tower frames: the region that changed most when
// both carry regions, else the whole-frame vectors.
function gdist(a, b) {
  if (a.r && b.r && a.r.length === b.r.length) {
    let m = 0; for (let i = 0; i < a.r.length; i++) m = Math.max(m, dist(a.r[i], b.r[i]));
    return m;
  }
  return dist(a.g, b.g);
}

async function towerFrame() {
  const f = frameToJpeg(vid, 640, 0.7);
  const r = await fetch("/v1/vision/frame", { method: "POST",
    headers: { "content-type": "application/json" }, body: JSON.stringify({ image: f.url }) });
  const j = await r.json();
  if (!r.ok) throw new Error((j.error && j.error.message) || r.status);
  const g = f32(j.embedding_f32), dim = j.dim || g.length;
  let regs = null;
  if (j.regions_f32) {
    const all = f32(j.regions_f32); regs = [];
    for (let i = 0; i + dim <= all.length; i += dim) regs.push(all.subarray(i, i + dim));
  }
  return { g, r: regs };
}

function beep() {
  if (!audio) return;
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = 880; g.gain.value = 0.15; o.connect(g); g.connect(audio.destination);
  o.start(); o.stop(audio.currentTime + 0.25);
}
// The system's own voices, so speaking stays on the machine too; a line
// with Chinese in it asks for a Chinese voice.
function speak(line) {
  if (!line || !("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(line);
  u.lang = /[一-鿿]/.test(line) ? "zh-CN" : "en-US";
  speechSynthesis.speak(u);
}
// What a matched rule does.  `seen` is the model's own sentence, used when
// the rule has nothing to say.
function act(rule, seen) {
  document.title = "● " + rule.when;
  if (rule.fx === "none") return;
  beep(); setTimeout(beep, 350);
  if ("Notification" in window && Notification.permission === "granted") {
    new Notification("Tempo9: " + rule.when, { body: rule.say || seen });
  }
  speak(rule.say);
}

function logEntry(thumbURL, yes, text, why, secs) {
  const li = document.createElement("li");
  const img = new Image(); img.src = thumbURL; li.appendChild(img);
  const div = document.createElement("div");
  const head = document.createElement("div");
  head.className = yes ? "yes" : ""; head.textContent = text.split("\n").filter(Boolean).join(" — ");
  const meta = document.createElement("div"); meta.className = "why";
  meta.textContent = new Date().toLocaleTimeString() + " · " + why + " · " + secs.toFixed(1) + " s";
  div.appendChild(head); div.appendChild(meta); li.appendChild(div);
  $("log").prepend(li);
}

// Ask which rule holds now.  A rule acts when it becomes the answer, not
// for as long as it stays the answer: one greeting per visitor, re-armed
// when a later look sees the rule no longer holding (the "rest" look in
// tick makes sure an emptied doorway is looked at).  The first look acts
// too, fast or not: a first look that only recorded waited for a change
// that, with someone already in view, never came.
async function check(why) {
  const rules = readRules();
  if (!rules.length) { $("gateState").textContent = t("needRule"); return; }
  busy = true; W.asked++; $("cAsked").textContent = W.asked;
  $("gateState").textContent = t("asking").replace("{why}", t(why));
  const shot = frameToJpeg(vid, 1440, 0.85);
  const prompt = "You are keeping an eye on a live " + srcKind + " for the user.\n" +
    "Conditions:\n" + rules.map((r, i) => (i + 1) + ". " + r.when).join("\n") + "\n" +
    "Look at this frame. On the first line write only the number of the first condition " +
    "that is true in this frame, or 0 if none is. On the second line, one short sentence " +
    "saying what you see that decides it." + t("answerLang");
  try {
    const r = await askModel(shot.url, prompt, 80);
    const lines = r.text.trim().split("\n").map((s) => s.trim()).filter(Boolean);
    let k = parseInt(((lines[0] || "").match(/\d+/) || ["0"])[0], 10);
    if (!(k >= 1 && k <= rules.length)) k = 0;
    const seen = lines.slice(1).join(" ") || lines[0] || "";
    const rule = k ? rules[k - 1] : null;
    W.series.push({ mark: rule ? "yes" : "no" });
    logEntry(frameToJpeg(vid, 192, 0.7).url, !!rule,
             (rule ? k + ". " + rule.when : t("nothing")) + "\n" + seen, t(why), r.total);
    if (rule && k !== lastMatch) act(rule, seen);
    if (!rule) document.title = "Tempo9";
    lastMatch = k;
  } catch (e) {
    $("gateState").textContent = t("modelErr") + e.message;
  }
  W.lastAsk = performance.now() / 1000; W.pending = false;
  busy = false;
}

async function tick() {
  if (!running || busy || vid.readyState < 2) return;
  busy = true;
  const now = performance.now() / 1000;
  const backstop = +$("backstop").value;
  const { fire, settle } = gates();
  W.frames++; $("cFrames").textContent = W.frames;
  let dPrev;
  try {
    const thumb = tinyGray(vid);
    // Identical at 32x18 against the last frame the tower SAW (not the last
    // sampled), so a slow change still adds up to a tower call; and at
    // least one tower call every 10 s regardless -- every 2 s while a fast
    // watch has someone in view, because the last sliver of a visitor
    // leaving the frame is under the pixel check, and until the tower sees
    // the empty doorway the next visitor is "the same answer".
    const towerGap = $("fast").checked && lastMatch ? 2 : 10;
    if (W.tiny && meanAbs(thumb, W.tiny) < GATES[srcKind].pixel && now - W.towerAt < towerGap) {
      dPrev = 0;
    } else {
      const v = await towerFrame();
      W.tower++; $("cTower").textContent = W.tower; W.tiny = thumb; W.towerAt = now;
      if (v.r) W.regional = true;
      dPrev = W.prev ? gdist(W.prev, v) : 1;
      W.dAnchor = W.anchor ? gdist(W.anchor, v) : 1;
      // The scene's own noise: frame-to-frame distance while nothing is
      // pending and nothing crossed the threshold.
      if (W.prev && !W.pending && dPrev < fire)
        W.noise = W.noise == null ? dPrev : 0.9 * W.noise + 0.1 * dPrev;
      W.prev = v;
    }
  } catch (e) {
    $("gateState").textContent = t("towerErr") + e.message; busy = false; return;
  }
  $("cDist").textContent = W.dAnchor < 1 ? W.dAnchor.toFixed(4) : "–";
  W.series.push({ d: W.dAnchor }); if (W.series.length > 400) W.series.splice(0, W.series.length - 400);
  draw(fire);
  busy = false;

  const fast = $("fast").checked;
  // The frame the model is about to see becomes the anchor, so the
  // distance to it is zero until the tower measures another frame -- a
  // pixel-skipped tick must not keep the old, above-threshold distance (it
  // did, and a settled screen was asked about every second).
  const rebase = () => { W.anchor = W.prev; W.dAnchor = 0; };
  if (!W.anchor) {                          // first look: the baseline
    rebase(); await check("first"); return;
  }
  if (W.dAnchor > fire) W.pending = true;
  const settled = W.pending && (fast || dPrev < settle);
  // A fast look is taken mid-motion, and someone half out of the door is
  // still "an adult".  The rest of their exit can fall under the fire
  // threshold, so the empty doorway was never asked about, the rule never
  // stopped being the answer, and the next visitor was not greeted.  Hence
  // one more look once a fast-watched scene comes to rest somewhere the
  // model has not seen.
  const rested = W.followUp && dPrev < settle && W.dAnchor > settle;
  const due = now - W.lastAsk >= backstop;
  if (settled) {
    rebase(); await check("change"); W.followUp = fast;
  } else if (rested) {
    W.followUp = false; rebase(); await check("rest");
  } else if (due) {
    // Region distances never fall to the whole frame's 0.003 (the noise
    // floor alone is ~0.013), so "changed at all" is the settle level there.
    if (W.dAnchor >= (W.regional ? settle : 0.003)) { rebase(); await check("backstopWhy"); }
    else W.lastAsk = now;                   // checked: nothing has changed
  } else {
    $("gateState").textContent = W.pending ? t("changing") : holding() || t("quiet");
  }
}
// Why nothing happens while a rule holds: it acted and waits for the
// answer to change.
function holding() {
  const r = lastMatch && readRules()[lastMatch - 1];
  return r ? t("holding").replace("{rule}", r.when) : "";
}

function draw(fire) {
  const c = $("chart"), x = c.getContext("2d"), w = c.width, h = c.height;
  const css = getComputedStyle(document.documentElement);
  x.clearRect(0, 0, w, h);
  const top = Math.max(fire * 3, 0.05);
  const y = (d) => h - 8 - Math.min(d, top) / top * (h - 16);
  x.strokeStyle = css.getPropertyValue("--yes"); x.setLineDash([8, 6]); x.lineWidth = 2;
  x.beginPath(); x.moveTo(0, y(fire)); x.lineTo(w, y(fire)); x.stroke(); x.setLineDash([]);
  const pts = W.series.slice(-200), step = w / 200;
  x.strokeStyle = css.getPropertyValue("--acc"); x.lineWidth = 3; x.beginPath();
  let n = 0, lastD = 0;
  pts.forEach((p, i) => {
    if (p.d === undefined) return;
    lastD = p.d < 1 ? p.d : 0;
    const px = i * step, py = y(lastD);
    if (n++ === 0) x.moveTo(px, py); else x.lineTo(px, py);
  });
  x.stroke();
  pts.forEach((p, i) => {
    if (!p.mark) return;
    x.fillStyle = p.mark === "yes" ? css.getPropertyValue("--yes") : css.getPropertyValue("--mut");
    x.beginPath(); x.arc(i * step, 14, 7, 0, 7); x.fill();
  });
}

$("start").onclick = async () => {
  if (!model) return;
  audio = audio || new AudioContext(); audio.resume();
  if ("Notification" in window && Notification.permission === "default") Notification.requestPermission();
  Object.assign(W, { anchor: null, prev: null, pending: false, followUp: false, lastAsk: performance.now() / 1000,
                     tiny: null, towerAt: 0, dAnchor: 1, frames: 0, tower: 0, asked: 0, series: [],
                     noise: null });
  for (const id of ["cFrames", "cTower", "cAsked"]) $(id).textContent = "0";
  $("log").innerHTML = ""; lastMatch = 0;
  running = true; $("start").disabled = true; $("stop").disabled = false;
  // A worker's timer keeps its 1 s pace in a background tab.
  if (!ticker) {
    ticker = new Worker(URL.createObjectURL(new Blob(
      ["setInterval(() => postMessage(0), 1000)"], { type: "text/javascript" })));
    ticker.onmessage = () => tick();
  }
};
$("stop").onclick = () => {
  running = false; $("start").disabled = false; $("stop").disabled = true;
  $("gateState").textContent = t("stopped"); document.title = "Tempo9";
};
</script>
</body>
</html>
"""#
}
