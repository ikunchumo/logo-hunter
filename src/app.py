"""
app.py — 图片目标检测器（苹果风双页：扫描 / 训练）

启动：python src/app.py  [--port 8770]
打开：http://127.0.0.1:8770

接口
  POST /api/scan          {"path": "...", "recursive": true}
  GET  /api/cached?dir=   上次扫描结果（秒开）
  GET  /api/thumb?path=   缩略图（命中自动画框）
  POST /api/train/start   {"per_class","epochs","lr","batch","aug","size"}
  POST /api/train/stop
  GET  /api/train/status
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
from detector import BadgeDetector  # noqa: E402
from scan import iter_images        # noqa: E402
import trainer                      # noqa: E402

DET: BadgeDetector | None = None
CACHE: dict[str, dict] = {}
DIR_CACHE: dict[str, dict] = {}
SAMPLES = ROOT / "data" / "samples"
CUSTOM_TPL = ROOT / "assets" / "custom_template.png"
DEFAULT_TPL = ROOT / "assets" / "template.png"
TPL_INFO: dict = {"source": "default", "path": str(DEFAULT_TPL),
                  "width": 0, "height": 0, "keypoints": 0}


def reload_detector(path: Path) -> dict:
    """换模板 = 重建检测器 + 清空所有缓存。

    关键点太少的模板（<80）会在 RANSAC 里靠运气凑出内点，产生大量假匹配，
    所以直接拒绝使用。
    """
    global DET
    probe = BadgeDetector(path)
    kp = len(probe.kp_t)
    if kp < 80:
        raise ValueError(
            f"这张图只提取到 {kp} 个特征点，太少，匹配会不可靠。"
            "请换一张边缘清晰、细节丰富、长边 300px 以上的图")
    DET = probe
    CACHE.clear()
    DIR_CACHE.clear()
    TPL_INFO.update({"path": str(path), "width": DET.tw, "height": DET.th,
                     "keypoints": kp,
                     "source": "custom" if path == CUSTOM_TPL else "default"})
    return dict(TPL_INFO)


def template_info() -> dict:
    if DET is None:
        return dict(TPL_INFO)
    TPL_INFO.update({"width": DET.tw, "height": DET.th,
                     "keypoints": len(DET.kp_t)})
    return dict(TPL_INFO)

PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>图像检测器</title>
<style>
  :root{
    --bg:#f5f5f7; --card:#ffffff; --ink:#1d1d1f; --sub:#6e6e73;
    --line:#d2d2d7; --blue:#0071e3; --green:#34c759; --red:#ff3b30; --orange:#ff9500;
    --shadow:0 2px 18px rgba(0,0,0,.06); --r:20px;
  }
  *{margin:0;padding:0;box-sizing:border-box;-webkit-font-smoothing:antialiased}
  body{background:var(--bg);color:var(--ink);
       font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text",
       "PingFang SC","Microsoft YaHei",system-ui,sans-serif;
       letter-spacing:-.01em;padding-bottom:80px}

  nav{position:sticky;top:0;z-index:20;backdrop-filter:saturate(180%) blur(20px);
      -webkit-backdrop-filter:saturate(180%) blur(20px);
      background:rgba(245,245,247,.78);border-bottom:.5px solid rgba(0,0,0,.08)}
  .nav-in{max-width:1120px;margin:0 auto;height:56px;display:flex;align-items:center;
      gap:14px;padding:0 24px}
  .mark{width:30px;height:30px;border-radius:9px;background:#fff;display:flex;
      align-items:center;justify-content:center;box-shadow:0 1px 5px rgba(0,0,0,.14)}
  .brand{font-size:16px;font-weight:600;letter-spacing:-.02em}
  .brand span{color:var(--sub);font-weight:400;font-size:12.5px;margin-left:8px}
  .seg{margin-left:auto;background:rgba(118,118,128,.12);border-radius:11px;
       padding:2px;display:flex;gap:2px}
  .seg button{border:0;background:transparent;color:var(--ink);font-size:13px;
       font-weight:500;padding:6px 16px;border-radius:9px;cursor:pointer;
       transition:all .25s cubic-bezier(.4,0,.2,1);font-family:inherit}
  .seg button.on{background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.12)}

  .wrap{max-width:1120px;margin:0 auto;padding:44px 24px 0}
  h1{font-size:40px;font-weight:600;letter-spacing:-.03em;line-height:1.1}
  h1 .grad{background:linear-gradient(135deg,#0071e3,#5e5ce6);
       -webkit-background-clip:text;background-clip:text;color:transparent}
  .lead{color:var(--sub);font-size:15.5px;margin-top:12px;line-height:1.55;max-width:660px}

  .card{background:var(--card);border-radius:var(--r);box-shadow:var(--shadow);
        padding:22px}
  .row{display:flex;gap:12px;align-items:center}
  input[type=text]{flex:1;border:.5px solid var(--line);background:#fafafa;
      border-radius:12px;padding:13px 16px;font-size:14.5px;color:var(--ink);
      font-family:inherit;outline:none;transition:.2s}
  input[type=text]:focus{border-color:var(--blue);background:#fff;
      box-shadow:0 0 0 4px rgba(0,113,227,.12)}
  .btn{border:0;border-radius:12px;padding:13px 24px;font-size:14.5px;font-weight:600;
      font-family:inherit;cursor:pointer;transition:.2s;white-space:nowrap}
  .btn.primary{background:var(--blue);color:#fff}
  .btn.primary:hover{background:#0077ed;transform:translateY(-.5px)}
  .btn.ghost{background:rgba(118,118,128,.12);color:var(--ink)}
  .btn:disabled{opacity:.42;cursor:not-allowed;transform:none}

  .stats{display:flex;gap:34px;margin:26px 2px 18px;flex-wrap:wrap}
  .stat b{display:block;font-size:26px;font-weight:600;letter-spacing:-.02em}
  .stat span{font-size:12.5px;color:var(--sub)}
  .stat b.hit{color:var(--green)}

  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(196px,1fr));
      gap:16px;margin-top:6px}
  .tile{background:var(--card);border-radius:16px;overflow:hidden;
      box-shadow:var(--shadow);position:relative;transition:.28s cubic-bezier(.4,0,.2,1)}
  .tile:hover{transform:translateY(-3px);box-shadow:0 10px 28px rgba(0,0,0,.1)}
  .tile img{width:100%;height:140px;object-fit:cover;display:block;background:#eee}
  .tile .cap{padding:9px 11px;font-size:11.5px;color:var(--sub);display:flex;
      justify-content:space-between;gap:6px}
  .tile .cap i{font-style:normal;color:var(--ink);overflow:hidden;
      text-overflow:ellipsis;white-space:nowrap}
  .badge{position:absolute;top:9px;left:9px;font-size:10.5px;font-weight:700;
      padding:3px 9px;border-radius:20px;background:var(--green);color:#fff;
      box-shadow:0 2px 8px rgba(52,199,89,.4)}
  .tile.hit{outline:2px solid var(--green);outline-offset:-2px}
  .tile.miss{outline:2px solid var(--orange);outline-offset:-2px}
  .tile.fp{outline:2px solid var(--red);outline-offset:-2px}
  .badge.miss{background:var(--orange)}
  .badge.fp{background:var(--red)}
  .badge.ok{background:#98989d}

  /* 模板卡片 */
  .tplcard{margin-top:26px;display:flex;align-items:center;gap:16px;
           padding:16px 20px}
  .tplcard img{width:52px;height:52px;object-fit:contain;border-radius:10px;
           background:#fafafa;border:.5px solid var(--line);padding:3px}
  .tplmeta{flex:1;min-width:0}
  .tplname{font-size:13.5px;font-weight:600}
  .tplsub{font-size:12px;color:var(--sub);margin-top:3px}
  .tplsub.warn{color:var(--orange)}
  .tile .prob{height:4px;background:#ededf0;margin:0 11px;border-radius:4px;overflow:hidden}
  .tile .prob > div{height:100%;border-radius:4px}
  .tile .ptxt{padding:5px 11px 9px;font-size:10.5px;color:var(--sub);
              font-variant-numeric:tabular-nums}

  .empty{text-align:center;color:var(--sub);font-size:14px;padding:70px 0;line-height:2}

  /* 训练页 */
  .train{display:grid;grid-template-columns:340px 1fr;gap:20px;margin-top:26px}
  @media(max-width:900px){.train{grid-template-columns:1fr}}
  .field{margin-bottom:20px}
  .field label{display:flex;justify-content:space-between;font-size:13px;
      color:var(--sub);margin-bottom:8px}
  .field label b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
  input[type=range]{width:100%;-webkit-appearance:none;height:4px;border-radius:4px;
      background:rgba(118,118,128,.2);outline:none}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;
      border-radius:50%;background:#fff;box-shadow:0 1px 5px rgba(0,0,0,.22);
      cursor:pointer;transition:.15s}
  input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.08)}
  .chips{display:flex;gap:8px;margin-top:6px}
  .chip{flex:1;text-align:center;font-size:12.5px;padding:8px 0;border-radius:9px;
      background:rgba(118,118,128,.1);cursor:pointer;transition:.2s;
      border:.5px solid transparent}
  .chip.on{background:#fff;border-color:var(--line);font-weight:600;
      box-shadow:0 1px 4px rgba(0,0,0,.06)}

  canvas{width:100%;height:240px;display:block}
  .chart-head{display:flex;justify-content:space-between;align-items:center;
      margin-bottom:6px}
  .chart-head h3{font-size:15px;font-weight:600}
  .legend{display:flex;gap:14px;font-size:11.5px;color:var(--sub)}
  .legend i{display:inline-block;width:9px;height:9px;border-radius:50%;
      margin-right:5px;vertical-align:middle}

  .prog{height:6px;border-radius:6px;background:rgba(118,118,128,.15);overflow:hidden;
      margin:16px 0 6px}
  .prog > div{height:100%;width:0;background:linear-gradient(90deg,#0071e3,#5e5ce6);
      border-radius:6px;transition:width .4s cubic-bezier(.4,0,.2,1)}
  .log{margin-top:14px;background:#fafafa;border-radius:12px;padding:12px 14px;
      font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;
      color:var(--sub);height:132px;overflow-y:auto;line-height:1.75;
      border:.5px solid rgba(0,0,0,.05)}
  .result{display:grid;grid-template-columns:repeat(auto-fit,minmax(104px,1fr));
      gap:12px;margin-top:16px}
  .result .k{background:#fafafa;border-radius:14px;padding:13px 14px}
  .result .k b{display:block;font-size:21px;font-weight:600;letter-spacing:-.02em}
  .result .k span{font-size:11.5px;color:var(--sub)}
  .hint{font-size:12px;color:var(--sub);margin-top:10px;line-height:1.6}
  .hidden{display:none!important}
</style></head>
<body>

<nav><div class="nav-in">
  <div class="mark">
    <svg width="19" height="19" viewBox="0 0 24 24" fill="none"
         stroke="#0071e3" stroke-width="1.9" stroke-linecap="round"
         stroke-linejoin="round">
      <path d="M3 8V5.5A2.5 2.5 0 0 1 5.5 3H8"/>
      <path d="M16 3h2.5A2.5 2.5 0 0 1 21 5.5V8"/>
      <path d="M21 16v2.5a2.5 2.5 0 0 1-2.5 2.5H16"/>
      <path d="M8 21H5.5A2.5 2.5 0 0 1 3 18.5V16"/>
      <path d="M8.5 12.2l2.4 2.4 4.6-4.8"/>
    </svg>
  </div>
  <div class="brand">图像检测器<span>在图片库里检索指定目标</span></div>
  <div class="seg">
    <button id="tab-scan" class="on">扫描</button>
    <button id="tab-train">训练</button>
  </div>
</div></nav>

<!-- ================= 扫描页 ================= -->
<section class="wrap" id="view-scan">
  <h1>在一堆图片里<br><span class="grad">找出含目标的那几张。</span></h1>

  <!-- 当前检测模板：可上传替换 -->
  <div class="card tplcard">
    <img id="tpl-img" src="/api/template/image" alt="template">
    <div class="tplmeta">
      <div class="tplname">当前检测模板</div>
      <div class="tplsub" id="tpl-meta">加载中…</div>
    </div>
    <div class="row" style="flex:0 0 auto;gap:8px">
      <button class="btn ghost" id="tpl-upload">更换模板</button>
      <button class="btn ghost" id="tpl-reset">恢复默认</button>
    </div>
    <input type="file" id="tpl-file" accept="image/*" hidden>
  </div>

  <div class="card">
    <div class="row">
      <input type="text" id="path" placeholder="D:\photos">
      <button class="btn primary" id="go">开始扫描</button>
    </div>
  </div>

  <div class="stats" id="stats"></div>
  <div class="grid" id="grid"></div>
  <div class="empty" id="empty">还没有结果<br>填一个文件夹路径开始扫描</div>
</section>

<!-- ================= 训练页 ================= -->
<section class="wrap hidden" id="view-train">
  <div class="train">
    <div class="card">
      <div class="field">
        <label>每类样本数 <b id="v-samples">600</b></label>
        <input type="range" id="s-samples" min="100" max="1200" step="50" value="600">
      </div>
      <div class="field">
        <label>训练轮数 <b id="v-epochs">14</b></label>
        <input type="range" id="s-epochs" min="3" max="40" step="1" value="14">
      </div>
      <div class="field">
        <label>学习率 <b id="v-lr">0.0030</b></label>
        <input type="range" id="s-lr" min="0" max="100" step="1" value="55">
      </div>
      <div class="field">
        <label>批大小 <b id="v-batch">64</b></label>
        <input type="range" id="s-batch" min="16" max="128" step="16" value="64">
      </div>
      <div class="field">
        <label>数据增强强度 <b id="v-aug">0.60</b></label>
        <input type="range" id="s-aug" min="0" max="100" step="5" value="60">
      </div>
      <div class="field" style="margin-bottom:6px">
        <label>输入尺寸</label>
        <div class="chips" id="sizes">
          <div class="chip" data-size="64">64</div>
          <div class="chip on" data-size="96">96</div>
          <div class="chip" data-size="128">128</div>
        </div>
      </div>
      <div class="row" style="margin-top:18px">
        <button class="btn primary" id="start" style="flex:1">开始训练</button>
        <button class="btn ghost" id="stop" disabled>停止</button>
      </div>
      <div class="hint" id="device">设备探测中…</div>
    </div>

    <div class="card">
      <div class="chart-head">
        <h3>训练曲线</h3>
        <div class="legend">
          <span><i style="background:#0071e3"></i>loss</span>
          <span><i style="background:#34c759"></i>验证准确率</span>
        </div>
      </div>
      <canvas id="chart"></canvas>
      <div class="prog"><div id="bar"></div></div>
      <div class="hint" id="progress-text">等待开始</div>
      <div class="result hidden" id="result"></div>
      <div class="log" id="log">准备就绪。参数改完点「开始训练」。</div>
    </div>

    <!-- CNN 搜索演示：训练完之后，用这个模型扫一遍图库，和 SIFT 对照 -->
    <div class="card" id="cnn-card" style="grid-column:1 / -1">
      <div class="chart-head">
        <h3>CNN 搜索演示</h3>
        <div class="legend">
          <span><i style="background:#34c759"></i>判为含目标</span>
          <span><i style="background:#c7c7cc"></i>判为不含</span>
        </div>
      </div>
      <p class="hint" style="margin-top:2px">
        用刚训好的模型给示例图库逐张打分，P ≥ 50% 判为含目标。
        每张图右下角同时标出 SIFT 的结果作为对照。
      </p>
      <div class="row" style="margin-top:14px">
        <button class="btn primary" id="cnn-run" style="flex:0 0 auto;padding-left:26px;padding-right:26px">
          用当前模型扫描图库
        </button>
        <span class="hint" id="cnn-status" style="margin:0">点一下开始（约 1~2 秒，首次会先跑一遍 SIFT 对照）</span>
      </div>
      <div class="stats hidden" id="cnn-stats" style="margin:18px 2px 12px"></div>
      <div class="grid" id="cnn-grid"></div>
    </div>
  </div>
</section>

<script>
const $ = id => document.getElementById(id);

/* ---------- 分页 ---------- */
function show(v){
  $('view-scan').classList.toggle('hidden', v!=='scan');
  $('view-train').classList.toggle('hidden', v!=='train');
  $('tab-scan').classList.toggle('on', v==='scan');
  $('tab-train').classList.toggle('on', v==='train');
  if(v==='train') drawChart();
}
$('tab-scan').onclick = () => show('scan');
$('tab-train').onclick = () => show('train');

/* ---------- 扫描 ---------- */
const grid=$('grid'), stats=$('stats'), empty=$('empty');
let scanning=false;

function tileHTML(x){
  return `<div class="tile ${x.found?'hit':''}">
    ${x.found?'<span class="badge">命中</span>':''}
    <img loading="lazy" src="/api/thumb?path=${encodeURIComponent(x.file)}" alt="">
    <div class="cap"><i title="${x.name}">${x.name}</i>
      <span>${x.found?('inl '+x.inliers):''}</span></div>
  </div>`;
}

function render(d){
  const hits = d.results.filter(x=>x.found);
  stats.innerHTML = `
    <div class="stat"><b>${d.total}</b><span>张图片</span></div>
    <div class="stat"><b class="hit">${hits.length}</b><span>命中</span></div>
    <div class="stat"><b>${d.seconds}s</b><span>总耗时</span></div>
    <div class="stat"><b>${d.avg_ms}</b><span>ms / 张</span></div>`;
  const ordered = [...hits, ...d.results.filter(x=>!x.found)];
  grid.innerHTML = ordered.map(tileHTML).join('');
  empty.classList.add('hidden');
}

async function scan(path){
  if(scanning) return;
  scanning=true; $('go').disabled=true; $('go').textContent='扫描中…';
  empty.classList.add('hidden'); grid.innerHTML='';
  stats.innerHTML='<div class="stat"><b>…</b><span>正在逐张检测</span></div>';
  try{
    const r = await fetch('/api/scan',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({path})});
    const d = await r.json();
    if(d.error){ stats.innerHTML=`<div class="stat"><b style="color:#ff3b30">!</b>
      <span>${d.error}</span></div>`; }
    else render(d);
  }catch(e){
    stats.innerHTML=`<div class="stat"><b style="color:#ff3b30">!</b><span>${e.message}</span></div>`;
  }
  $('go').disabled=false; $('go').textContent='开始扫描'; scanning=false;
}
$('go').onclick = () => { const p=$('path').value.trim(); if(p) scan(p); };
$('path').addEventListener('keydown', e=>{ if(e.key==='Enter'){
  const p=$('path').value.trim(); if(p) scan(p); }});

/* ---------- 模板：查看 / 更换 / 恢复默认 ---------- */
async function refreshTemplate(){
  try{
    const t = await (await fetch('/api/template')).json();
    $('tpl-meta').textContent =
      `${t.width}×${t.height} · ${t.keypoints} 个特征点 · ` +
      (t.source === 'custom' ? '自定义' : '默认模板');
    $('tpl-meta').classList.remove('warn');
    $('tpl-img').src = '/api/template/image?t=' + Date.now();
  }catch(e){ $('tpl-meta').textContent = '模板信息读取失败'; }
}
refreshTemplate();

$('tpl-upload').onclick = () => $('tpl-file').click();
$('tpl-file').onchange = e => {
  const f = e.target.files[0];
  if(!f) return;
  const rd = new FileReader();
  rd.onload = async ev => {
    $('tpl-meta').textContent = '正在提取特征点…';
    $('tpl-meta').classList.remove('warn');
    try{
      const r = await (await fetch('/api/template',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({image: ev.target.result})})).json();
      if(r.error){ $('tpl-meta').textContent = r.error; $('tpl-meta').classList.add('warn'); return; }
      await refreshTemplate();
      if(r.warning){ $('tpl-meta').textContent = r.warning; $('tpl-meta').classList.add('warn'); }
      grid.innerHTML=''; empty.classList.remove('hidden');
      empty.innerHTML = '模板已更换<br>重新扫描生效';
      stats.innerHTML = '';
    }catch(err){ $('tpl-meta').textContent = '上传失败：' + err.message; }
  };
  rd.readAsDataURL(f);
  e.target.value = '';
};

$('tpl-reset').onclick = async () => {
  const r = await (await fetch('/api/template/reset',{method:'POST'})).json();
  await refreshTemplate();
  grid.innerHTML=''; stats.innerHTML='';
  empty.classList.remove('hidden');
  empty.innerHTML = '已恢复默认模板<br>重新扫描生效';
};

/* ---------- 训练 ---------- */
const lrFromSlider = v => Math.pow(10, -4 + v/100*2);   // 1e-4 ~ 1e-2
const sSamples=$('s-samples'), sEpochs=$('s-epochs'), sLr=$('s-lr'),
      sBatch=$('s-batch'), sAug=$('s-aug');
let size = 96;

function syncLabels(){
  $('v-samples').textContent = sSamples.value;
  $('v-epochs').textContent = sEpochs.value;
  $('v-lr').textContent = lrFromSlider(+sLr.value).toFixed(4);
  $('v-batch').textContent = sBatch.value;
  $('v-aug').textContent = (sAug.value/100).toFixed(2);
}
[sSamples,sEpochs,sLr,sBatch,sAug].forEach(s=>s.addEventListener('input',syncLabels));
syncLabels();

document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{
  document.querySelectorAll('.chip').forEach(x=>x.classList.remove('on'));
  c.classList.add('on'); size = +c.dataset.size;
});

/* canvas 曲线 */
const cv = $('chart'), ctx = cv.getContext('2d');
let state = {loss:[], val_acc:[]};

function fitCanvas(){
  const dpr = window.devicePixelRatio || 1;
  const r = cv.getBoundingClientRect();
  cv.width = r.width * dpr; cv.height = 240 * dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {w:r.width, h:240};
}

function drawChart(){
  const {w,h} = fitCanvas();
  ctx.clearRect(0,0,w,h);
  const pad = {l:38,r:14,t:14,b:24};
  const iw = w-pad.l-pad.r, ih = h-pad.t-pad.b;
  // 网格
  ctx.strokeStyle='rgba(0,0,0,.07)'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y = pad.t + ih*i/4;
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke();
  }
  // 轴标签
  ctx.fillStyle='#86868b'; ctx.font='10.5px -apple-system,sans-serif';
  ctx.textAlign='right';
  for(let i=0;i<=4;i++){
    const y = pad.t + ih*i/4;
    ctx.fillText((1-i/4).toFixed(2), pad.l-6, y+3.5);
  }
  const L = state.loss, A = state.val_acc, n = Math.max(L.length, A.length);
  if(n < 1) return;

  const px = i => pad.l + (n>1 ? iw*i/(n-1) : iw/2);
  const py = v => pad.t + ih*(1-Math.min(1,Math.max(0,v)));

  const line = (arr, color, fill) => {
    if(arr.length<1) return;
    ctx.beginPath();
    arr.forEach((v,i)=> i? ctx.lineTo(px(i),py(v)) : ctx.moveTo(px(i),py(v)));
    ctx.strokeStyle=color; ctx.lineWidth=2.2; ctx.lineJoin='round';
    ctx.stroke();
    if(fill && arr.length>1){
      ctx.lineTo(px(arr.length-1), pad.t+ih); ctx.lineTo(px(0), pad.t+ih);
      ctx.closePath();
      const g = ctx.createLinearGradient(0,pad.t,0,pad.t+ih);
      g.addColorStop(0, color+'22'); g.addColorStop(1, color+'00');
      ctx.fillStyle=g; ctx.fill();
    }
    // 末点
    const i=arr.length-1;
    ctx.beginPath(); ctx.arc(px(i),py(arr[i]),3.2,0,7); ctx.fillStyle=color; ctx.fill();
  };
  line(L, '#0071e3', true);
  line(A, '#34c759', false);
}

window.addEventListener('resize', ()=>{ if(!$('view-train').classList.contains('hidden')) drawChart(); });

let polling = null;
function setBusy(b){
  $('start').disabled = b; $('stop').disabled = !b;
  $('start').textContent = b ? '训练中…' : '开始训练';
}

function startPoll(){
  if(polling) clearInterval(polling);
  polling = setInterval(async ()=>{
    const s = await (await fetch('/api/train/status')).json();
    state.loss = s.loss; state.val_acc = s.val_acc;
    drawChart();
    if(s.total_epochs>0)
      $('bar').style.width = Math.min(100, s.epoch/s.total_epochs*100)+'%';
    $('progress-text').textContent = s.done
      ? `完成 · ${s.seconds}s`
      : (s.running ? `epoch ${s.epoch} / ${s.total_epochs} · ${s.seconds}s`
                   : (s.error ? '出错：'+s.error : '等待开始'));
    if(s.log && s.log.length){
      const el=$('log'); const atBottom = el.scrollHeight-el.scrollTop-el.clientHeight<40;
      el.innerHTML = s.log.map(l=>`<div>${l}</div>`).join('');
      if(atBottom) el.scrollTop = el.scrollHeight;
    }
    if(s.result){
      const r = s.result;
      $('result').classList.remove('hidden');
      $('result').innerHTML = `
        <div class="k"><b>${(r.test_acc*100).toFixed(1)}%</b><span>测试准确率</span></div>
        <div class="k"><b>${(r.precision*100).toFixed(1)}%</b><span>精确率</span></div>
        <div class="k"><b>${(r.recall*100).toFixed(1)}%</b><span>召回率</span></div>
        <div class="k"><b>${r.seconds}s</b><span>训练耗时</span></div>
        <div class="k"><b>${(r.params/1000).toFixed(0)}K</b><span>参数量</span></div>`;
    }
    if(!s.running){ setBusy(false); }
    if(!s.running && polling){ clearInterval(polling); polling=null; }
  }, 1000);
}

$('start').onclick = async ()=>{
  const body = {
    per_class: +sSamples.value, epochs: +sEpochs.value,
    lr: +lrFromSlider(+sLr.value).toFixed(5), batch: +sBatch.value,
    aug: +(sAug.value/100).toFixed(2), size: size
  };
  $('result').classList.add('hidden');
  setBusy(true); $('bar').style.width='0%';
  state = {loss:[], val_acc:[]}; drawChart();
  const r = await (await fetch('/api/train/start',{method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)})).json();
  if(!r.ok){ $('log').innerHTML = `<div style="color:#ff3b30">${r.error}</div>`;
             setBusy(false); return; }
  $('device').textContent = '设备：' + r.device;
  startPoll();
};
$('stop').onclick = async ()=>{ await fetch('/api/train/stop',{method:'POST'}); };

/* ---------- CNN 搜索演示 ---------- */
$('cnn-run').onclick = async ()=>{
  const btn = $('cnn-run'), st = $('cnn-status');
  btn.disabled = true; btn.textContent = '扫描中…';
  st.textContent = '正在用 CNN 推理，并准备 SIFT 对照…';
  $('cnn-grid').innerHTML = ''; $('cnn-stats').classList.add('hidden');
  try{
    const r = await (await fetch('/api/cnn/scan',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:'{}'})).json();
    if(r.error){ st.textContent = r.error; btn.disabled=false; btn.textContent='用当前模型扫描图库'; return; }

    const hits = r.results.filter(x=>x.cnn_found).length;
    $('cnn-stats').classList.remove('hidden');
    $('cnn-stats').innerHTML = `
      <div class="stat"><b>${r.total}</b><span>张图片</span></div>
      <div class="stat"><b class="hit">${hits}</b><span>CNN 判为含目标</span></div>
      <div class="stat"><b>${r.sift_hits}</b><span>SIFT 命中（对照）</span></div>
      <div class="stat"><b>${r.seconds}s</b><span>CNN 推理耗时</span></div>`;

    // 只按置信度排序，不做对错标注
    const sorted = [...r.results].sort((a,b)=> b.prob - a.prob);
    const LIMIT = 24;
    const showN = Math.min(LIMIT, sorted.length);
    const card = x => {
      const siftTxt = x.sift_found === null ? '' :
        `<span style="color:${x.sift_found?'#34c759':'#98989d'}">SIFT ${x.sift_found?('✓ '+x.sift_inliers):'✗'}</span>`;
      return `<div class="tile ${x.cnn_found?'hit':''}">
        ${x.cnn_found?'<span class="badge">命中</span>':''}
        <img loading="lazy" src="/api/thumb?path=${encodeURIComponent(x.file)}" alt="">
        <div class="cap"><i title="${x.name}">${x.name}</i>${siftTxt}</div>
        <div class="prob"><div style="width:${Math.max(2,x.prob*100)}%;
             background:${x.prob>=0.5?'#34c759':'#c7c7cc'}"></div></div>
        <div class="ptxt">P(有目标) ${(x.prob*100).toFixed(0)}%</div>
      </div>`;
    };
    $('cnn-grid').innerHTML = sorted.slice(0, showN).map(card).join('')
      + (sorted.length > showN
         ? `<div style="grid-column:1/-1;text-align:center;margin-top:6px">
              <button class="btn ghost" id="cnn-more" style="padding:9px 20px;font-size:13px">
                显示全部 ${sorted.length} 张（还有 ${sorted.length - showN} 张）
              </button></div>`
         : '');
    const more = $('cnn-more');
    if (more) more.onclick = () => {
      $('cnn-grid').innerHTML = sorted.map(card).join('');
    };
    st.textContent = `完成：${r.total} 张图里，CNN 判为含目标 ${hits} 张，SIFT 命中 ${r.sift_hits} 张`;
  }catch(e){
    st.textContent = '出错：' + e.message;
  }
  btn.disabled = false; btn.textContent = '用当前模型扫描图库';
};

/* 打开页面时同步一次训练状态（比如上次还在跑） */
(async ()=>{
  const s = await (await fetch('/api/train/status')).json();
  $('device').textContent = '设备：' + (s.device || '探测中…');
  if(s.running){ state.loss=s.loss; state.val_acc=s.val_acc; setBusy(true); startPoll(); }
  else if(s.result){
    state.loss=s.loss; state.val_acc=s.val_acc; drawChart();
    $('result').classList.remove('hidden');
    $('bar').style.width='100%';
    $('progress-text').textContent = `完成 · ${s.seconds}s`;
    const r=s.result;
    $('result').innerHTML = `
      <div class="k"><b>${(r.test_acc*100).toFixed(1)}%</b><span>测试准确率</span></div>
      <div class="k"><b>${(r.precision*100).toFixed(1)}%</b><span>精确率</span></div>
      <div class="k"><b>${(r.recall*100).toFixed(1)}%</b><span>召回率</span></div>
      <div class="k"><b>${r.seconds}s</b><span>训练耗时</span></div>
      <div class="k"><b>${(r.params/1000).toFixed(0)}K</b><span>参数量</span></div>`;
  }
})();

/* URL 参数：?view=train 直接进训练页；?dir=路径 直接扫这个文件夹 */
(async ()=>{
  const qs = new URLSearchParams(location.search);
  if(qs.get('view') === 'train') show('train');
  if(qs.get('cnn') === '1'){
    show('train');
    setTimeout(()=>{ const b=$('cnn-run'); if(b) b.click(); }, 250);
    return;
  }
  const dir = qs.get('dir');
  if(!dir) return;
  $('path').value = dir;
  scan(dir);
})();
</script></body></html>"""


def _sample_scan(limit: int = 60) -> dict:
    """扫示例图库：data/samples/{with_badge,without_badge}。"""
    key = "__samples__"
    if key in DIR_CACHE:
        return DIR_CACHE[key]
    t0 = time.time()
    results = []
    for d in (SAMPLES / "with_badge", SAMPLES / "without_badge"):
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jpg"))[:limit]:
            hit = DET.detect_file(p)
            rec = {"file": str(p), "name": p.name, "found": bool(hit.found),
                   "score": round(hit.score, 2), "inliers": int(hit.inliers),
                   "matches": int(hit.good_matches), "ms": round(hit.elapsed_ms, 1),
                   "box": [int(v) for v in hit.box] if hit.box else None}
            CACHE[str(p)] = rec
            results.append(rec)
    secs = round(time.time() - t0, 1)
    payload = {"total": len(results), "results": results, "seconds": secs,
               "avg_ms": round(secs * 1000 / max(len(results), 1))}
    DIR_CACHE[key] = payload
    return payload


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # ---------------- GET ----------------
    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif u.path in ("/template", "/favicon.ico"):
            svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
                'width="64" height="64" fill="none" stroke="#0071e3" '
                'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
                '<rect x="2" y="2" width="20" height="20" rx="5"/>'
                '<path d="M8.5 12.2l2.4 2.4 4.6-4.8"/></svg>')
            self._send(200, svg.encode("utf-8"), "image/svg+xml")
        elif u.path == "/api/cached":
            q = parse_qs(u.query)
            raw = unquote((q.get("dir") or [""])[0])
            if raw == "__samples__":
                self._json(_sample_scan())
                return
            try:
                key = str(Path(raw).resolve())
            except OSError:
                key = ""
            self._json(DIR_CACHE.get(key) or {"empty": True})
        elif u.path == "/api/train/status":
            self._json(trainer.status())
        elif u.path == "/api/template":
            self._json(template_info())
        elif u.path == "/api/template/image":
            p = Path(template_info()["path"])
            data = p.read_bytes() if p.exists() else b""
            self._send(200, data, "image/png")
        elif u.path == "/api/thumb":
            self._thumb(parse_qs(u.query))
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def _thumb(self, q):
        raw = unquote((q.get("path") or [""])[0])
        p = Path(raw)
        img = None
        if p.exists():
            data = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            self._send(404, b"", "image/jpeg")
            return
        rec = CACHE.get(str(p))
        if rec and rec.get("found") and rec.get("box"):
            x, y, w, h = rec["box"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (52, 199, 89), 3)
        s = 420 / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 84])
        self._send(200, buf.tobytes(), "image/jpeg")

    # ---------------- POST ----------------
    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(n) if n else b"{}"
        try:
            if path == "/api/scan":
                req = json.loads(body or b"{}")
                p = req.get("path", "")
                if p == "__samples__":
                    self._json(_sample_scan())
                    return
                folder = Path(p).expanduser()
                if not folder.exists() or not folder.is_dir():
                    self._json({"error": f"路径不存在：{folder}"}, 400)
                    return
                t0 = time.time()
                files = list(iter_images(folder, True))[:1500]
                from parallel_scan import scan_paths
                # 关键：把当前模板路径传下去，否则子进程会用默认模板
                results = scan_paths(files, workers=8,
                                     template_path=template_info()["path"])
                for rec in results:
                    CACHE[rec["file"]] = rec
                secs = round(time.time() - t0, 1)
                payload = {"total": len(results), "results": results,
                           "seconds": secs,
                           "avg_ms": round(secs * 1000 / max(len(results), 1))}
                DIR_CACHE[str(folder.resolve())] = payload
                self._json(payload)
            elif path == "/api/train/start":
                params = json.loads(body or b"{}")
                self._json(trainer.start(params))
            elif path == "/api/train/stop":
                self._json(trainer.stop())
            elif path == "/api/cnn/scan":
                cnn = trainer.scan_with_model(limit=60)
                try:
                    sift = _sample_scan()
                    smap = {r["file"]: r for r in sift["results"]}
                except Exception:  # noqa: BLE001
                    smap = {}
                for r in cnn["results"]:
                    s = smap.get(r["file"])
                    r["sift_found"] = bool(s["found"]) if s else None
                    r["sift_inliers"] = int(s["inliers"]) if s else 0
                cnn["sift_hits"] = sum(1 for r in cnn["results"] if r.get("sift_found"))
                self._json(cnn)
            elif path == "/api/template":
                req = json.loads(body or b"{}")
                raw = req.get("image", "")
                if not raw:
                    self._json({"error": "没有收到图片"}, 400)
                    return
                b64 = raw.split(",", 1)[1] if "," in raw else raw
                data = base64.b64decode(b64)
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if img is None:
                    self._json({"error": "图片解码失败，换 PNG / JPG 试试"}, 400)
                    return
                # 带 alpha 的图合成到白底（SIFT 要的是灰度纹理）
                if img.ndim == 3 and img.shape[2] == 4:
                    a = img[:, :, 3:4].astype(np.float32) / 255.0
                    bgr = img[:, :, :3].astype(np.float32)
                    img = (bgr * a + 255.0 * (1 - a)).astype(np.uint8)
                elif img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # 长边限制在 800，太大反而慢且没必要
                h, w = img.shape[:2]
                if max(h, w) > 800:
                    s = 800 / max(h, w)
                    img = cv2.resize(img, None, fx=s, fy=s,
                                     interpolation=cv2.INTER_AREA)
                CUSTOM_TPL.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(CUSTOM_TPL), img)
                try:
                    info = reload_detector(CUSTOM_TPL)
                except ValueError as e:
                    CUSTOM_TPL.unlink(missing_ok=True)
                    self._json({"error": str(e)}, 400)
                    return
                warn = None
                if info["keypoints"] < 400:
                    warn = (f"模板只有 {info['keypoints']} 个特征点，偏少，"
                            "目标细节越多（文字、花纹）效果越好")
                self._json({"ok": True, "warning": warn, **info})
            elif path == "/api/template/reset":
                if CUSTOM_TPL.exists():
                    CUSTOM_TPL.unlink()
                info = reload_detector(DEFAULT_TPL)
                self._json({"ok": True, **info})
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main():
    global DET
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    init_tpl = CUSTOM_TPL if CUSTOM_TPL.exists() else DEFAULT_TPL
    info = reload_detector(init_tpl)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"图像检测器已启动： http://{args.host}:{args.port}")
    print(f"模板 {info['width']}×{info['height']}，SIFT 点 {info['keypoints']}"
          f"（{info['source']}）")
    print("训练池：" + ("已就绪" if (ROOT / 'data' / 'train_pool.npz').exists()
                        else "缺失，请运行 src/build_train_pool.py"))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
