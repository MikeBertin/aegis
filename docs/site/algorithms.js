/* AEGIS — "algorithms from scratch" demo. Runs the real modules via Pyodide+NumPy. */
"use strict";

const COL = { bg:"#010409", fg:"#c9d1d9", muted:"#8b949e", grid:"#21262d",
  green:"#3fb950", red:"#f85149", amber:"#d29922", blue:"#58a6ff" };

const PY = `
import sys, json
import numpy as np
sys.path.insert(0, "/home/pyodide")
from aegis.kalman import CVKalman1D
from aegis.assignment import linear_sum_assignment
from aegis.nms import nms
from aegis.stereo_match import block_match_disparity, make_synthetic_pair

def kalman_run(R, Q, seed):
    rng = np.random.RandomState(int(seed)); n, dt = 130, 0.1
    t = np.arange(n) * dt
    true = 20*np.sin(0.12*t) + 10*np.sin(0.05*t)   # smooth, CV-trackable
    meas = true + rng.randn(n)*np.sqrt(R)
    kf = CVKalman1D(process_var=Q, meas_var=R)
    est, std = [], []
    for i in range(n):
        p, _ = kf.update(float(meas[i]), dt)
        est.append(p); std.append(kf.pos_var**0.5)
    rms = lambda a: float(np.sqrt(np.mean((np.asarray(a)-true)**2)))
    return json.dumps({"t": t.tolist(), "true": true.tolist(), "meas": meas.tolist(),
                       "est": est, "std": std,
                       "rms_meas": round(rms(meas),2), "rms_est": round(rms(est),2)})

def assign(cost):
    cost = [[float(x) for x in row] for row in cost]
    pairs = linear_sum_assignment(cost)
    total = sum(cost[i][j] for i, j in pairs)
    n, m = len(cost), len(cost[0])
    flat = sorted((cost[i][j], i, j) for i in range(n) for j in range(m))
    ut, ud, g = set(), set(), []
    for c, i, j in flat:
        if i not in ut and j not in ud:
            ut.add(i); ud.add(j); g.append((i, j))
    return json.dumps({"pairs": pairs, "total": round(total,2),
                       "greedy": round(sum(cost[i][j] for i,j in g),2)})

def nms_run(boxes, scores, thr):
    keep = nms([tuple(b) for b in boxes], [float(s) for s in scores], float(thr))
    return json.dumps({"keep": keep})

def stereo_run(block, maxd):
    L, R, _ = make_synthetic_pair(size=90, bg_disp=4, fg_disp=16, seed=3)
    disp = block_match_disparity(L, R, max_disparity=int(maxd), block_size=int(block))
    return json.dumps({"left": L.tolist(), "right": R.tolist(), "disp": disp.tolist(),
                       "fg": round(float(np.median(disp[34:56, 34:56])),1),
                       "bg": round(float(np.median(disp[8:28, 60:84])),1)})
`;

let py = {};

async function boot() {
  const pyodide = await loadPyodide();
  document.getElementById("loadmsg").textContent = "Loading NumPy…";
  await pyodide.loadPackage("numpy");
  try { pyodide.FS.mkdir("/home/pyodide/aegis"); } catch (e) {}
  for (const [path, src] of Object.entries(window.AEGIS_MODULES)) {
    pyodide.FS.writeFile("/home/pyodide/" + path, src);
  }
  pyodide.runPython(PY);
  for (const f of ["kalman_run", "assign", "nms_run", "stereo_run"]) py[f] = pyodide.globals.get(f);
  document.getElementById("loading").style.display = "none";
  initKalman(); initHungarian(); initNMS(); initStereo();
}

function fit(id) {
  const c = document.getElementById(id), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = w * dpr; c.height = h * dpr;
  const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/* ---------- ① Kalman ---------- */
function initKalman() {
  ["kR", "kQ"].forEach(id => {
    const el = document.getElementById(id), out = document.getElementById(id + "o");
    el.addEventListener("input", () => { out.value = el.value; drawKalman(); });
    out.value = el.value;
  });
  drawKalman();
}
function drawKalman() {
  const R = +document.getElementById("kR").value, Q = +document.getElementById("kQ").value;
  const d = JSON.parse(py.kalman_run(R, Q, 1));
  const { ctx, w, h } = fit("kalCanvas");
  const pad = 30, tmax = d.t[d.t.length - 1];
  let lo = Infinity, hi = -Infinity;
  d.meas.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
  lo -= 4; hi += 4;
  const X = t => pad + t / tmax * (w - 2 * pad);
  const Y = v => h - pad - (v - lo) / (hi - lo) * (h - 2 * pad);
  ctx.clearRect(0, 0, w, h);
  // uncertainty band ±2σ
  ctx.fillStyle = "rgba(63,185,80,0.18)"; ctx.beginPath();
  d.t.forEach((t, i) => { const x = X(t), y = Y(d.est[i] + 2*d.std[i]); i ? ctx.lineTo(x,y):ctx.moveTo(x,y); });
  for (let i = d.t.length - 1; i >= 0; i--) ctx.lineTo(X(d.t[i]), Y(d.est[i] - 2*d.std[i]));
  ctx.closePath(); ctx.fill();
  // true path
  ctx.strokeStyle = COL.muted; ctx.lineWidth = 1.5; ctx.beginPath();
  d.t.forEach((t, i) => { const fn = i?"lineTo":"moveTo"; ctx[fn](X(t), Y(d.true[i])); }); ctx.stroke();
  // measurements
  ctx.fillStyle = COL.red;
  d.t.forEach((t, i) => { ctx.beginPath(); ctx.arc(X(t), Y(d.meas[i]), 1.8, 0, 7); ctx.fill(); });
  // estimate
  ctx.strokeStyle = COL.green; ctx.lineWidth = 2.2; ctx.beginPath();
  d.t.forEach((t, i) => { const fn = i?"lineTo":"moveTo"; ctx[fn](X(t), Y(d.est[i])); }); ctx.stroke();
  ctx.fillStyle = COL.muted; ctx.font = "11px monospace";
  ctx.fillText("— true   • measurements   — Kalman estimate (±2σ)", pad, 16);
  document.getElementById("kReadout").textContent =
    `RMS error vs truth:  raw measurements ${d.rms_meas}   →   Kalman ${d.rms_est}  (smoothed)`;
}

/* ---------- ② Hungarian ---------- */
let costMatrix = [[8,4,7,3],[5,2,9,6],[4,7,1,8],[6,5,3,2]];
function initHungarian() {
  const tbl = document.getElementById("costTable");
  tbl.innerHTML = "";
  costMatrix.forEach((row, i) => {
    const tr = document.createElement("tr");
    row.forEach((val, j) => {
      const td = document.createElement("td"); td.id = `c${i}_${j}`;
      const inp = document.createElement("input");
      inp.value = val; inp.addEventListener("input", () => {
        costMatrix[i][j] = parseFloat(inp.value) || 0; drawHungarian();
      });
      td.appendChild(inp); tr.appendChild(td);
    });
    tbl.appendChild(tr);
  });
  drawHungarian();
}
function drawHungarian() {
  const d = JSON.parse(py.assign(costMatrix));
  document.querySelectorAll("#costTable td").forEach(td => td.classList.remove("assigned"));
  d.pairs.forEach(([i, j]) => document.getElementById(`c${i}_${j}`).classList.add("assigned"));
  const better = d.greedy > d.total;
  document.getElementById("hReadout").innerHTML =
    `optimal total = <b style="color:${COL.green}">${d.total}</b><br>` +
    `greedy total = <b style="color:${better?COL.amber:COL.fg}">${d.greedy}</b>` +
    (better ? `  <span style="color:${COL.amber}">(greedy is worse by ${(d.greedy-d.total).toFixed(2)})</span>` : `  (greedy is optimal here)`);
}

/* ---------- ③ NMS ---------- */
const nmsBoxes = (() => {
  // three clusters of overlapping boxes, each with a clear winner.
  const out = [];
  const cluster = (cx, cy, top) => {
    out.push({ b: [cx-45, cy-35, cx+45, cy+35], s: top });
    for (let k = 0; k < 4; k++) {
      const dx = (Math.random()-0.5)*40, dy = (Math.random()-0.5)*30;
      out.push({ b: [cx-45+dx, cy-35+dy, cx+45+dx, cy+35+dy], s: top - 0.15 - Math.random()*0.3 });
    }
  };
  cluster(150, 150, 0.95); cluster(420, 180, 0.9); cluster(680, 140, 0.88);
  return out;
})();
function initNMS() {
  const el = document.getElementById("nT"), out = document.getElementById("nTo");
  el.addEventListener("input", () => { out.value = el.value; drawNMS(); });
  out.value = el.value; drawNMS();
}
function drawNMS() {
  const thr = +document.getElementById("nT").value;
  const keep = new Set(JSON.parse(py.nms_run(nmsBoxes.map(o => o.b), nmsBoxes.map(o => o.s), thr)).keep);
  const { ctx, w, h } = fit("nmsCanvas");
  ctx.clearRect(0, 0, w, h);
  // suppressed first (so kept boxes draw on top)
  nmsBoxes.forEach((o, i) => {
    if (keep.has(i)) return;
    const [x1, y1, x2, y2] = o.b;
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(139,148,158,0.55)"; ctx.setLineDash([4, 3]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  });
  ctx.setLineDash([]);
  nmsBoxes.forEach((o, i) => {
    if (!keep.has(i)) return;
    const [x1, y1, x2, y2] = o.b;
    ctx.fillStyle = "rgba(63,185,80,0.14)"; ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    ctx.lineWidth = 2.5; ctx.strokeStyle = COL.green;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = COL.green; ctx.font = "12px monospace";
    ctx.fillText(o.s.toFixed(2), x1 + 4, y1 - 5);
  });
  ctx.setLineDash([]);
  document.getElementById("nReadout").textContent =
    `${nmsBoxes.length} raw boxes  →  ${keep.size} kept  (IoU threshold ${thr})`;
}

/* ---------- ④ Block-matching stereo ---------- */
let stereoTimer = null;
function initStereo() {
  ["sB", "sD"].forEach(id => {
    const el = document.getElementById(id), out = document.getElementById(id + "o");
    el.addEventListener("input", () => {
      out.value = el.value;
      clearTimeout(stereoTimer); stereoTimer = setTimeout(drawStereo, 120);
    });
    out.value = el.value;
  });
  drawStereo();
}
function grayCanvas(arr) {
  const h = arr.length, w = arr[0].length;
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const ctx = c.getContext("2d"), img = ctx.createImageData(w, h);
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
    const v = arr[i][j], k = (i*w+j)*4;
    img.data[k] = img.data[k+1] = img.data[k+2] = v; img.data[k+3] = 255;
  }
  ctx.putImageData(img, 0, 0); return c;
}
function dispCanvas(arr, maxd) {
  const h = arr.length, w = arr[0].length;
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const ctx = c.getContext("2d"), img = ctx.createImageData(w, h);
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
    const t = Math.min(1, arr[i][j] / maxd), k = (i*w+j)*4;
    img.data[k] = 30 + t*225; img.data[k+1] = 20 + t*60; img.data[k+2] = 80 + t*60; img.data[k+3] = 255;
  }
  ctx.putImageData(img, 0, 0); return c;
}
function blit(src, id) {
  const dst = document.getElementById(id), ctx = dst.getContext("2d");
  ctx.imageSmoothingEnabled = false; ctx.clearRect(0,0,dst.width,dst.height);
  ctx.drawImage(src, 0, 0, dst.clientWidth, dst.clientHeight);
}
function drawStereo() {
  const block = +document.getElementById("sB").value, maxd = +document.getElementById("sD").value;
  document.getElementById("sReadout").textContent = "computing…";
  setTimeout(() => {
    const d = JSON.parse(py.stereo_run(block, maxd));
    blit(grayCanvas(d.left), "stLeft");
    blit(grayCanvas(d.right), "stRight");
    blit(dispCanvas(d.disp, maxd), "stDisp");
    document.getElementById("sReadout").textContent =
      `recovered disparity:  foreground ${d.fg}px (closer)   vs   background ${d.bg}px (farther)`;
  }, 10);
}

boot().catch(e => { document.getElementById("loadmsg").textContent = "Failed: " + e; });
