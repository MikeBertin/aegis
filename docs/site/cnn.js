/* AEGIS CNN demo — runs the real from-scratch conv.py in the browser (Pyodide+NumPy). */
"use strict";

const COL = { green:"#3fb950", red:"#f85149", amber:"#d29922", muted:"#8b949e" };
const COLORS = {  // RGB used to draw; 'red' balloon is the designated target
  red:[220,20,20], blue:[20,20,220], green:[20,180,20], yellow:[220,220,20],
};
const KERNELS = {
  "edge":     [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],
  "sharpen":  [[0,-1,0],[-1,5,-1],[0,-1,0]],
  "blur":     [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],
  "sobel-x":  [[1,0,-1],[2,0,-2],[1,0,-1]],
  "sobel-y":  [[1,2,1],[0,0,0],[-1,-2,-1]],
  "identity": [[0,0,0],[0,1,0],[0,0,0]],
};

const PY = `
import json, numpy as np
import cnn_conv as C
W = {}
def set_weights(s):
    global W
    W = {k: np.array(v, dtype=float) for k, v in json.loads(s).items()}
def classify(flat):
    x = np.array(flat, dtype=float).reshape(3, 32, 32)
    p1 = C.max_pool2d(C.relu(C.conv2d(x, W['conv1_w'], W['conv1_b'], padding=1)))
    p2 = C.max_pool2d(C.relu(C.conv2d(p1, W['conv2_w'], W['conv2_b'], padding=1)))
    f1 = C.relu(C.linear(p2.reshape(-1), W['fc1_w'], W['fc1_b']))
    probs = C.softmax(C.linear(f1, W['fc2_w'], W['fc2_b']))
    return json.dumps({"prob": float(probs[1]), "feats": [p1[i].tolist() for i in range(8)]})
def apply_kernel(flat, kernel):
    x = np.array(flat, dtype=float).reshape(3, 32, 32).mean(0, keepdims=True)
    k = np.array(kernel, dtype=float).reshape(1, 1, 3, 3)
    return json.dumps({"inp": x[0].tolist(), "out": C.conv2d(x, k, padding=1)[0].tolist()})
`;

let pyClassify, pyKernel;
const state = { color: "red", shape: "balloon", kernel: "edge", flat: null };

async function boot() {
  const pyodide = await loadPyodide();
  document.getElementById("loadmsg").textContent = "Loading NumPy…";
  await pyodide.loadPackage("numpy");
  pyodide.FS.writeFile("/home/pyodide/cnn_conv.py", window.CNN_CONV_SRC);
  pyodide.runPython(PY);
  pyodide.globals.get("set_weights")(JSON.stringify(window.CNN_WEIGHTS));
  pyClassify = pyodide.globals.get("classify");
  pyKernel = pyodide.globals.get("apply_kernel");
  document.getElementById("loading").style.display = "none";
  initUI();
  redraw();
}

/* ---------- the shared 32x32 patch ---------- */
const patch = document.createElement("canvas");
patch.width = patch.height = 32;

function drawPatch() {
  const ctx = patch.getContext("2d");
  const img = ctx.createImageData(32, 32);
  for (let i = 0; i < 32 * 32; i++) {           // noisy background
    const n = 30 + Math.random() * 70;
    img.data[i*4] = n; img.data[i*4+1] = n; img.data[i*4+2] = n; img.data[i*4+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const [r, g, b] = COLORS[state.color];
  ctx.fillStyle = `rgb(${r},${g},${b})`;
  if (state.shape === "balloon") {
    ctx.beginPath(); ctx.ellipse(16, 15, 7, 9, 0, 0, 7); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.ellipse(13, 12, 1.6, 1.6, 0, 0, 7); ctx.fill();
    ctx.strokeStyle = "rgb(40,40,40)"; ctx.beginPath(); ctx.moveTo(16, 24); ctx.lineTo(16, 30); ctx.stroke();
  } else {
    ctx.fillRect(9, 6, 14, 18);
  }
}

function readFlat() {
  const d = patch.getContext("2d").getImageData(0, 0, 32, 32).data;
  const B = [], G = [], R = [];
  for (let i = 0; i < 32 * 32; i++) {
    R.push(d[i*4] / 255); G.push(d[i*4+1] / 255); B.push(d[i*4+2] / 255);
  }
  return B.concat(G, R);  // channel-first BGR, to match training (cv2)
}

function blit(srcCanvas, dstId) {
  const dst = document.getElementById(dstId);
  const ctx = dst.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, dst.width, dst.height);
  ctx.drawImage(srcCanvas, 0, 0, dst.width, dst.height);
}

/* ---------- render helpers ---------- */
function heatmap(map, size) {
  // map: 2D array; returns a canvas normalised to [0,1] greyscale.
  const h = map.length, w = map[0].length;
  let lo = Infinity, hi = -Infinity;
  for (const row of map) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const span = hi - lo || 1;
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const ctx = c.getContext("2d"); const img = ctx.createImageData(w, h);
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {
    const t = (map[i][j] - lo) / span, idx = (i * w + j) * 4;
    img.data[idx] = 40 + t * 100; img.data[idx+1] = 60 + t * 175; img.data[idx+2] = 50 + t * 40;
    img.data[idx+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

/* ---------- updates ---------- */
function redraw() {
  drawPatch();
  state.flat = readFlat();
  blit(patch, "patchView");
  runConv();
  runClassify();
}

function runConv() {
  const res = JSON.parse(pyKernel(state.flat, KERNELS[state.kernel]));
  blit(heatmap(res.inp, 0), "convIn");
  blit(heatmap(res.out, 0), "convOut");
  document.getElementById("kmat").textContent =
    KERNELS[state.kernel].map(r => r.map(v => (Math.round(v*100)/100).toString().padStart(5)).join(" ")).join("\n");
}

function runClassify() {
  const res = JSON.parse(pyClassify(state.flat));
  const pct = Math.round(res.prob * 100);
  const v = document.getElementById("verdict");
  if (res.prob >= 0.6) { v.textContent = "✓ TARGET — designated red balloon"; v.style.color = COL.green; }
  else { v.textContent = "✗ NOT A TARGET"; v.style.color = COL.amber; }
  const bar = document.getElementById("probbar");
  bar.style.width = pct + "%";
  bar.style.background = res.prob >= 0.6 ? COL.green : COL.amber;
  document.getElementById("probtxt").textContent = `target probability: ${pct}%`;
  const grid = document.getElementById("feats");
  grid.innerHTML = "";
  res.feats.forEach(m => {
    const cv = document.createElement("canvas"); cv.width = cv.height = 48;
    const ctx = cv.getContext("2d"); ctx.imageSmoothingEnabled = false;
    ctx.drawImage(heatmap(m, 0), 0, 0, 48, 48);
    grid.appendChild(cv);
  });
}

function initUI() {
  document.querySelectorAll(".swatch").forEach(s => {
    if (s.dataset.color === state.color) s.classList.add("sel");
    s.addEventListener("click", () => {
      document.querySelectorAll(".swatch").forEach(x => x.classList.remove("sel"));
      s.classList.add("sel"); state.color = s.dataset.color; redraw();
    });
  });
  document.querySelectorAll("[data-shape]").forEach(b => {
    b.addEventListener("click", () => {
      document.querySelectorAll("[data-shape]").forEach(x => x.classList.remove("sel"));
      b.classList.add("sel"); state.shape = b.dataset.shape; redraw();
    });
  });
  document.getElementById("redraw").addEventListener("click", redraw);
  const kc = document.getElementById("kernels");
  Object.keys(KERNELS).forEach(name => {
    const b = document.createElement("button");
    b.textContent = name; if (name === state.kernel) b.classList.add("sel");
    b.addEventListener("click", () => {
      kc.querySelectorAll("button").forEach(x => x.classList.remove("sel"));
      b.classList.add("sel"); state.kernel = name; runConv();
    });
    kc.appendChild(b);
  });
}

boot().catch(e => { document.getElementById("loadmsg").textContent = "Failed: " + e; });
