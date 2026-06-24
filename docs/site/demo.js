/* AEGIS interactive demo — drives the real Python modules (via Pyodide). */
"use strict";

const COL = {
  bg:"#010409", fg:"#c9d1d9", muted:"#8b949e", grid:"#21262d",
  green:"#3fb950", red:"#f85149", amber:"#d29922", blue:"#58a6ff",
};

// The Python "API" we expose to JS. Imports the real aegis modules and returns
// JSON so we never wrestle with proxy lifetimes.
const PY_DRIVER = `
import sys, json
sys.path.insert(0, "/home/pyodide")
from aegis.controller import default_pan_tilt
from aegis import simulator as sim
from aegis.safety import SafetyGate
from aegis.tracker import Detection

_SCEN = {
    "step": (lambda: sim.step(20.0, -10.0), 2.5),
    "sine": (lambda: sim.sine(26.0, 13.0, 0.5), 4.0),
    "ramp": (lambda: sim.ramp(15.0, 0.0, -20.0, 0.0), 4.0),
}

def run_sim(kp, ki, kd, slew, scenario):
    motion, dur = _SCEN[scenario]
    ctrl = default_pan_tilt(kp=kp, ki=ki, kd=kd, max_slew_deg_s=slew)
    res = sim.run(ctrl, motion(), duration=dur, fps=30)
    metrics = sim.step_metrics(res) if scenario == "step" else sim.tracking_metrics(res)
    return json.dumps({
        "t": res.t, "taz": res.target_az, "tel": res.target_el,
        "pan": res.pan, "tilt": res.tilt, "metrics": metrics, "scenario": scenario,
    })

_GATE = SafetyGate()

def eval_safety(label, tbox, pbox, armed, locked):
    target = Detection(0, label, 0.95, tuple(tbox))
    dets = [target]
    if pbox is not None:
        dets.append(Detection(0, "person", 0.9, tuple(pbox)))
    d = _GATE.evaluate(target, dets, locked=locked, armed=armed)
    return json.dumps({"permit": d.permit, "reason": d.reason})
`;

let runSim, evalSafety;

async function boot() {
  const msg = document.getElementById("loadmsg");
  const pyodide = await loadPyodide();
  msg.textContent = "Loading AEGIS modules…";

  // Materialise the bundled real source into Pyodide's filesystem.
  try { pyodide.FS.mkdir("/home/pyodide/aegis"); } catch (e) {}
  for (const [path, src] of Object.entries(window.AEGIS_MODULES)) {
    pyodide.FS.writeFile("/home/pyodide/" + path, src);
  }
  pyodide.runPython(PY_DRIVER);
  runSim = pyodide.globals.get("run_sim");
  evalSafety = pyodide.globals.get("eval_safety");

  document.getElementById("loading").style.display = "none";
  initTuner();
  initSafety();
}

/* ---------- canvas helpers (retina-crisp) ---------- */
function fit(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/* ============================================================
   PANEL 1 — PID tuner: animated turret viz + response plot
   ============================================================ */
let tuner = { data:null, frame:0, raf:null };

function initTuner() {
  const ids = ["kp","ki","kd","slew"];
  ids.forEach(id => {
    const el = document.getElementById(id);
    const out = document.getElementById(id + "O");
    const sync = () => { out.value = el.value; recompute(); };
    el.addEventListener("input", sync);
    out.value = el.value;
  });
  document.querySelectorAll('input[name=scen]').forEach(r =>
    r.addEventListener("change", recompute));
  recompute();
  animateTurret();
}

function currentScenario() {
  return document.querySelector('input[name=scen]:checked').value;
}

function recompute() {
  const kp = +document.getElementById("kp").value;
  const ki = +document.getElementById("ki").value;
  const kd = +document.getElementById("kd").value;
  const slew = +document.getElementById("slew").value;
  const json = runSim(kp, ki, kd, slew, currentScenario());
  tuner.data = JSON.parse(json);
  tuner.frame = 0;
  drawPlot();
  showMetrics();
}

function showMetrics() {
  const m = tuner.data.metrics;
  const parts = Object.entries(m).map(([k,v]) => `${k}=${v}`).join("   ");
  document.getElementById("metrics").textContent = parts;
}

// Map angle (deg) -> canvas px, given axis ranges.
function mapper(w, h, axRange, elRange, pad=34) {
  const [a0,a1] = axRange, [e0,e1] = elRange;
  return {
    x: a => pad + (a - a0) / (a1 - a0) * (w - 2*pad),
    y: e => h - pad - (e - e0) / (e1 - e0) * (h - 2*pad),
  };
}

function drawAxes(ctx, w, h, M, xlabel, ylabel) {
  ctx.strokeStyle = COL.grid; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(M.x(0), 0); ctx.lineTo(M.x(0), h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, M.y(0)); ctx.lineTo(w, M.y(0)); ctx.stroke();
  ctx.fillStyle = COL.muted; ctx.font = "11px monospace";
  ctx.fillText(xlabel, w-90, h-8); ctx.save();
  ctx.translate(10, 60); ctx.rotate(-Math.PI/2); ctx.fillText(ylabel, 0, 0); ctx.restore();
}

function animateTurret() {
  const canvas = document.getElementById("vizTurret");
  const { ctx, w, h } = fit(canvas);
  const d = tuner.data;
  if (d) {
    const M = mapper(w, h, [-32,32], [-20,20]);
    ctx.clearRect(0,0,w,h);
    drawAxes(ctx, w, h, M, "azimuth°", "elev°");
    const i = tuner.frame % d.t.length;
    // target trail
    ctx.strokeStyle = "rgba(248,81,73,0.35)"; ctx.lineWidth = 1; ctx.beginPath();
    for (let k=0;k<=i;k++){ const fn=k?"lineTo":"moveTo"; ctx[fn](M.x(d.taz[k]),M.y(d.tel[k])); }
    ctx.stroke();
    const ta=d.taz[i], te=d.tel[i], pa=d.pan[i], pe=d.tilt[i];
    // link
    ctx.strokeStyle = COL.amber; ctx.beginPath();
    ctx.moveTo(M.x(pa),M.y(pe)); ctx.lineTo(M.x(ta),M.y(te)); ctx.stroke();
    // target
    ctx.fillStyle = COL.red; ctx.beginPath(); ctx.arc(M.x(ta),M.y(te),9,0,7); ctx.fill();
    // aim crosshair
    ctx.strokeStyle = COL.green; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(M.x(pa)-12,M.y(pe)); ctx.lineTo(M.x(pa)+12,M.y(pe));
    ctx.moveTo(M.x(pa),M.y(pe)-12); ctx.lineTo(M.x(pa),M.y(pe)+12); ctx.stroke();
    ctx.fillStyle = COL.muted; ctx.font="11px monospace";
    ctx.fillText("● target", w-92, 18); ctx.fillStyle=COL.green; ctx.fillText("✛ turret aim", w-92, 34);
    tuner.frame = (tuner.frame + 1) % d.t.length;
  }
  tuner.raf = requestAnimationFrame(() => setTimeout(animateTurret, 1000/30));
}

function drawPlot() {
  const canvas = document.getElementById("vizPlot");
  const { ctx, w, h } = fit(canvas);
  const d = tuner.data; if (!d) return;
  ctx.clearRect(0,0,w,h);
  const tmax = d.t[d.t.length-1];
  let lo=0, hi=0;
  for (const a of d.taz.concat(d.pan)) { lo=Math.min(lo,a); hi=Math.max(hi,a); }
  const pad=34, M={ x:t=>pad+t/tmax*(w-2*pad), y:a=>h-pad-(a-lo)/(hi-lo||1)*(h-2*pad) };
  // grid baseline
  ctx.strokeStyle=COL.grid; ctx.beginPath(); ctx.moveTo(0,M.y(0)); ctx.lineTo(w,M.y(0)); ctx.stroke();
  // target (dashed)
  ctx.strokeStyle=COL.red; ctx.setLineDash([5,4]); ctx.lineWidth=1.4; ctx.beginPath();
  d.t.forEach((t,k)=>{const fn=k?"lineTo":"moveTo";ctx[fn](M.x(t),M.y(d.taz[k]));}); ctx.stroke();
  ctx.setLineDash([]);
  // pan (solid)
  ctx.strokeStyle=COL.green; ctx.lineWidth=2.2; ctx.beginPath();
  d.t.forEach((t,k)=>{const fn=k?"lineTo":"moveTo";ctx[fn](M.x(t),M.y(d.pan[k]));}); ctx.stroke();
  ctx.fillStyle=COL.muted; ctx.font="11px monospace";
  ctx.fillText("azimuth response — target (red) vs pan (green)", pad, 16);
  ctx.fillText("time →", w-58, h-8);
}

/* ============================================================
   PANEL 2 — Safety-gate playground (drag the person)
   ============================================================ */
let safety = { person:{x:560,y:70,w:90,h:230}, target:[380,150,470,260], drag:false, off:[0,0] };

function initSafety() {
  const canvas = document.getElementById("vizSafety");
  canvas.addEventListener("mousedown", e => {
    const p = pos(canvas,e), b=safety.person;
    if (p.x>=b.x && p.x<=b.x+b.w && p.y>=b.y && p.y<=b.y+b.h){
      safety.drag=true; safety.off=[p.x-b.x, p.y-b.y];
    }
  });
  window.addEventListener("mousemove", e => {
    if(!safety.drag) return;
    const p = pos(canvas,e);
    safety.person.x = Math.max(0, Math.min(canvas.clientWidth-safety.person.w, p.x-safety.off[0]));
    safety.person.y = Math.max(0, Math.min(canvas.clientHeight-safety.person.h, p.y-safety.off[1]));
    drawSafety();
  });
  window.addEventListener("mouseup", ()=>safety.drag=false);
  ["targetClass","armed","locked"].forEach(id =>
    document.getElementById(id).addEventListener("change", drawSafety));
  drawSafety();
}

function pos(canvas, e){
  const r = canvas.getBoundingClientRect();
  return { x:(e.clientX-r.left), y:(e.clientY-r.top) };
}

function drawSafety() {
  const canvas = document.getElementById("vizSafety");
  const { ctx, w, h } = fit(canvas);
  const label = document.getElementById("targetClass").value;
  const armed = document.getElementById("armed").checked;
  const locked = document.getElementById("locked").checked;
  const b = safety.person;
  const pbox = [b.x, b.y, b.x+b.w, b.y+b.h];
  const res = JSON.parse(evalSafety(label, safety.target, pbox, armed, locked));

  ctx.clearRect(0,0,w,h);
  // target box
  const [x1,y1,x2,y2] = safety.target;
  const targetColour = res.permit ? COL.red : COL.green;
  ctx.strokeStyle = targetColour; ctx.lineWidth = 2.5;
  ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  ctx.fillStyle = targetColour; ctx.font="12px monospace"; ctx.fillText(label, x1, y1-6);
  // lock crosshair
  if (locked){ const cx=(x1+x2)/2, cy=(y1+y2)/2; ctx.strokeStyle=COL.green;
    ctx.beginPath(); ctx.moveTo(cx-9,cy); ctx.lineTo(cx+9,cy); ctx.moveTo(cx,cy-9); ctx.lineTo(cx,cy+9); ctx.stroke(); }
  // person box
  ctx.strokeStyle = COL.blue; ctx.lineWidth = 2.5; ctx.strokeRect(b.x,b.y,b.w,b.h);
  ctx.fillStyle = COL.blue; ctx.fillText("person ⇕ drag me", b.x, b.y+b.h+16);

  // HUD
  const hud = document.getElementById("gateHud");
  if (res.permit){ hud.textContent = "● ARMED — CLEAR TO FIRE"; hud.style.color = COL.red; }
  else { hud.textContent = "○ BLOCKED — " + res.reason; hud.style.color = COL.amber; }
}

boot().catch(err => {
  document.getElementById("loadmsg").textContent = "Failed to start: " + err;
});
