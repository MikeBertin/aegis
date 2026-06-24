/* AEGIS fire-control demo — drives the real ballistics.py / stereo.py (Pyodide). */
"use strict";

const COL = {
  bg:"#010409", fg:"#c9d1d9", muted:"#8b949e", grid:"#21262d",
  green:"#3fb950", red:"#f85149", amber:"#d29922", blue:"#58a6ff",
};

const PY_DRIVER = `
import sys, json, math
sys.path.insert(0, "/home/pyodide")
from aegis.ballistics import DartModel, firing_solution, simulate_shot, G
from aegis.stereo import StereoRig

_rig = StereoRig(focal_px=800.0, baseline_m=0.06, cx=320.0, cy=240.0)

def _traj(az, el, dart, rng, gravity):
    a, e = math.radians(az), math.radians(el)
    d = (math.sin(a)*math.cos(e), math.sin(e), math.cos(a)*math.cos(e))
    pos = [0.0, 0.0, 0.0]
    vel = [d[0]*dart.muzzle_speed, d[1]*dart.muzzle_speed, d[2]*dart.muzzle_speed]
    pts, t, dt = [], 0.0, 0.005
    while t < 2.5 and pos[2] < rng + 0.4 and pos[1] > -3.0:
        pts.append([pos[0], pos[1], pos[2]])
        sp = math.sqrt(vel[0]**2+vel[1]**2+vel[2]**2)
        ax = -dart.drag_k*sp*vel[0]
        ay = (-G if gravity else 0.0) - dart.drag_k*sp*vel[1]
        az_ = -dart.drag_k*sp*vel[2]
        pos = [pos[0]+vel[0]*dt, pos[1]+vel[1]*dt, pos[2]+vel[2]*dt]
        vel = [vel[0]+ax*dt, vel[1]+ay*dt, vel[2]+az_*dt]
        t += dt
    return pts

def solve(rng, vlat, speed, gravity, drag_k):
    dart = DartModel(float(speed), float(drag_k))
    p = (0.0, 0.0, float(rng))          # target at turret height, straight ahead
    v = (float(vlat), 0.0, 0.0)         # crossing right
    sol = firing_solution(p, v, dart, gravity)
    el0 = math.degrees(math.atan2(p[1], math.hypot(p[0], p[2])))  # straight-at-target elevation
    sol_pts = _traj(sol.aim_az, sol.aim_el, dart, rng, gravity) if sol.ok else []
    naive_pts = _traj(0.0, el0, dart, rng, gravity)
    hit, ht, hclose = simulate_shot(sol.aim_az, sol.aim_el, dart, p, v, gravity) if sol.ok else (False, 0, 9)
    nhit, nt, nclose = simulate_shot(0.0, el0, dart, p, v, gravity)
    return json.dumps({
        "ok": sol.ok, "tof": sol.tof, "lead": round(sol.lead_deg, 1),
        "holdover": round(sol.holdover_deg, 1),
        "intercept": list(sol.intercept),
        "hit": hit, "closest_cm": round(hclose*100), "hit_t": ht,
        "naive_hit": nhit, "naive_cm": round(nclose*100),
        "sol_pts": sol_pts, "naive_pts": naive_pts,
        "rng": rng, "vlat": vlat, "speed": speed, "gravity": gravity,
        "drag_k": drag_k, "speed_at": round(dart.speed_at(rng), 1),
        "disparity": round(_rig.disparity(rng), 1),
        "err3": round(_rig.depth_error(3.0)*100), "err6": round(_rig.depth_error(6.0)*100),
        "err_here": round(_rig.depth_error(rng)*100),
    })
`;

let solve;

async function boot() {
  const pyodide = await loadPyodide();
  document.getElementById("loadmsg").textContent = "Loading AEGIS modules…";
  try { pyodide.FS.mkdir("/home/pyodide/aegis"); } catch (e) {}
  for (const [path, src] of Object.entries(window.AEGIS_MODULES)) {
    pyodide.FS.writeFile("/home/pyodide/" + path, src);
  }
  pyodide.runPython(PY_DRIVER);
  solve = pyodide.globals.get("solve");
  document.getElementById("loading").style.display = "none";
  init();
}

function fit(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

let state = { data:null, frame:0 };

function init() {
  const units = { speed:" m/s", range:" m", vlat:" m/s", drag:" /m" };
  ["speed","range","vlat","drag"].forEach(id => {
    const el = document.getElementById(id), out = document.getElementById(id + "O");
    const sync = () => { out.value = el.value + units[id]; recompute(); };
    el.addEventListener("input", sync); out.value = el.value + units[id];
  });
  document.getElementById("gravity").addEventListener("change", recompute);
  recompute();
  loop();
}

function recompute() {
  const speed = +document.getElementById("speed").value;
  const rng = +document.getElementById("range").value;
  const vlat = +document.getElementById("vlat").value;
  const drag = +document.getElementById("drag").value;
  const gravity = document.getElementById("gravity").checked;
  state.data = JSON.parse(solve(rng, vlat, speed, gravity, drag));
  state.frame = 0;
  drawStereo();
  drawSide();
  updateVerdict();
  const note = document.getElementById("dragNote");
  if (note) note.textContent = drag > 0
    ? `drag: ${speed} m/s muzzle → ${state.data.speed_at} m/s at ${rng} m`
    : "no drag (ideal constant-speed dart)";
}

function updateVerdict() {
  const d = state.data;
  const v = document.getElementById("verdict");
  if (!d.ok) { v.textContent = "no solution — target outruns the dart"; v.style.color = COL.amber; return; }
  v.innerHTML =
    `<span style="color:${COL.green}">SOLUTION:</span> ` +
    `TOF ${(d.tof*1000).toFixed(0)} ms · lead ${d.lead}° · hold-over ${d.holdover}° · ` +
    `<b style="color:${d.hit?COL.green:COL.red}">${d.hit?"HIT":"MISS"}</b> (${d.closest_cm} cm)` +
    `&nbsp;&nbsp;|&nbsp;&nbsp;<span style="color:${COL.muted}">naive aim:</span> ` +
    `<b style="color:${d.naive_hit?COL.green:COL.red}">${d.naive_hit?"HIT":"MISS"}</b> (${d.naive_cm} cm)`;
  v.style.color = COL.fg;
}

/* ---- stereo ranging panel ---- */
function drawStereo() {
  const { ctx, w, h } = fit(document.getElementById("stereoCanvas"));
  const d = state.data;
  ctx.clearRect(0,0,w,h);
  // two camera frames
  const cw = w*0.26, gap = 14, x0 = 10, y0 = 16, ch = h-50;
  const dispFrac = Math.min(0.45, d.disparity/120);  // visual disparity
  for (let i=0;i<2;i++){
    const x = x0 + i*(cw+gap);
    ctx.strokeStyle = COL.grid; ctx.strokeRect(x, y0, cw, ch);
    ctx.fillStyle = COL.muted; ctx.font="11px monospace";
    ctx.fillText(i?"right cam":"left cam", x+6, y0-4);
    // target dot shifted by disparity between the two views
    const dotX = x + cw*0.5 + (i? -dispFrac*cw : dispFrac*cw)*0.5;
    ctx.fillStyle = COL.red; ctx.beginPath(); ctx.arc(dotX, y0+ch*0.5, 7, 0, 7); ctx.fill();
  }
  // depth scale bar on the right
  const bx = x0 + 2*(cw+gap) + 14;
  ctx.fillStyle = COL.fg; ctx.font="12px monospace";
  ctx.fillText(`range ${d.rng.toFixed(1)} m  →  disparity ${d.disparity}px`, bx, y0+18);
  ctx.fillStyle = COL.muted;
  ctx.fillText(`range uncertainty here: ±${d.err_here} cm`, bx, y0+42);
  ctx.fillText(`(±${d.err3} cm @ 3 m   vs   ±${d.err6} cm @ 6 m — quadratic)`, bx, y0+62);
  document.getElementById("stereoReadout").textContent =
    `Z = focal·baseline / disparity = 800px · 0.06m / ${d.disparity}px = ${d.rng.toFixed(2)} m`;
}

/* ---- side view: gravity arc + holdover ---- */
function drawSide() {
  const { ctx, w, h } = fit(document.getElementById("sideCanvas"));
  const d = state.data;
  ctx.clearRect(0,0,w,h);
  const pad=36, rng=d.rng;
  // world->canvas: z (0..rng+0.4) horizontal; y (-0.6..+0.7) vertical
  const zmax = rng+0.4, ymin=-0.6, ymax=0.7;
  const X = z => pad + z/zmax*(w-2*pad);
  const Y = y => h-pad - (y-ymin)/(ymax-ymin)*(h-2*pad);
  // ground / turret-height line
  ctx.strokeStyle = COL.grid; ctx.beginPath(); ctx.moveTo(0,Y(0)); ctx.lineTo(w,Y(0)); ctx.stroke();
  ctx.fillStyle = COL.muted; ctx.font="11px monospace";
  ctx.fillText("turret height", 6, Y(0)-4); ctx.fillText("range (m) →", w-90, h-10);
  // target at (rng, 0)
  ctx.fillStyle = COL.red; ctx.beginPath(); ctx.arc(X(rng), Y(0), 8, 0, 7); ctx.fill();
  ctx.fillText("target", X(rng)-18, Y(0)-12);
  // naive arc (grey) then solution arc (green)
  plotArc(ctx, d.naive_pts, X, Y, COL.muted, 1.4, true);
  plotArc(ctx, d.sol_pts, X, Y, COL.green, 2.2, false);
  ctx.fillStyle = COL.green; ctx.fillText(`hold-over ${d.holdover}°`, pad, 18);
  ctx.fillStyle = COL.muted; ctx.fillText("grey = naive (drops low)", pad, 34);
}

function plotArc(ctx, pts, X, Y, colour, lw, dashed) {
  if (!pts.length) return;
  ctx.strokeStyle = colour; ctx.lineWidth = lw;
  ctx.setLineDash(dashed?[5,4]:[]);
  ctx.beginPath();
  pts.forEach((p,i) => { const fn=i?"lineTo":"moveTo"; ctx[fn](X(p[2]), Y(p[1])); });
  ctx.stroke(); ctx.setLineDash([]);
}

/* ---- top-down: lead + animated darts ---- */
function loop() {
  const { ctx, w, h } = fit(document.getElementById("topCanvas"));
  const d = state.data;
  if (d) {
    const rng=d.rng, xspan=Math.max(2.0, Math.abs(d.intercept[0])+1.2);
    const pad=36;
    const X = x => w/2 + x/xspan*(w/2-pad);
    const Z = z => h-pad - z/(rng+0.4)*(h-2*pad);
    ctx.clearRect(0,0,w,h);
    // axes
    ctx.strokeStyle = COL.grid; ctx.beginPath();
    ctx.moveTo(w/2,0); ctx.lineTo(w/2,h); ctx.stroke();
    ctx.fillStyle = COL.muted; ctx.font="11px monospace";
    ctx.fillText("lateral (m)", w-78, h-10); ctx.fillText("turret", w/2+4, h-8);
    // turret
    ctx.fillStyle = COL.accent || "#76b900"; ctx.fillStyle="#76b900";
    ctx.beginPath(); ctx.arc(w/2, Z(0), 6, 0, 7); ctx.fill();
    // intercept marker
    if (d.ok){ ctx.strokeStyle=COL.amber; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.arc(X(d.intercept[0]), Z(d.intercept[2]), 8, 0, 7); ctx.stroke();
      ctx.fillStyle=COL.amber; ctx.fillText("intercept", X(d.intercept[0])+10, Z(d.intercept[2])); }

    const n = Math.max(d.sol_pts.length, d.naive_pts.length, 1);
    const i = state.frame % n;
    const tnow = i*0.005;
    // target position now (crossing right at vlat)
    const tx = d.vlat*tnow;
    ctx.fillStyle = COL.red; ctx.beginPath(); ctx.arc(X(tx), Z(rng), 8, 0, 7); ctx.fill();
    // darts
    drawDart(ctx, d.sol_pts, i, X, Z, COL.green);
    drawDart(ctx, d.naive_pts, i, X, Z, COL.muted);
    ctx.fillStyle=COL.green; ctx.fillText("● solution dart", 12, 18);
    ctx.fillStyle=COL.muted; ctx.fillText("● naive dart", 12, 34);
    ctx.fillStyle=COL.red; ctx.fillText("● target", 12, 50);
    state.frame = (state.frame+1) % (n+25);  // pause at end
  }
  requestAnimationFrame(() => setTimeout(loop, 1000/60));
}

function drawDart(ctx, pts, i, X, Z, colour) {
  if (!pts.length) return;
  const j = Math.min(i, pts.length-1);
  const p = pts[j];
  ctx.fillStyle = colour; ctx.beginPath(); ctx.arc(X(p[0]), Z(p[2]), 4, 0, 7); ctx.fill();
}

boot().catch(err => { document.getElementById("loadmsg").textContent = "Failed to start: " + err; });
