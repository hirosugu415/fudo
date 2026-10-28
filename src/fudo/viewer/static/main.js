// 最小のビューア：地形の高さ、川の線、月ごとの流量と積雪。見た目には凝らない（SPEC 第13章）。
import * as THREE from "three";
import { OrbitControls } from "/static/vendor/three/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const cfg = await (await fetch("/api/config")).json();
$("speed").value = cfg.playback_months_per_second;

// ---- three.js の準備 ----
const view = $("view");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1d21);
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 500);
camera.position.set(0, 22, 26);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 1.0);
sun.position.set(-20, 30, -10);
scene.add(sun);

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

// ---- 世界 ----
let world = null;  // { id, nx, ny, dxKm, elev, sea, lake, alluv, meanQ, receiver, mesh, baseColors, river }
let run = null;    // { id, startIndex, written, total }

async function fetchArray(url, Type) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return new Type(await r.arrayBuffer());
}

// ない変数（ウォームアップの年には積雪などがない）は null を返す
async function fetchOptional(url, Type) {
  const r = await fetch(url);
  if (!r.ok) return null;
  return new Type(await r.arrayBuffer());
}

// 土地利用の色（0 未開、1 水田、2 畑、3 集落、4 商業地）
const USE_COLORS = { 1: [0.55, 0.75, 0.35], 2: [0.72, 0.62, 0.40], 3: [0.62, 0.58, 0.55], 4: [0.80, 0.35, 0.30] };
const USE_LABELS = ["未開", "水田", "畑", "集落", "商業地"];

function landColor(h, alluvial) {
  // 標高で緑→茶→灰→白。沖積層は黄緑寄り。
  const stops = [[0, [0.36, 0.56, 0.30]], [400, [0.52, 0.52, 0.32]], [1000, [0.50, 0.42, 0.32]],
                 [1800, [0.62, 0.62, 0.62]], [2600, [0.95, 0.95, 0.95]]];
  let c = stops[stops.length - 1][1];
  for (let i = 0; i < stops.length - 1; i++) {
    const [h0, c0] = stops[i], [h1, c1] = stops[i + 1];
    if (h <= h1) { const t = Math.max(0, (h - h0) / (h1 - h0)); c = c0.map((v, k) => v + t * (c1[k] - v)); break; }
  }
  if (alluvial) c = [c[0] * 0.6 + 0.66 * 0.4, c[1] * 0.6 + 0.70 * 0.4, c[2] * 0.6 + 0.36 * 0.4];
  return c;
}

function vertexY(i) {
  const h = world.sea[i] ? 0 : world.elev[i];
  return (h / 1000) * cfg.vertical_exaggeration;
}

function buildTerrain() {
  const { nx, ny, dxKm } = world;
  const n = nx * ny;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = Math.floor(i / nx), c = i % nx;
    pos[3 * i] = (c + 0.5) * dxKm - (nx * dxKm) / 2;
    pos[3 * i + 1] = vertexY(i);
    pos[3 * i + 2] = (r + 0.5) * dxKm - (ny * dxKm) / 2;
    let rgb;
    if (world.sea[i]) { const d = Math.min(1, -world.elev[i] / 150); rgb = [0.12 - 0.06 * d, 0.30 - 0.12 * d, 0.55 - 0.15 * d]; }
    else if (world.lake[i]) rgb = [0.30, 0.50, 0.80];
    else rgb = landColor(world.elev[i], world.alluv[i] > 2);
    col.set(rgb, 3 * i);
  }
  const idx = new Uint32Array((nx - 1) * (ny - 1) * 6);
  let k = 0;
  for (let r = 0; r < ny - 1; r++) for (let c = 0; c < nx - 1; c++) {
    const a = r * nx + c, b = a + 1, d = a + nx, e = d + 1;
    idx.set([a, d, b, b, d, e], k); k += 6;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true }));
  world.baseColors = col.slice();
  world.mesh = mesh;
  scene.add(mesh);
}

function buildRivers() {
  const { nx, receiver, meanQ } = world;
  const segs = [];
  for (let i = 0; i < receiver.length; i++) {
    if (world.sea[i] || world.lake[i] || meanQ[i] < cfg.river_display_min_m3s) continue;
    const r = receiver[i];
    if (r < 0) continue;
    segs.push([i, r]);
  }
  const pos = new Float32Array(segs.length * 4 * 3);
  const idx = new Uint32Array(segs.length * 6);
  for (let s = 0; s < segs.length; s++) idx.set([4 * s, 4 * s + 1, 4 * s + 2, 4 * s + 2, 4 * s + 1, 4 * s + 3], 6 * s);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  const m = new THREE.MeshBasicMaterial({ color: 0x3f8cff, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -4 });
  const mesh = new THREE.Mesh(g, m);
  world.river = { segs, mesh };
  scene.add(mesh);
  updateRivers(meanQ);
}

function updateRivers(Q) {
  const { segs, mesh } = world.river;
  const p = world.mesh.geometry.attributes.position.array;
  const pos = mesh.geometry.attributes.position.array;
  const lift = 0.004 * cfg.vertical_exaggeration;
  for (let s = 0; s < segs.length; s++) {
    const [i, r] = segs[s];
    // 流量が0の月（渇水・凍結）は描かない。それ以外は太さの下限を設けて小川も見えるようにする
    const q = Math.max(Q[i], 0);
    const wm = q > 0 ? Math.max(cfg.river_min_width_m, cfg.river_width_per_sqrt_m3s * Math.sqrt(q)) : 0;
    const w = wm / 1000 / 2;
    const x0 = p[3 * i], z0 = p[3 * i + 2], x1 = p[3 * r], z1 = p[3 * r + 2];
    const y = Math.max(p[3 * i + 1], p[3 * r + 1]) + lift;
    let dx = x1 - x0, dz = z1 - z0; const L = Math.hypot(dx, dz) || 1;
    const ox = (-dz / L) * w, oz = (dx / L) * w;
    pos.set([x0 + ox, y, z0 + oz, x0 - ox, y, z0 - oz, x1 + ox, y, z1 + oz, x1 - ox, y, z1 - oz], 12 * s);
  }
  mesh.geometry.attributes.position.needsUpdate = true;
  mesh.geometry.computeBoundingSphere();
}

function updateColors(snow, landUse) {
  const col = world.mesh.geometry.attributes.color.array;
  const base = world.baseColors;
  const n = world.nx * world.ny;
  for (let i = 0; i < n; i++) {
    let rgb = [base[3 * i], base[3 * i + 1], base[3 * i + 2]];
    const u = landUse ? landUse[i] : 0;
    if (u > 0 && !world.sea[i]) rgb = USE_COLORS[u];
    const t = (snow && !world.sea[i]) ? Math.min(1, snow[i] / 100) : 0;
    for (let k = 0; k < 3; k++) col[3 * i + k] = rgb[k] * (1 - t) + 0.97 * t;
  }
  world.mesh.geometry.attributes.color.needsUpdate = true;
}

// 道：状態が一定以上のマスどうしをつなぐ帯
function updateRoads(road) {
  if (world.roads) { scene.remove(world.roads); world.roads.geometry.dispose(); world.roads.material.dispose(); world.roads = null; }
  if (!road) return;
  const { nx, ny } = world;
  const p = world.mesh.geometry.attributes.position.array;
  const vis = cfg.road_visible_level;
  const segs = [];
  for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) {
    const i = r * nx + c;
    if (road[i] < vis) continue;
    for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
      const rr = r + dr, cc = c + dc;
      if (rr >= ny || cc < 0 || cc >= nx) continue;
      const j = rr * nx + cc;
      if (road[j] >= vis) segs.push([i, j, Math.min(road[i], road[j])]);
    }
  }
  const pos = new Float32Array(segs.length * 12);
  const idx = new Uint32Array(segs.length * 6);
  const lift = 0.006 * cfg.vertical_exaggeration;
  segs.forEach(([i, j, lv], s) => {
    const w = (cfg.road_width_m * (0.5 + lv)) / 1000 / 2;
    const x0 = p[3 * i], z0 = p[3 * i + 2], x1 = p[3 * j], z1 = p[3 * j + 2];
    const y = Math.max(p[3 * i + 1], p[3 * j + 1]) + lift;
    const L = Math.hypot(x1 - x0, z1 - z0) || 1;
    const ox = (-(z1 - z0) / L) * w, oz = ((x1 - x0) / L) * w;
    pos.set([x0 + ox, y, z0 + oz, x0 - ox, y, z0 - oz, x1 + ox, y, z1 + oz, x1 - ox, y, z1 - oz], 12 * s);
    idx.set([4 * s, 4 * s + 1, 4 * s + 2, 4 * s + 2, 4 * s + 1, 4 * s + 3], 6 * s);
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  world.roads = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0x8a5a2b, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -3 }));
  scene.add(world.roads);
}

// 集落：世帯数に比例した高さの箱
function updateHouses(households) {
  if (world.houses) { scene.remove(world.houses); world.houses.geometry.dispose(); world.houses.material.dispose(); world.houses = null; }
  if (!households) return;
  const cells = [];
  for (let i = 0; i < households.length; i++) if (households[i] > 0) cells.push(i);
  if (cells.length === 0) return;
  const geo = new THREE.BoxGeometry(world.dxKm * 0.7, 1, world.dxKm * 0.7);
  geo.translate(0, 0.5, 0);
  const mesh = new THREE.InstancedMesh(geo, new THREE.MeshLambertMaterial({ color: 0xd8d0c0 }), cells.length);
  const p = world.mesh.geometry.attributes.position.array;
  const m = new THREE.Matrix4();
  cells.forEach((i, k) => {
    const hgt = (households[i] * cfg.house_height_m_per_household / 1000) * cfg.vertical_exaggeration;
    m.makeScale(1, Math.max(hgt, 0.001), 1);
    m.setPosition(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
    mesh.setMatrixAt(k, m);
  });
  world.houses = mesh;
  scene.add(mesh);
}

function clearWorld() {
  if (!world) return;
  for (const m of [world.mesh, world.river && world.river.mesh, world.roads, world.houses]) {
    if (m) { scene.remove(m); m.geometry.dispose(); m.material.dispose(); }
  }
  world = null;
}

async function loadWorld(id) {
  const meta = await (await fetch(`/api/worlds/${id}`)).json();
  const base = `/api/worlds/${id}/field/`;
  const [elev, sea, lake, alluv, meanQ] = await Promise.all([
    fetchArray(base + "elevation", Float32Array), fetchArray(base + "sea", Uint8Array),
    fetchArray(base + "lake", Uint8Array), fetchArray(base + "alluvium_m", Float32Array),
    fetchArray(base + "mean_discharge_m3s", Float32Array),
  ]);
  clearWorld();
  stopRun();
  world = { id, nx: meta.grid.nx, ny: meta.grid.ny, dxKm: meta.grid.cell_size_m / 1000, elev, sea, lake, alluv, meanQ,
            receiver: Int32Array.from(meta.receiver) };
  buildTerrain();
  buildRivers();
  $("desc").textContent = `シード ${meta.seed}（world_id ${id}）\n\n` + meta.description;
  $("seed").value = String(meta.seed);
  $("sim").disabled = false;
  $("time").textContent = "平年値（年平均流量）";
}

// ---- 操作：マップを生成 ----
$("gen").onclick = async () => {
  const text = $("seed").value.trim();
  const body = text === "" ? {} : { seed: Number(text) };
  if (text !== "" && !/^\d+$/.test(text)) { $("status").textContent = "シードは0以上の整数にしてください"; return; }
  $("gen").disabled = true; $("sim").disabled = true;
  $("status").textContent = "世界を生成しています（初回は数十秒かかります）…";
  try {
    const r = await fetch("/api/worlds", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error(await r.text());
    const res = await r.json();
    await loadWorld(res.world_id);
    $("status").textContent = `生成しました（シード ${res.seed}）。気に入らなければシードを消してもう一度「マップを生成」。`;
  } catch (e) { $("status").textContent = "生成に失敗しました: " + e.message; }
  $("gen").disabled = false;
};

// ---- 操作：シミュレート開始 ----
let ws = null, playing = false, timer = null, loading = false;

function stopRun() {
  if (ws) { ws.close(); ws = null; }
  run = null; playing = false; clearTimeout(timer);
  $("slider").disabled = true; $("play").disabled = true; $("play").textContent = "再生"; $("progress").textContent = "";
}

$("sim").onclick = async () => {
  if (!world) return;
  const r = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ world_id: world.id }) });
  const res = await r.json();
  stopRun();
  run = { id: res.run_id, written: 0, total: 0, startIndex: 0 };
  $("status").textContent = `シミュレーション ${res.run_id} を開始しました。`;
  ws = new WebSocket(`ws://${location.host}/ws/runs/${res.run_id}`);
  ws.onmessage = (ev) => {
    const s = JSON.parse(ev.data);
    if (!run || s.run_id !== run.id) return;
    const first = run.written === 0 && s.months_written > 0;
    Object.assign(run, { written: s.months_written, total: s.months_total, times: s.times });
    $("slider").max = String(Math.max(0, run.written - 1));
    $("progress").textContent = `計算済み ${run.written} / ${run.total}` + (s.complete ? "（完了）" : "");
    if (first) { $("slider").disabled = false; $("play").disabled = false; showMonth(0); }
  };
};

function ymLabel(k) {
  const [y, m] = run.times[k];
  return m === 0 ? `${y}年（ウォームアップ）` : `${y}年${m}月`;
}

async function showMonth(k) {
  if (!run || loading || k >= run.written) return;
  loading = true;
  try {
    const base = `/api/runs/${run.id}/month/${k}`;
    const f = (name) => fetchOptional(`${base}/field/${name}`, Float32Array);
    const [q, snow, landUse, road, households, pop, price, sd, ss, inn, meta] = await Promise.all([
      f("discharge_m3s"), f("snow_mm"), f("land_use"), f("road"), f("households"), f("population"),
      f("land_price"), f("shops_daily"), f("shops_specialty"), f("inns"), (await fetch(base + "/meta")).json(),
    ]);
    if (q) updateRivers(q);
    updateColors(snow, landUse); updateRoads(road); updateHouses(households);
    run.cur = { landUse, households, pop, price, road, sd, ss, inn };
    $("slider").value = String(k);
    const stat = meta.stat_population !== undefined
      ? `　人口 ${Math.round(meta.stat_population)}・世帯 ${Math.round(meta.stat_households)}・店 ${Math.round(meta.stat_shops)}` : "";
    $("time").textContent = `${ymLabel(k)}　気温の平年差 ${meta.temperature_anomaly_c.toFixed(2)}℃・降水 ×${meta.precip_factor.toFixed(2)}${stat}`;
  } finally { loading = false; }
}

$("slider").oninput = () => showMonth(Number($("slider").value));
$("play").onclick = () => {
  playing = !playing;
  $("play").textContent = playing ? "一時停止" : "再生";
  if (playing) tick();
};
function tick() {
  if (!playing || !run) return;
  const k = Number($("slider").value) + 1;
  // 計算が追いついていない月には進まない
  if (k < run.written) showMonth(k);
  const sp = Math.max(0.1, Number($("speed").value) || cfg.playback_months_per_second);
  timer = setTimeout(tick, 1000 / sp);
}

// ---- マスのクリック：数値を表示 ----
const ray = new THREE.Raycaster();
renderer.domElement.addEventListener("click", (ev) => {
  if (!world) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const v = new THREE.Vector2(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
  ray.setFromCamera(v, camera);
  const hit = ray.intersectObject(world.mesh)[0];
  if (!hit) return;
  const c = Math.round((hit.point.x + (world.nx * world.dxKm) / 2) / world.dxKm - 0.5);
  const r = Math.round((hit.point.z + (world.ny * world.dxKm) / 2) / world.dxKm - 0.5);
  if (r < 0 || c < 0 || r >= world.ny || c >= world.nx) return;
  const i = r * world.nx + c;
  let txt = `マス (${r}, ${c})\n標高 ${world.elev[i].toFixed(0)} m\n` +
    (world.sea[i] ? "海" : world.lake[i] ? "湖" : "陸") + `\n沖積層 ${world.alluv[i].toFixed(1)} m\n年平均流量 ${world.meanQ[i].toFixed(2)} m³/s`;
  const cur = run && run.cur;
  if (cur && cur.landUse) {
    txt += `\n用途 ${USE_LABELS[cur.landUse[i]]}\n人口 ${cur.pop[i]}・世帯 ${cur.households[i]}` +
      `\n地価 ${cur.price[i].toFixed(1)} 円/ha\n道 ${cur.road[i].toFixed(2)}` +
      `\n店 日用品 ${cur.sd[i]}・専門 ${cur.ss[i]}・宿 ${cur.inn[i]}`;
  }
  $("cell").textContent = txt;
});
