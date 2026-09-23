/* Finland climate probability dashboard — frontend.
 * Static-file architecture: fetches meta.json + per-(variable, ref-year, week)
 * histogram binaries produced by scripts/export_web_assets.py. All statistics
 * (tail shares, ratios, medians, percentiles) are computed client-side from
 * two histograms. Language policy: strictly descriptive, no causal claims.
 */

const DATA_BASE = "../data/web";
const N_BINS = 170;
const BIN_START = -45.0;
const BIN_WIDTH = 0.5;
const BASELINE_YEAR = 2000;
const GRID_COLS = 68;
const GRID_ROWS = 116;
const CELL_PX = 5;

const MONTHS = ["January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December"];
const MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]; // Feb 29 dropped

/* Palette (dataviz reference instance). Series colors also live in style.css. */
const PALETTE = {
  light: {
    surface: "#fcfcfb", mid: "#f0efec", bluePole: "#1c5cab", redPole: "#e34948",
    seq: ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
    notObserved: "#c3c2b7", ink: "#0b0b0b", muted: "#898781", grid: "#e1e0d9",
    seriesBase: "#2a78d6", seriesChosen: "#1baf7a",
  },
  dark: {
    surface: "#1a1a19", mid: "#383835", bluePole: "#3987e5", redPole: "#e66767",
    seq: ["#10315e", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
    notObserved: "#52514e", ink: "#ffffff", muted: "#898781", grid: "#2c2c2a",
    seriesBase: "#3987e5", seriesChosen: "#199e70",
  },
};

/* ---------- small color math (sRGB <-> OKLab) for smooth ramps ---------- */

function hexToRgb(h) {
  return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
}
function srgbToLinear(c) { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; }
function linearToSrgb(c) {
  const v = c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
  return Math.max(0, Math.min(255, Math.round(v * 255)));
}
function rgbToOklab([r, g, b]) {
  const [lr, lg, lb] = [srgbToLinear(r), srgbToLinear(g), srgbToLinear(b)];
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
          1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
          0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function oklabToRgb([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [linearToSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
          linearToSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
          linearToSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s)];
}
function mixHex(hexA, hexB, t) {
  const a = rgbToOklab(hexToRgb(hexA)), b = rgbToOklab(hexToRgb(hexB));
  const [r, g, bl] = oklabToRgb([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
  return `rgb(${r},${g},${bl})`;
}
/* diverging: t in [-1, 1], negative -> blue pole, positive -> red pole */
function divergingColor(t, pal) {
  return t < 0 ? mixHex(pal.mid, pal.bluePole, -t) : mixHex(pal.mid, pal.redPole, t);
}
/* sequential: t in [0, 1] across the mode's anchor list */
function sequentialColor(t, pal) {
  const seq = pal.seq;
  const x = Math.max(0, Math.min(0.99999, t)) * (seq.length - 1);
  const i = Math.floor(x);
  return mixHex(seq[i], seq[i + 1], x - i);
}

/* ---------------------------- state & data ---------------------------- */

const state = {
  variable: "tmax",
  monthIdx: 6, day: 15,          // July 15 default
  threshold: 25.0,
  direction: "atleast",
  dirManual: false,
  refYear: 2025,
  layer: "ratio",
  selected: -1,                  // land-cell index
};

let META = null;
let cellIndexByRowCol = null;    // Int32Array(GRID_ROWS*GRID_COLS) -> land idx or -1
const histCache = new Map();     // "var/R/week" -> Uint16Array(nCells*N_BINS)

function pal() {
  return matchMedia("(prefers-color-scheme: dark)").matches ? PALETTE.dark : PALETTE.light;
}

function doyFromDate(monthIdx, day) {
  let d = 0;
  for (let m = 0; m < monthIdx; m++) d += MONTH_DAYS[m];
  return d + day - 1;            // 0-based, 365-day calendar
}
function dateFromDoy(doy) {
  let m = 0, d = doy;
  while (d >= MONTH_DAYS[m]) { d -= MONTH_DAYS[m]; m++; }
  return `${MONTHS[m]} ${d + 1}`;
}
function currentWeek() {
  const doy = doyFromDate(state.monthIdx, state.day);
  const doys = META.weeks.doys_0based;
  let best = 0, bestDist = 1e9;
  for (let k = 0; k < doys.length; k++) {
    const dist = Math.min(Math.abs(doys[k] - doy), 365 - Math.abs(doys[k] - doy));
    if (dist < bestDist) { bestDist = dist; best = k; }
  }
  return best;
}

async function fetchHist(variable, refYear, week) {
  const key = `${variable}/${refYear}/${week}`;
  if (histCache.has(key)) return histCache.get(key);
  const url = `${DATA_BASE}/hist/${variable}/R${refYear}/w${String(week).padStart(2, "0")}.bin`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch failed: ${url} (${resp.status})`);
  const arr = new Uint16Array(await resp.arrayBuffer()); // data written little-endian
  if (arr.length !== META.n_cells * N_BINS) throw new Error(`bad size: ${url}`);
  histCache.set(key, arr);
  return arr;
}

/* ------------------------ statistics from histograms ------------------------ */

function edgeIndex(threshold) { return Math.round((threshold - BIN_START) / BIN_WIDTH); }

/* observed share of days satisfying the threshold in the chosen direction */
function tailShare(hist, offset, threshold, direction) {
  const e = edgeIndex(threshold);
  let tail = 0, total = 0;
  for (let b = 0; b < N_BINS; b++) {
    const c = hist[offset + b];
    total += c;
    if (direction === "atleast" ? b >= e : b <= e) tail += c;
  }
  return { p: total ? tail / total : NaN, tail, total };
}
function cellMedian(hist, offset) {
  let total = 0;
  for (let b = 0; b < N_BINS; b++) total += hist[offset + b];
  let cum = 0;
  for (let b = 0; b < N_BINS; b++) {
    cum += hist[offset + b];
    if (cum >= total / 2) return BIN_START + (b + 0.5) * BIN_WIDTH;
  }
  return NaN;
}
function cellPercentile(hist, offset, threshold) {
  const e = edgeIndex(threshold);
  let below = 0, total = 0;
  for (let b = 0; b < N_BINS; b++) {
    const c = hist[offset + b];
    total += c;
    if (b < e) below += c;
  }
  return total ? (100 * below) / total : NaN;
}
function pooledMedian(hist) {
  const pooled = new Float64Array(N_BINS);
  for (let i = 0; i < hist.length; i++) pooled[i % N_BINS] += hist[i];
  let total = 0;
  for (let b = 0; b < N_BINS; b++) total += pooled[b];
  let cum = 0;
  for (let b = 0; b < N_BINS; b++) {
    cum += pooled[b];
    if (cum >= total / 2) return BIN_START + (b + 0.5) * BIN_WIDTH;
  }
  return 0;
}

/* ------------------------------ formatting ------------------------------ */

function fmtShare(p) {
  if (isNaN(p)) return "–";
  if (p === 0) return "0%";
  if (p < 0.001) return "<0.1%";
  return p < 0.095 ? `${(p * 100).toFixed(1)}%` : `${Math.round(p * 100)}%`;
}
function fmtRatio(r) {
  if (r >= 100) return Math.round(r).toString();
  if (r >= 10) return r.toFixed(1).replace(/\.0$/, "");
  if (r >= 0.1) return r.toFixed(r >= 3 ? 1 : 2).replace(/(\.\d)0$/, "$1").replace(/\.00$/, "");
  return r.toFixed(2);
}
function windowLabel(refYear) { return `${refYear - 29}–${refYear}`; }
function dirWord() { return state.direction === "atleast" ? "at least" : "at most"; }
function fmtTemp(x) { return `${x % 1 === 0 ? x.toFixed(0) : x.toFixed(1)} °C`; }
function ordinal(n) {
  const v = n % 100;
  return `${n}${v >= 11 && v <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] || "th"}`;
}

/* -------------------------------- map -------------------------------- */

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const mapTooltip = document.getElementById("map-tooltip");
let hatchPattern = null;

function makeHatch(p) {
  const off = document.createElement("canvas");
  off.width = off.height = 8;
  const c = off.getContext("2d");
  c.fillStyle = p.surface;
  c.fillRect(0, 0, 8, 8);
  c.strokeStyle = p.notObserved;
  c.lineWidth = 1.5;
  c.beginPath();
  c.moveTo(-2, 6); c.lineTo(6, -2);
  c.moveTo(2, 10); c.lineTo(10, 2);
  c.stroke();
  return ctx.createPattern(off, "repeat");
}

/* per-render cell values; also consumed by tooltip + detail view */
let mapValues = null;   // Float64Array per land cell: layer value (NaN = not observed)
let mapShares = null;   // {pChosen, pBase} Float64Arrays
let histChosen = null, histBase = null;

function computeMapValues() {
  const n = META.n_cells;
  mapValues = new Float64Array(n);
  const pChosen = new Float64Array(n), pBase = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const off = i * N_BINS;
    const sc = tailShare(histChosen, off, state.threshold, state.direction);
    const sb = tailShare(histBase, off, state.threshold, state.direction);
    pChosen[i] = sc.p; pBase[i] = sb.p;
    if (state.layer === "ratio") {
      mapValues[i] = sb.tail === 0 ? NaN : sc.p / sb.p;
    } else if (state.layer === "prob") {
      mapValues[i] = sc.p;
    } else {
      mapValues[i] = state.threshold - cellMedian(histBase, off);
    }
  }
  mapShares = { pChosen, pBase };
}

function cellColor(i, p, probMax) {
  const v = mapValues[i];
  if (state.layer === "ratio") {
    if (isNaN(v)) return null; // not observed -> hatch
    if (v === 0) return divergingColor(-1, p);
    return divergingColor(Math.max(-1, Math.min(1, Math.log2(v) / 2)), p);
  }
  if (state.layer === "prob") return sequentialColor(probMax > 0 ? v / probMax : 0, p);
  return divergingColor(Math.max(-1, Math.min(1, v / 10)), p);
}

function renderMap() {
  const p = pal();
  const dpr = window.devicePixelRatio || 1;
  const w = GRID_COLS * CELL_PX, h = GRID_ROWS * CELL_PX;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  hatchPattern = makeHatch(p);

  let probMax = 0;
  if (state.layer === "prob") {
    for (let i = 0; i < mapValues.length; i++) {
      if (mapValues[i] > probMax) probMax = mapValues[i];
    }
    probMax = Math.max(probMax, 0.02);
  }

  const cells = META.cells.row_col;
  for (let i = 0; i < cells.length; i++) {
    const [row, col] = cells[i];
    const color = cellColor(i, p, probMax);
    ctx.fillStyle = color === null ? hatchPattern : color;
    ctx.fillRect(col * CELL_PX, row * CELL_PX, CELL_PX, CELL_PX);
  }
  if (state.selected >= 0) {
    const [row, col] = cells[state.selected];
    ctx.strokeStyle = p.ink;
    ctx.lineWidth = 2;
    ctx.strokeRect(col * CELL_PX - 1, row * CELL_PX - 1, CELL_PX + 2, CELL_PX + 2);
  }
  renderLegend(probMax);
  renderMapNote();
}

function renderLegend(probMax) {
  const p = pal();
  const el = document.getElementById("map-legend");
  el.replaceChildren();
  const bar = document.createElement("div");
  bar.className = "bar";
  const ticks = document.createElement("div");
  ticks.className = "ticks";
  const stops = [];
  let tickLabels;
  if (state.layer === "ratio") {
    for (let i = 0; i <= 10; i++) stops.push(divergingColor(i / 5 - 1, p));
    tickLabels = ["≤0.25× less frequent", "0.5×", "1× same", "2×", "≥4× more frequent"];
  } else if (state.layer === "prob") {
    for (let i = 0; i <= 10; i++) stops.push(sequentialColor(i / 10, p));
    tickLabels = ["0%", "", fmtShare(probMax / 2), "", fmtShare(probMax)];
  } else {
    for (let i = 0; i <= 10; i++) stops.push(divergingColor(i / 5 - 1, p));
    tickLabels = ["≤−10 °C below median", "", "0", "", "≥+10 °C above median"];
  }
  bar.style.background = `linear-gradient(90deg, ${stops.join(",")})`;
  el.appendChild(bar);
  for (const t of tickLabels) {
    const s = document.createElement("span");
    s.textContent = t;
    ticks.appendChild(s);
  }
  el.appendChild(ticks);
  if (state.layer === "ratio") {
    const extra = document.createElement("div");
    extra.className = "extra";
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = "repeating-linear-gradient(45deg, transparent 0 3px, " + p.notObserved + " 3px 4px)";
    extra.appendChild(sw);
    extra.appendChild(document.createTextNode(
      "not observed in the 1971–2000 baseline (fewer than ~1 in 930 days) — no ratio"));
    el.appendChild(extra);
  }
}

function renderMapHeading() {
  const t = document.getElementById("map-title");
  const s = document.getElementById("map-subtitle");
  const varName = { tday: "daily mean", tmax: "daily max", tmin: "daily min" }[state.variable];
  const q = `${varName} ${dirWord()} ${fmtTemp(state.threshold)}`;
  if (state.layer === "ratio") {
    t.textContent = `Observed frequency ratio: ${windowLabel(state.refYear)} vs 1971–2000`;
  } else if (state.layer === "prob") {
    t.textContent = `Observed frequency in ${windowLabel(state.refYear)}`;
  } else {
    t.textContent = "Threshold minus baseline median (orientation layer)";
  }
  const snapped = dateFromDoy(META.weeks.doys_0based[currentWeek()]);
  s.textContent = `${q}, within ±15 days of ${dateFromDoy(doyFromDate(state.monthIdx, state.day))}` +
    ` (window centred on ${snapped}). Click a cell for detail.`;
}

function renderMapNote() {
  const note = document.getElementById("map-note");
  if (state.layer === "ratio" && state.refYear === BASELINE_YEAR) {
    note.hidden = false;
    note.textContent = "The chosen reference period is the baseline itself (1971–2000), " +
      "so the ratio is 1 everywhere. Choose a different reference year to compare.";
  } else {
    note.hidden = true;
  }
}

/* map hover + click */
function cellAtEvent(ev) {
  const rect = canvas.getBoundingClientRect();
  const col = Math.floor((ev.clientX - rect.left) / CELL_PX);
  const row = Math.floor((ev.clientY - rect.top) / CELL_PX);
  if (col < 0 || col >= GRID_COLS || row < 0 || row >= GRID_ROWS) return -1;
  return cellIndexByRowCol[row * GRID_COLS + col];
}

canvas.addEventListener("pointermove", (ev) => {
  const i = cellAtEvent(ev);
  if (i < 0) { mapTooltip.hidden = true; return; }
  const rect = canvas.parentElement.getBoundingClientRect();
  mapTooltip.hidden = false;
  mapTooltip.style.left = `${Math.min(ev.clientX - rect.left + 14, rect.width - 200)}px`;
  mapTooltip.style.top = `${ev.clientY - rect.top + 14}px`;
  mapTooltip.replaceChildren(...tooltipContent(i));
});
canvas.addEventListener("pointerleave", () => { mapTooltip.hidden = true; });
canvas.addEventListener("click", (ev) => {
  const i = cellAtEvent(ev);
  if (i >= 0) { state.selected = i; renderMap(); renderDetail(); }
});

function tooltipContent(i) {
  const nodes = [];
  const loc = document.createElement("div");
  loc.textContent = `${META.cells.lat_wgs84[i].toFixed(2)}°N, ${META.cells.lon_wgs84[i].toFixed(2)}°E`;
  nodes.push(loc);
  const v = document.createElement("div");
  v.className = "tt-value";
  if (state.layer === "clim") {
    v.textContent = `${mapValues[i] >= 0 ? "+" : ""}${mapValues[i].toFixed(1)} °C vs baseline median`;
    nodes.push(v);
    return nodes;
  }
  const pc = mapShares.pChosen[i], pb = mapShares.pBase[i];
  if (state.layer === "ratio") {
    if (isNaN(mapValues[i])) {
      v.textContent = "not observed in baseline";
    } else if (mapValues[i] === 0) {
      v.textContent = "0× (not observed in chosen window)";
    } else {
      v.textContent = `${fmtRatio(mapValues[i])}× vs baseline`;
    }
  } else {
    v.textContent = `${fmtShare(pc)} of days`;
  }
  nodes.push(v);
  for (const [label, val] of [[windowLabel(state.refYear), pc], ["1971–2000", pb]]) {
    const row = document.createElement("div");
    row.textContent = `${label}: ${fmtShare(val)} of days`;
    nodes.push(row);
  }
  return nodes;
}

/* ------------------------------ detail view ------------------------------ */

function renderDetail() {
  const body = document.getElementById("detail-body");
  const title = document.getElementById("detail-title");
  const sub = document.getElementById("detail-subtitle");
  if (state.selected < 0) {
    body.hidden = true;
    title.textContent = "Pick a cell on the map";
    sub.textContent = "Click any grid cell to compare its two observed temperature distributions.";
    return;
  }
  body.hidden = false;
  const i = state.selected;
  title.textContent = `Cell at ${META.cells.lat_wgs84[i].toFixed(2)}°N, ${META.cells.lon_wgs84[i].toFixed(2)}°E`;
  const snapped = dateFromDoy(META.weeks.doys_0based[currentWeek()]);
  const varName = { tday: "daily mean", tmax: "daily max", tmin: "daily min" }[state.variable];
  sub.textContent = `${varName} temperature, ±15 days around ${snapped}`;

  const off = i * N_BINS;
  const isBaselineOnly = state.refYear === BASELINE_YEAR;
  const sc = tailShare(histChosen, off, state.threshold, state.direction);
  const sb = tailShare(histBase, off, state.threshold, state.direction);

  renderStats(sc, sb, isBaselineOnly, off);
  renderChart(off, isBaselineOnly);
  renderChartLegend(isBaselineOnly);
  renderDetailNotes(sc, sb, isBaselineOnly, off);
}

function makeStat(label, value, sub, keyColor, hero) {
  const d = document.createElement("div");
  d.className = hero ? "stat hero" : "stat";
  const l = document.createElement("div");
  l.className = "lbl";
  if (keyColor) {
    const k = document.createElement("span");
    k.className = "tt-key";
    k.style.borderTopColor = keyColor;
    l.appendChild(k);
  }
  l.appendChild(document.createTextNode(label));
  const v = document.createElement("div");
  v.className = "val";
  v.textContent = value;
  d.append(l, v);
  if (sub) {
    const s = document.createElement("div");
    s.className = "sub";
    s.textContent = sub;
    d.appendChild(s);
  }
  return d;
}

function renderStats(sc, sb, isBaselineOnly, off) {
  const p = pal();
  const row = document.getElementById("stats-row");
  row.replaceChildren();
  const q = `${dirWord()} ${fmtTemp(state.threshold)}`;

  row.appendChild(makeStat(
    `1971–2000: days ${q}`, fmtShare(sb.p),
    `≈ ${(sb.p * 31).toFixed(1)} of 31 days · ${sb.tail} of ${sb.total}`,
    p.seriesBase, false));

  if (isBaselineOnly) return;

  row.appendChild(makeStat(
    `${windowLabel(state.refYear)}: days ${q}`, fmtShare(sc.p),
    `≈ ${(sc.p * 31).toFixed(1)} of 31 days · ${sc.tail} of ${sc.total}`,
    p.seriesChosen, false));

  if (sb.tail > 0 && sc.tail > 0) {
    const r = sc.p / sb.p;
    row.appendChild(makeStat(
      "Observed frequency ratio", `${fmtRatio(r)}×`,
      r >= 1 ? `more frequent than in 1971–2000`
             : `less frequent than in 1971–2000`,
      null, true));
  }
}

function renderDetailNotes(sc, sb, isBaselineOnly, off) {
  const note = document.getElementById("detail-note");
  const fine = document.getElementById("detail-fineprint");
  const msgs = [];
  if (isBaselineOnly) {
    msgs.push("This is the baseline climate (1971–2000) — choose a different reference year to compare.");
  } else {
    const empty = [];
    if (sb.tail === 0) empty.push("1971–2000");
    if (sc.tail === 0) empty.push(windowLabel(state.refYear));
    if (empty.length) {
      msgs.push(`No days ${dirWord()} ${fmtTemp(state.threshold)} observed in ` +
        `${empty.join(" or ")} (fewer than ~1 in ${Math.max(sb.total, sc.total)} days) — ` +
        "the ratio is not shown rather than extrapolated.");
    }
  }
  note.hidden = msgs.length === 0;
  note.textContent = msgs.join(" ");

  const pctB = cellPercentile(histBase, off, state.threshold);
  let fineTxt = `${fmtTemp(state.threshold)} sits at the ${ordinal(Math.round(pctB))} percentile of the baseline window`;
  if (!isBaselineOnly) {
    const pctC = cellPercentile(histChosen, off, state.threshold);
    fineTxt += ` and the ${ordinal(Math.round(pctC))} of ${windowLabel(state.refYear)}`;
  }
  fine.textContent = fineTxt + ". Observed shares of pooled daily values; no uncertainty bands " +
    "are shown because consecutive days are strongly autocorrelated (see methods).";
}

/* ------------------------------ detail chart ------------------------------ */

const CHART_H = 320;
const MARGIN = { top: 14, right: 14, bottom: 30, left: 44 };

function renderChartLegend(isBaselineOnly) {
  const p = pal();
  const el = document.getElementById("chart-legend");
  el.replaceChildren();
  const entries = [["1971–2000 (baseline)", p.seriesBase]];
  if (!isBaselineOnly) entries.push([windowLabel(state.refYear), p.seriesChosen]);
  for (const [label, color] of entries) {
    const item = document.createElement("span");
    item.className = "item";
    const k = document.createElement("span");
    k.className = "tt-key";
    k.style.borderTopColor = color;
    item.append(k, document.createTextNode(label));
    el.appendChild(item);
  }
}

function svgEl(name, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

let chartGeom = null;

function renderChart(off, isBaselineOnly) {
  const p = pal();
  const svg = document.getElementById("detail-chart");
  svg.replaceChildren();
  const W = svg.clientWidth || 640;
  svg.setAttribute("viewBox", `0 0 ${W} ${CHART_H}`);

  const base = [], chosen = [];
  let totalB = 0, totalC = 0;
  for (let b = 0; b < N_BINS; b++) {
    base.push(histBase[off + b]); totalB += histBase[off + b];
    chosen.push(histChosen[off + b]); totalC += histChosen[off + b];
  }
  const fb = base.map((c) => c / totalB);
  const fc = chosen.map((c) => c / totalC);

  let lo = N_BINS - 1, hi = 0;
  for (let b = 0; b < N_BINS; b++) {
    if (base[b] > 0 || (!isBaselineOnly && chosen[b] > 0)) { lo = Math.min(lo, b); hi = Math.max(hi, b); }
  }
  lo = Math.max(0, lo - 3); hi = Math.min(N_BINS - 1, hi + 3);
  const eIdx = edgeIndex(state.threshold);

  const x0 = MARGIN.left, x1 = W - MARGIN.right, y0 = CHART_H - MARGIN.bottom, y1 = MARGIN.top;
  const binX = (b) => x0 + ((b - lo) / (hi - lo + 1)) * (x1 - x0);
  const tempX = (t) => binX((t - BIN_START) / BIN_WIDTH);
  let yMax = 0;
  for (let b = lo; b <= hi; b++) yMax = Math.max(yMax, fb[b], isBaselineOnly ? 0 : fc[b]);
  yMax *= 1.08;
  const fy = (f) => y0 - (f / yMax) * (y0 - y1);

  /* gridlines + axes (hairline, recessive) */
  const yTicks = 3;
  for (let i = 1; i <= yTicks; i++) {
    const f = (yMax * i) / (yTicks + 0.3);
    svg.appendChild(svgEl("line", { x1: x0, x2: x1, y1: fy(f), y2: fy(f), stroke: p.grid, "stroke-width": 1 }));
    const lbl = svgEl("text", { x: x0 - 6, y: fy(f) + 4, "text-anchor": "end", fill: p.muted, "font-size": 11 });
    lbl.textContent = `${(f * 100).toFixed(1)}%`;
    svg.appendChild(lbl);
  }
  svg.appendChild(svgEl("line", { x1: x0, x2: x1, y1: y0, y2: y0, stroke: p.muted, "stroke-width": 1 }));
  const tempLo = BIN_START + lo * BIN_WIDTH, tempHi = BIN_START + (hi + 1) * BIN_WIDTH;
  const step = tempHi - tempLo > 30 ? 10 : 5;
  for (let t = Math.ceil(tempLo / step) * step; t <= tempHi; t += step) {
    const x = tempX(t);
    svg.appendChild(svgEl("line", { x1: x, x2: x, y1: y0, y2: y0 + 4, stroke: p.muted, "stroke-width": 1 }));
    const lbl = svgEl("text", { x, y: y0 + 17, "text-anchor": "middle", fill: p.muted, "font-size": 11 });
    lbl.textContent = `${t}°`;
    svg.appendChild(lbl);
  }
  const xTitle = svgEl("text", { x: (x0 + x1) / 2, y: CHART_H - 4, "text-anchor": "middle", fill: p.muted, "font-size": 11 });
  xTitle.textContent = "daily temperature (°C) — share of days per 0.5 °C bin";
  svg.appendChild(xTitle);

  /* series: tail fill (direction side), soft area wash, 2px line */
  const seriesList = isBaselineOnly
    ? [[fb, p.seriesBase]]
    : [[fb, p.seriesBase], [fc, p.seriesChosen]];
  for (const [f, color] of seriesList) {
    const pts = [];
    for (let b = lo; b <= hi; b++) pts.push([binX(b) + (binX(b + 1) - binX(b)) / 2, fy(f[b])]);
    const line = pts.map((pt, k) => `${k ? "L" : "M"}${pt[0].toFixed(1)},${pt[1].toFixed(1)}`).join("");

    const area = `${line}L${pts[pts.length - 1][0].toFixed(1)},${y0}L${pts[0][0].toFixed(1)},${y0}Z`;
    svg.appendChild(svgEl("path", { d: area, fill: color, opacity: 0.10 }));

    const inTail = (b) => state.direction === "atleast" ? b >= eIdx : b <= eIdx;
    const tailPts = [];
    for (let b = lo; b <= hi; b++) {
      if (inTail(b)) tailPts.push([binX(b) + (binX(b + 1) - binX(b)) / 2, fy(f[b])]);
    }
    if (tailPts.length) {
      const anchorX = state.direction === "atleast"
        ? Math.max(tempX(state.threshold), binX(lo)) : tailPts[0][0];
      const endX = state.direction === "atleast"
        ? tailPts[tailPts.length - 1][0] : Math.min(tempX(state.threshold + BIN_WIDTH), binX(hi + 1));
      const tailLine = tailPts.map((pt, k) => `${k ? "L" : "M"}${pt[0].toFixed(1)},${pt[1].toFixed(1)}`).join("");
      const tailArea = `${tailLine}L${endX.toFixed(1)},${y0}L${anchorX.toFixed(1)},${y0}Z`;
      svg.appendChild(svgEl("path", { d: tailArea, fill: color, opacity: 0.30 }));
    }
    svg.appendChild(svgEl("path", {
      d: line, fill: "none", stroke: color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));
  }

  /* threshold line */
  const tx = tempX(state.threshold);
  if (tx >= x0 && tx <= x1) {
    svg.appendChild(svgEl("line", { x1: tx, x2: tx, y1: y1, y2: y0, stroke: p.ink, "stroke-width": 1.5, opacity: 0.55 }));
    const lbl = svgEl("text", {
      x: tx + (tx > (x0 + x1) / 2 ? -5 : 5), y: y1 + 10,
      "text-anchor": tx > (x0 + x1) / 2 ? "end" : "start", fill: p.ink, "font-size": 11.5, "font-weight": 600,
    });
    lbl.textContent = fmtTemp(state.threshold);
    svg.appendChild(lbl);
  }

  chartGeom = { lo, hi, x0, x1, y0, y1, binX, fb, fc, isBaselineOnly, W };
  attachChartHover(svg);
}

function attachChartHover(svg) {
  const tt = document.getElementById("chart-tooltip");
  const p = pal();
  const cross = svgEl("line", { y1: chartGeom.y1, y2: chartGeom.y0, stroke: p.muted, "stroke-width": 1, visibility: "hidden" });
  svg.appendChild(cross);
  svg.onpointermove = (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * chartGeom.W;
    const { lo, hi, x0, x1, binX } = chartGeom;
    if (px < x0 || px > x1) { tt.hidden = true; cross.setAttribute("visibility", "hidden"); return; }
    const b = Math.max(lo, Math.min(hi, Math.round(lo + ((px - x0) / (x1 - x0)) * (hi - lo + 1) - 0.5)));
    const cx = binX(b) + (binX(b + 1) - binX(b)) / 2;
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx);
    cross.setAttribute("visibility", "visible");
    tt.hidden = false;
    const wrapRect = svg.parentElement.getBoundingClientRect();
    tt.style.left = `${Math.min(ev.clientX - wrapRect.left + 14, wrapRect.width - 210)}px`;
    tt.style.top = `${ev.clientY - wrapRect.top - 10}px`;
    const t0 = BIN_START + b * BIN_WIDTH;
    const head = document.createElement("div");
    head.className = "tt-value";
    head.textContent = `${fmtTemp(t0)} to ${fmtTemp(t0 + BIN_WIDTH)}`;
    const rows = [head];
    const entries = [[chartGeom.fb[b], "1971–2000", p.seriesBase]];
    if (!chartGeom.isBaselineOnly) entries.push([chartGeom.fc[b], windowLabel(state.refYear), p.seriesChosen]);
    for (const [f, label, color] of entries) {
      const row = document.createElement("div");
      row.className = "tt-row";
      const k = document.createElement("span");
      k.className = "tt-key";
      k.style.borderTopColor = color;
      const strong = document.createElement("strong");
      strong.textContent = fmtShare(f);
      row.append(k, strong, document.createTextNode(` ${label}`));
      rows.push(row);
    }
    tt.replaceChildren(...rows);
  };
  svg.onpointerleave = () => { tt.hidden = true; cross.setAttribute("visibility", "hidden"); };
}

/* ------------------------------ controls ------------------------------ */

function autoDirection() {
  // default: "at least" if X is above the baseline median (cell if selected,
  // else pooled across Finland), "at most" otherwise
  const off = state.selected >= 0 ? state.selected * N_BINS : -1;
  const median = off >= 0 ? cellMedian(histBase, off) : pooledMedian(histBase);
  return state.threshold >= median ? "atleast" : "atmost";
}

function syncDirectionUI() {
  for (const btn of document.querySelectorAll("#ctl-direction button")) {
    btn.setAttribute("aria-pressed", String(btn.dataset.dir === state.direction));
  }
  document.getElementById("dir-auto-note").classList.toggle("off", state.dirManual);
}

async function update(fetchNeeded = true) {
  document.getElementById("year-label").textContent =
    `${windowLabel(state.refYear)}${state.refYear === BASELINE_YEAR ? " (baseline)" : ""}`;
  if (fetchNeeded) {
    const week = currentWeek();
    [histChosen, histBase] = await Promise.all([
      fetchHist(state.variable, state.refYear, week),
      fetchHist(state.variable, BASELINE_YEAR, week),
    ]);
  }
  if (!state.dirManual) state.direction = autoDirection();
  syncDirectionUI();
  computeMapValues();
  renderMapHeading();
  renderMap();
  renderDetail();
}

function wireControls() {
  const monthSel = document.getElementById("ctl-month");
  const daySel = document.getElementById("ctl-day");
  MONTHS.forEach((m, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = m;
    monthSel.appendChild(o);
  });
  const fillDays = () => {
    const n = MONTH_DAYS[state.monthIdx];
    daySel.replaceChildren();
    for (let d = 1; d <= n; d++) {
      const o = document.createElement("option");
      o.value = d; o.textContent = d;
      daySel.appendChild(o);
    }
    daySel.value = String(Math.min(state.day, n));
    state.day = Number(daySel.value);
  };
  monthSel.value = String(state.monthIdx);
  fillDays();
  daySel.value = String(state.day);

  monthSel.onchange = () => { state.monthIdx = Number(monthSel.value); fillDays(); update(); };
  daySel.onchange = () => { state.day = Number(daySel.value); update(); };

  document.getElementById("ctl-variable").value = state.variable;
  document.getElementById("ctl-variable").onchange = (e) => { state.variable = e.target.value; update(); };

  const thr = document.getElementById("ctl-threshold");
  thr.value = state.threshold;
  thr.onchange = () => {
    let v = Math.round(Number(thr.value) * 2) / 2;
    if (isNaN(v)) v = state.threshold;
    v = Math.max(-45, Math.min(39.5, v));
    thr.value = v;
    state.threshold = v;
    update(false);
  };

  for (const btn of document.querySelectorAll("#ctl-direction button")) {
    btn.onclick = () => {
      state.direction = btn.dataset.dir;
      state.dirManual = true;
      update(false);
    };
  }
  document.getElementById("dir-auto-note").onclick = () => {
    state.dirManual = false;
    update(false);
  };

  const yr = document.getElementById("ctl-refyear");
  yr.value = state.refYear;
  yr.oninput = () => { state.refYear = Number(yr.value); update(); };

  document.getElementById("ctl-layer").onchange = (e) => { state.layer = e.target.value; update(false); };

  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => update(false));
}

async function init() {
  const resp = await fetch(`${DATA_BASE}/meta.json`);
  if (!resp.ok) {
    document.getElementById("map-subtitle").textContent =
      "Data not found — run scripts/export_web_assets.py and serve the project root.";
    return;
  }
  META = await resp.json();
  cellIndexByRowCol = new Int32Array(GRID_ROWS * GRID_COLS).fill(-1);
  META.cells.row_col.forEach(([row, col], i) => { cellIndexByRowCol[row * GRID_COLS + col] = i; });
  wireControls();
  await update();
}

init().catch((err) => {
  console.error(err);
  document.getElementById("map-subtitle").textContent = `Failed to load data: ${err.message}`;
});
