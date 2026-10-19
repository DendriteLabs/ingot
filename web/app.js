// Polls /api/state.json and renders every section; css/ingot.css styles exactly this markup.

import { el, mount, link } from "./js/dom.js";
import { fmtDateTime, fmtDuration, fmtRelative, shortHotkey } from "./js/format.js";
import { $, minerLink, mountHeader, placeTip, svgEl } from "./js/page.js";

const POLL_MS = 5000;
const STAGES = {
  SUBMITTED: ["validation", "queued"],
  VALIDATING: ["validation", "working"],
  VALID: ["dedup", "queued"],
  DEDUPING: ["dedup", "working"],
  UNIQUE: ["eval", "queued"],
  EVALUATING: ["eval", "working"],
  WON: ["coronation", "queued"],
  CROWNING: ["coronation", "working"],
};
const FAIL_CODE = { INVALID: "invalid", DUPLICATE: "duplicate", FAILED: "crashed", UPLOAD: "upload" };

// a loser's KL against the king it faced, as a share of the king's: "4.7% above" or "0.3% below"
const vsKing = (e) => {
  const lead = 100 * (1 - Number(e.score_challenger) / Number(e.score_king));
  return `${Math.abs(lead).toFixed(1)}% ${lead < 0 ? "above" : "below"} ${e.king}`;
};
const hubUrl = (repo) => `https://huggingface.co/${repo}`;
const uidCell = (uid, hotkey) => el("td", { class: "uid" }, uid == null ? "—" : hotkey ? minerLink(state.netuid, hotkey, String(uid)) : String(uid));
const hotkeyCell = (hotkey, cls = "hotkey") => el("td", { class: cls }, hotkey ? minerLink(state.netuid, hotkey, shortHotkey(hotkey)) : "—");

let state = null;
let benchmarksLoading = false;
let benchmarkDocument = null;

async function tick() {
  try {
    const res = await fetch("/api/state.json", { cache: "no-store" });
    if (!res.ok) return;
    state = await res.json();
    render();
    // the benchmark cards draw the kings from the state: redraw them once it is here or changed
    loadBenchmarks();
  } catch (err) {
    console.warn("state fetch failed", err);
  }
}

function render() {
  $("link-github").href = state.links.github;
  $("link-miner").href = state.links.github;
  $("link-dataset").hidden = !state.links.dataset;
  if (state.links.dataset) $("link-dataset").href = state.links.dataset;
  $("foot-netuid").textContent = `Ingot · subnet ${state.netuid}`;
  renderHero();
  renderKingsChart();
  renderLineage();
  renderReign();
  renderRegistration();
  renderPipeline();
  renderHistory();
  renderFails();
}

// each eval scores its own samples, so divergences from different evals do not compare. A king's
// divergence relative to genesis is chained instead: the king it beat times the challenger / king
// ratio measured in its coronation eval. Returns the kings with `rel` (1 = genesis) and `gain` (%).
function kingGains() {
  let rel = 1;
  return state.kings.map((k) => {
    if (k.duel_challenger != null) rel *= Number(k.duel_challenger) / Number(k.duel_king);
    return { ...k, rel, gain: 100 * (1 - rel) };
  });
}

function renderHero() {
  const king = state.king;
  const url = king?.hf_repo ? hubUrl(king.hf_repo) : null;
  mount($("hero-king"), king ? (url ? link(url, king.name) : king.name) : "—");
  mount($("hero-repo"), url ? link(url, king.hf_repo) : king?.hotkey || "no king yet");
  mount($("hero-teacher"), link(hubUrl(state.teacher), state.teacher));
}

// the kings in order with their chart value (`v`, the gain over genesis) and the step each took:
// how much lower its KL was than the king it beat, in that duel
function kingSteps() {
  const crowned = kingGains().map((k) => ({ ...k, crowned: true, evalId: k.eval_id, t: new Date(k.crowned_at).getTime(), v: k.gain }));
  crowned.forEach((k, j) => Object.assign(k, { beat: crowned[j - 1], step: k.duel_king != null ? 100 * (1 - Number(k.duel_challenger) / Number(k.duel_king)) : null }));
  return crowned;
}

const numeralOf = (k) => (k.name === "GENESIS" ? "genesis" : k.name.split("-")[1]);

function renderLineage() {
  // the kings as a stack of stamps, newest on top; at the bottom genesis, the stock quant every
  // king is measured against
  const kings = kingSteps();
  const current = kings.at(-1);
  const rows = kings.slice().reverse().map((k) => {
    const url = k.hf_repo ? hubUrl(k.hf_repo) : null;
    const genesis = k.step == null;
    return el("li", { class: k === current ? "current" : genesis ? "genesis" : "" },
      el("span", { class: "numeral" }, genesis ? "" : numeralOf(k)),
      el("span", { class: "who" },
        el("span", { class: "name" }, url ? link(url, genesis ? "Genesis" : k.name, { title: k.hf_repo }) : k.name),
        el("span", { class: "meta-line" }, genesis ? "Qwen's stock FP8 quant" : `crowned ${fmtRelative(k.crowned_at)}`)),
      genesis
        ? el("span", { class: "gain baseline" }, "baseline")
        : el("span", { class: "gain", title: `${k.gain.toFixed(2)}% closer to the original ${state.teacher} than its stock FP8 quant` }, `${k.gain.toFixed(1)}%`));
  });
  mount($("lineage"), el("ol", { "aria-label": "The kings, newest first" }, ...rows));
}

function renderKingsChart() {
  const box = $("kings-chart");
  // every judged model is a dot in the order it was judged, kings brass and losers grey. The y
  // axis is how much closer to the teacher a model is than genesis. The brass staircase is the bar
  // to beat: flat while a king holds, one step up at each coronation.
  mount($("kings-meta"), "closer to the original ", link(hubUrl(state.teacher), state.teacher.split("/").pop()), " than Qwen's stock quantization");
  if (state.kings.length < 2) {
    mount(box, el("div", { class: "empty" }, "the staircase starts with the first coronation."));
    return;
  }
  const crowned = kingSteps();
  const byId = new Map(crowned.map((k) => [k.id, k]));
  const lost = state.losers.filter((e) => byId.has(e.king_id)).map((e) => ({
    ...e, crowned: false, evalId: e.id, t: new Date(e.finished_at).getTime(),
    v: 100 * (1 - byId.get(e.king_id).rel * (Number(e.score_challenger) / Number(e.score_king))),
  }));
  const points = [...crowned, ...lost].sort((a, b) => a.t - b.t).map((p, i) => Object.assign(p, { i }));
  const PAD = { top: 18, right: 24, bottom: 28, left: 48 };
  const HEIGHT = 280;
  const width = Math.max(box.clientWidth || 0, 320);
  const w = width - PAD.left - PAD.right;
  const h = HEIGHT - PAD.top - PAD.bottom;
  const last = points.length - 1;
  const current = crowned.at(-1);
  const vTop = Math.max(current.v, 1);
  const vMax = Math.max(vTop, ...points.map((p) => p.v)) * 1.08;
  // a loser can be far worse than genesis; the floor is capped so the kings keep the room
  const vMin = Math.max(Math.min(0, ...points.map((p) => p.v)), -vTop * 0.3);
  const x = (i) => PAD.left + (i / last) * w;
  const y = (v) => PAD.top + (1 - (Math.max(v, vMin) - vMin) / (vMax - vMin)) * h;
  const numeral = (p) => (p.name === "GENESIS" ? "genesis" : p.name.split("-")[1]);

  const svg = svgEl("svg", { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, role: "img" });
  svg.append(svgEl("rect", { x: PAD.left, y: PAD.top, width: w, height: h, fill: "transparent", "pointer-events": "all" }));
  for (const v of [0, current.v]) {
    svg.append(svgEl("line", { x1: PAD.left, y1: y(v), x2: width - PAD.right, y2: y(v), class: "grid" }));
    svg.append(svgEl("text", { x: PAD.left - 8, y: y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, `${v.toFixed(0)}%`));
  }
  // a numeral under each king while they fit, the current king always
  const labelled = [];
  for (const p of crowned) {
    if (labelled.length && x(p.i) - x(labelled.at(-1)) < 40) {
      if (p !== current) continue;
      labelled.pop();
    }
    labelled.push(p.i);
  }
  for (const i of labelled) {
    svg.append(svgEl("text", { x: x(i), y: HEIGHT - 8, class: "tick", "text-anchor": "middle" }, numeral(points[i])));
  }
  // square marks and steps on whole pixels: the chart's pixel look is in its shapes, not its type
  const px = (p) => [Math.round(x(p.i)), Math.round(y(p.v))];
  const mark = (p, size, cls) => {
    const [cx, cy] = px(p);
    const half = Math.floor(size / 2);
    return svgEl("rect", { x: cx - half, y: cy - half, width: size, height: size, class: cls });
  };
  const dots = points.map((p) => (p === current ? mark(p, 11, "king-dot current") : p.crowned ? mark(p, 7, "king-dot") : mark(p, 3, "king-dot lost")));
  const steps = crowned.map((p, j) => (j ? `H${px(p)[0]}V${px(p)[1]}` : `M${px(p).join(",")}`));
  // the gain over genesis, as a faint gold under the staircase down to the genesis line
  svg.append(gradient("kings-fill"));
  svg.append(svgEl("path", { d: `${steps.join("")}H${Math.round(x(last))}V${Math.round(y(0))}H${px(crowned[0])[0]}Z`, class: "area", fill: "url(#kings-fill)" }));
  for (const dot of dots.filter((_, i) => !points[i].crowned)) svg.append(dot);
  svg.append(svgEl("path", { d: `${steps.join("")}H${Math.round(x(last))}`, class: "king-line" }));
  for (const dot of dots.filter((_, i) => points[i].crowned)) svg.append(dot);

  const crosshair = svgEl("line", { y1: PAD.top, y2: HEIGHT - PAD.bottom, class: "crosshair", visibility: "hidden" });
  svg.append(crosshair);
  const tip = el("div", { class: "hero-chart-tip", hidden: true });
  let hover = -1;
  svg.addEventListener("pointermove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / Math.max(rect.width, 1)) * width;
    const i = Math.min(last, Math.max(0, Math.round(((px - PAD.left) / w) * last)));
    dots[hover]?.classList.remove("active");
    dots[i].classList.add("active");
    hover = i;
    const p = points[i];
    crosshair.setAttribute("x1", Math.round(x(i)) + 0.5);
    crosshair.setAttribute("x2", Math.round(x(i)) + 0.5);
    crosshair.removeAttribute("visibility");
    mount(tip,
      el("div", { class: "hero-chart-tip-row" },
        el("span", { class: `swatch ${p.crowned ? "k-win" : "k-loss"}` }),
        el("b", {}, `${p.v.toFixed(1)}%`),
        el("span", { class: "lbl" }, p.crowned ? p.name : `lost to ${p.king}`)),
      el("div", { class: "hero-chart-tip-meta" }, p.crowned
        ? `${p.step != null ? `KL ${p.step.toFixed(1)}% below ${p.beat.name} · ` : ""}crowned ${fmtRelative(p.crowned_at)}${p.uid != null ? ` · uid ${p.uid}` : ""}`
        : `KL ${vsKing(p)} · ${fmtRelative(p.finished_at)} · uid ${p.uid}`),
      p.evalId != null ? el("div", { class: "hero-chart-tip-meta" }, `click to open eval ${p.evalId}`) : null);
    svg.style.cursor = p.evalId != null ? "pointer" : "";
    placeTip(tip, svg, width, x(i), PAD.top);
  });
  svg.addEventListener("click", (ev) => {
    const id = points[hover]?.evalId;
    if (id == null) return;
    const url = `eval.html?id=${id}`;
    if (ev.ctrlKey || ev.metaKey) window.open(url, "_blank", "noopener");
    else location.href = url;
  });
  svg.addEventListener("pointerleave", () => {
    dots[hover]?.classList.remove("active");
    hover = -1;
    crosshair.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
  mount(box, el("div", { class: "hero-chart-plot" }, svg, tip));
}

async function loadBenchmarks() {
  if (benchmarksLoading) return;
  benchmarksLoading = true;
  const box = $("benchmark-charts");
  try {
    const res = await fetch("/api/benchmarks.json", { cache: "no-store" });
    if (res.status === 404) {
      if (benchmarkDocument !== null) throw new Error("benchmark feed unavailable");
      $("benchmarks-meta").textContent = "awaiting results";
      mount(box, el("div", { class: "empty" }, "No external benchmark results published yet."));
      return;
    }
    if (!res.ok) throw new Error(`benchmark feed: ${res.status}`);
    const doc = await res.json();
    const { results } = doc;
    if (!Array.isArray(results)) throw new Error("benchmark feed needs a results array");
    $("benchmarks-meta").textContent = "";
    // the x axis is every king, so a new king redraws the cards even when the feed did not change
    const fingerprint = JSON.stringify([doc, (state?.kings ?? []).map((k) => k.name)]);
    if (fingerprint === benchmarkDocument) {
      return;
    }
    renderBenchmarkCharts(results, BENCHMARKS);
    benchmarkDocument = fingerprint;
  } catch (err) {
    if (benchmarkDocument !== null) {
      $("benchmarks-meta").textContent = "connection interrupted · retrying";
      return;
    }
    $("benchmarks-meta").textContent = "unavailable";
    mount(box, el("div", { class: "empty" }, "Benchmark results are unavailable. Retrying shortly."));
    console.warn("benchmark fetch failed", err);
  } finally {
    benchmarksLoading = false;
  }
}

const ROMAN = { I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1000 };
// a benchmark result's model is a king's repo name, Ingot-<numeral>-<suffix>; genesis is generation 0
function generation(model) {
  const m = /^Ingot-([IVXLCDM]+)-/.exec(model || "");
  if (m) {
    const n = [...m[1]].reduce((sum, c, i, a) => sum + (ROMAN[c] < ROMAN[a[i + 1]] ? -ROMAN[c] : ROMAN[c]), 0);
    return { numeral: m[1], ordinal: n };
  }
  if (/genesis/i.test(model || "")) return { numeral: "genesis", ordinal: 0 };
  return { numeral: model || "?", ordinal: Infinity };
}

// <defs> with a vertical gradient named `id` fading to transparent, for the fill under a chart's
// line: `tone` picks its colour (fill-gold, fill-grey in the stylesheet)
function gradient(id, tone = "fill-gold") {
  const fill = svgEl("linearGradient", { id, x1: 0, y1: 0, x2: 0, y2: 1 });
  fill.append(svgEl("stop", { offset: "0", class: `${tone} top` }), svgEl("stop", { offset: "1", class: `${tone} bottom` }));
  const defs = svgEl("defs");
  defs.append(fill);
  return defs;
}

// the benchmarks shown, in this order; the feed carries more, run on fewer kings or not settled
const BENCHMARKS = ["SWE-rebench", "SWE-bench Verified", "SWE-bench Multilingual"];

function renderBenchmarkCharts(results, names) {
  const complete = (r) => r.status === "complete" && typeof r.score === "number" && Number.isFinite(r.score);
  // the x axis is every king so far, benchmarked or not: the kings the dashboard knows plus any a
  // completed run names
  const axis = [...new Map(
    [...(state?.kings ?? []).map((k) => k.name), ...results.filter(complete).map((r) => r.model)]
      .map(generation).filter((g) => g.ordinal > 0 && Number.isFinite(g.ordinal)).map((g) => [g.ordinal, g]),
  ).values()].sort((a, b) => a.ordinal - b.ordinal);
  mount($("benchmark-charts"), ...names.map((name, card_i) => {
    // only completed runs are shown; a running or stopped one is a score over part of the tasks
    const runs = results.filter((r) => r.benchmark === name && complete(r)).map((r) => ({ ...r, ...generation(r.model) }));
    const genesis = runs.find((r) => r.ordinal === 0);
    const byOrdinal = new Map(runs.map((r) => [r.ordinal, r]));
    // a click on a king opens its Hugging Face repo, the one the dashboard knows it by
    const repoOf = (model) => (state?.kings ?? []).find((k) => k.hf_repo?.split("/").pop() === model)?.hf_repo;
    const rows = axis.map((g) => {
      const r = byOrdinal.get(g.ordinal) ?? { ...g, model: `Ingot-${g.numeral}`, status: "no completed run" };
      return { ...r, repo: repoOf(r.model) };
    });
    const scored = rows.filter(complete);
    const latest = scored.at(-1);
    const card = el("article", { class: "benchmark-card" }, el("div", { class: "benchmark-chart-head" }, el("h3", {}, name)));
    if (!scored.length && !genesis) {
      card.append(el("div", { class: "chart" }, el("div", { class: "empty" }, "No completed runs yet.")));
      return card;
    }
    const width = 400, height = 130;
    const pad = { left: 44, right: 16, top: 14, bottom: 28 };
    const base = height - pad.bottom;
    // the y axis runs to the best score, genesis included, so the chart spends its height on the range
    const hi = Math.max(...scored.map((r) => r.score), genesis ? genesis.score : 0) || 1;
    const x = (i) => pad.left + (rows.length <= 1 ? 0.5 : i / (rows.length - 1)) * (width - pad.left - pad.right);
    const y = (value) => pad.top + (1 - value / hi) * (base - pad.top);
    const pct = (v) => `${Number.isInteger(v) ? v : v.toFixed(1)}%`;
    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${name}: score by king, 0 to ${pct(hi)}` });
    const fill = `benchmark-fill-${card_i}`;
    svg.append(gradient(fill));
    // ticks: the baseline and the top; genesis gets its own unless it is the top or sits on it
    for (const value of [0, hi]) {
      svg.append(svgEl("line", { x1: pad.left, x2: width - pad.right, y1: y(value), y2: y(value), class: "grid" }),
        svgEl("text", { x: pad.left - 8, y: y(value), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, pct(value)));
    }
    if (genesis) {
      svg.append(svgEl("line", { x1: pad.left, x2: width - pad.right, y1: y(genesis.score), y2: y(genesis.score), class: "genesis-line" }),
        svgEl("text", { x: width - pad.right, y: y(genesis.score) - 4, class: "tick genesis-label", "text-anchor": "end" }, "genesis"));
      if (genesis.score < hi && y(genesis.score) - y(hi) >= 12 && base - y(genesis.score) >= 12) {
        svg.append(svgEl("text", { x: pad.left - 8, y: y(genesis.score), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, pct(genesis.score)));
      }
    }
    // the gold under the line: one area per run of kings with a score, since a gap breaks the line
    const segments = [];
    rows.forEach((r, i) => {
      if (!complete(r)) return;
      if (segments.length && segments.at(-1).at(-1) === i - 1) segments.at(-1).push(i); else segments.push([i]);
    });
    for (const seg of segments) {
      if (seg.length < 2) continue;
      const top = seg.map((i) => `${Math.round(x(i))},${Math.round(y(rows[i].score))}`).join("L");
      svg.append(svgEl("path", { d: `M${top}L${Math.round(x(seg.at(-1)))},${base}L${Math.round(x(seg[0]))},${base}Z`,
        class: "area", fill: `url(#${fill})` }));
      for (let k = 1; k < seg.length; k++) {
        svg.append(svgEl("path", { d: `M${Math.round(x(seg[k - 1]))},${Math.round(y(rows[seg[k - 1]].score))}L${Math.round(x(seg[k]))},${Math.round(y(rows[seg[k]].score))}`, class: "king-line benchmark-line" }));
      }
    }
    const every = Math.max(1, Math.ceil(rows.length / 8));
    rows.forEach((r, i) => {
      if (i % every === 0 || i === rows.length - 1) {
        svg.append(svgEl("text", { x: x(i), y: height - 12, class: "tick", "text-anchor": "middle" }, r.numeral));
      }
    });
    const tip = el("div", { class: "hero-chart-tip", hidden: true });
    const crosshair = svgEl("line", { y1: pad.top, y2: base, class: "crosshair", visibility: "hidden" });
    svg.append(crosshair);
    const points = [];
    const hide = () => {
      tip.hidden = true;
      crosshair.setAttribute("visibility", "hidden");
      points.forEach((point) => point.classList.remove("active"));
    };
    let hover = null;
    const show = (r, i) => {
      hover = r;
      points.forEach((point) => point.classList.remove("active"));
      points[i]?.classList.add("active");
      crosshair.setAttribute("x1", Math.round(x(i)) + 0.5);
      crosshair.setAttribute("x2", Math.round(x(i)) + 0.5);
      crosshair.removeAttribute("visibility");
      mount(tip, el("b", {}, r.model), el("div", { class: "hero-chart-tip-meta" }, complete(r) ? `${r.score.toFixed(2)}% · ${r.completed ?? "—"} / ${r.samples ?? "—"} tasks` : r.status),
        r.repo ? el("div", { class: "hero-chart-tip-meta" }, "click to open on Hugging Face") : null);
      svg.style.cursor = r.repo ? "pointer" : "";
      placeTip(tip, svg, width, x(i), pad.top);
    };
    svg.addEventListener("click", () => hover?.repo && window.open(hubUrl(hover.repo), "_blank", "noopener"));
    rows.forEach((r, i) => {
      if (!complete(r)) return;
      const description = `${r.model}: ${r.score.toFixed(2)}% · ${r.completed ?? "—"} / ${r.samples ?? "—"} tasks`;
      const point = svgEl("rect", { x: Math.round(x(i)) - 3, y: Math.round(y(r.score)) - 3, width: 7, height: 7, tabindex: "0", class: `king-dot benchmark-point${r === latest ? " current" : ""}`, "aria-label": description });
      point.append(svgEl("title", {}, description));
      points[i] = point;
      point.addEventListener("focus", () => show(r, i));
      point.addEventListener("keydown", (ev) => ev.key === "Enter" && r.repo && window.open(hubUrl(r.repo), "_blank", "noopener"));
      point.addEventListener("blur", hide);
      svg.append(point);
    });
    svg.addEventListener("pointermove", (ev) => {
      const rect = svg.getBoundingClientRect();
      const at = ((ev.clientX - rect.left) / rect.width) * width;
      const i = rows.length <= 1 ? 0 : Math.min(rows.length - 1, Math.max(0, Math.round((at - pad.left) / (width - pad.left - pad.right) * (rows.length - 1))));
      if (rows[i]) show(rows[i], i);
    });
    svg.addEventListener("pointerleave", hide);
    card.append(el("div", { class: "chart hero-chart-plot" }, svg, tip));
    return card;
  }));
}

function renderReign() {
  const { reign } = state;
  $("reign-meta").textContent = `${reign.slots} slots · ${Math.round(reign.share * 100)}% each · ${reign.reign_days} days`;
  const now = Date.now();
  // the emission as five squares of one share each, a king's shares lit from the left
  const shares = (lit) => el("span", { class: "rc-slots", title: `${lit} of ${reign.slots} shares` },
    ...Array.from({ length: reign.slots }, (_, i) => el("i", { class: i < lit ? "on" : "" })));
  const rows = reign.kings.map((k) => {
    const url = k.hf_repo ? hubUrl(k.hf_repo) : null;
    const slots = shares(Math.round(k.weight / reign.share));
    return el("tr", { class: k.name === state.king?.name ? "reign-current" : "" },
      el("td", { class: "rc-model" }, url ? link(url, k.name, { title: k.hf_repo }) : k.name),
      uidCell(k.uid, k.hotkey),
      hotkeyCell(k.hotkey, "rc-hotkey"),
      el("td", { class: "when", title: fmtDateTime(k.crowned_at) }, fmtRelative(k.crowned_at)),
      el("td", { class: "when", title: `earns ${Math.round(k.weight * 100)}% until ${fmtDateTime(k.ends_at)}, or until ${reign.slots} newer kings are crowned` },
        `in ${fmtDuration(new Date(k.ends_at).getTime() - now)}`),
      el("td", { class: "r rc-weight" },
        slots, el("span", {}, `${Math.round(k.weight * 100)}%`)));
  });
  // one row per slot: a slot no king holds stays as an empty row; its share burns
  const dash = () => el("span", { class: "muted-dash" }, "—");
  for (let i = rows.length; i < reign.slots; i++) {
    rows.push(el("tr", { title: "open slot: its share burns" },
      el("td", {}, dash()), el("td", {}, dash()), el("td", {}, dash()), el("td", {}, dash()), el("td", {}, dash()),
      el("td", { class: "r rc-weight" }, shares(0), dash())));
  }
  mount($("reign-wrap"), el("div", { class: "data-table-wrap" },
    el("table", { class: "data-table reign-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "model"), el("th", {}, "uid"), el("th", {}, "hotkey"), el("th", {}, "crowned"),
        el("th", {}, "incentive ends"), el("th", { class: "r" }, "weight"))),
      el("tbody", {}, rows))));
}

// the registration burn over the last 24 hours as a step line: each sample holds until the next one
function renderRegistration() {
  const box = $("registration-chart");
  const points = state.registration.map(([t, block, tao]) => ({ t: t * 1000, block, tao }));
  const latest = points.at(-1);
  $("registration-meta").textContent = latest ? `last 24h · now ${latest.tao.toFixed(4)} TAO` : "last 24h";
  if (points.length < 2) {
    mount(box, el("div", { class: "empty" }, "registration prices appear after the first samples."));
    return;
  }
  // beside weight holders the chart takes the table's height (the table, not its stretched panel)
  const table = $("reign-wrap").querySelector("table");
  const beside = matchMedia("(min-width: 1280px)").matches && table ? table.offsetHeight : 0;
  const HEIGHT = Math.max(220, beside);
  const PAD = { top: 20, right: 20, bottom: 26, left: 52 };
  const width = Math.max(box.clientWidth || 0, 320);
  const w = width - PAD.left - PAD.right;
  const h = HEIGHT - PAD.top - PAD.bottom;
  const t0 = points[0].t;
  const t1 = points.at(-1).t;
  const lo = Math.min(...points.map((p) => p.tao));
  const hi = Math.max(...points.map((p) => p.tao));
  const pad = Math.max((hi - lo) * 0.1, hi * 0.02);
  const x = (t) => PAD.left + ((t - t0) / Math.max(t1 - t0, 1)) * w;
  const y = (v) => PAD.top + (1 - (v - (lo - pad)) / (hi - lo + 2 * pad)) * h;
  const when = (t) => new Date(t).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });

  const svg = svgEl("svg", { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, role: "img" });
  svg.append(svgEl("rect", { x: PAD.left, y: PAD.top, width: w, height: h, fill: "transparent", "pointer-events": "all" }));
  for (const v of [lo, hi]) {
    svg.append(svgEl("line", { x1: PAD.left, y1: y(v) + 0.5, x2: width - PAD.right, y2: y(v) + 0.5, class: "grid" }));
    svg.append(svgEl("text", { x: PAD.left - 8, y: y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, v.toFixed(3)));
  }
  for (const [t, anchor] of [[t0, "start"], [(t0 + t1) / 2, "middle"], [t1, "end"]]) {
    svg.append(svgEl("text", { x: x(t), y: HEIGHT - 8, class: "tick", "text-anchor": anchor }, when(t)));
  }
  // a sample every 25 blocks: joined straight, a price step is steep enough to read as one
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.tao).toFixed(1)}`).join("");
  svg.append(gradient("registration-fill", "fill-grey"));
  svg.append(svgEl("path", { d: `${d}V${HEIGHT - PAD.bottom}H${x(points[0].t)}Z`, class: "area", fill: "url(#registration-fill)" }));
  svg.append(svgEl("path", { d, class: "reg-line" }));
  const square = (p, cls) => svgEl("rect", { x: Math.round(x(p.t)) - 3, y: Math.round(y(p.tao)) - 3, width: 7, height: 7, class: cls });
  svg.append(square(latest, "reg-now"));
  const active = square(latest, "reg-active");
  active.setAttribute("visibility", "hidden");
  const crosshair = svgEl("line", { y1: PAD.top, y2: HEIGHT - PAD.bottom, class: "crosshair", visibility: "hidden" });
  svg.append(crosshair, active);

  const tip = el("div", { class: "hero-chart-tip", hidden: true });
  svg.addEventListener("pointermove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const t = t0 + (((ev.clientX - rect.left) / Math.max(rect.width, 1)) * width - PAD.left) / w * (t1 - t0);
    let p = points[0];
    for (const q of points) if (q.t <= t) p = q;
    crosshair.setAttribute("x1", Math.round(x(p.t)) + 0.5);
    crosshair.setAttribute("x2", Math.round(x(p.t)) + 0.5);
    crosshair.removeAttribute("visibility");
    active.setAttribute("x", x(p.t) - 3);
    active.setAttribute("y", y(p.tao) - 3);
    active.removeAttribute("visibility");
    mount(tip,
      el("div", { class: "hero-chart-tip-row" }, el("b", {}, `${p.tao.toFixed(4)} TAO`), el("span", { class: "lbl" }, `block ${p.block.toLocaleString()}`)),
      el("div", { class: "hero-chart-tip-meta" }, fmtDateTime(p.t)));
    placeTip(tip, svg, width, x(p.t), PAD.top);
  });
  svg.addEventListener("pointerleave", () => {
    crosshair.setAttribute("visibility", "hidden");
    active.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
  mount(box, el("div", { class: "hero-chart-plot" }, svg, tip));
}

// a running eval's samples_done, written every 1/20th of its samples
function evalProgress(r) {
  if (r.eval_id == null || !r.sample_count) return null;
  if (!r.samples_done) return el("div", { class: "q-progress-label" }, "loading models");
  const pct = Math.min(100, (100 * r.samples_done) / r.sample_count);
  return el("div", { class: "q-progress", title: `${pct.toFixed(0)}%` },
    el("div", { class: "q-progress-bar" }, el("div", { class: "q-progress-fill", style: `width: ${pct}%` })),
    el("span", { class: "q-progress-label" }, `${r.samples_done} / ${r.sample_count}`));
}

function renderPipeline() {
  const rows = state.queue.rows;
  const working = rows.filter((r) => STAGES[r.state]?.[1] === "working").length;
  $("pipeline-meta").textContent = rows.length ? `${working} working · ${rows.length - working} queued` : "queue idle";
  const row = (r) => {
    const [stage, status] = STAGES[r.state] ?? [r.state.toLowerCase(), "queued"];
    return el("tr", { class: status === "working" ? "working" : "" },
      el("td", {}, el("span", { class: "q-stage-badge" }, stage)),
      el("td", {}, el("span", { class: status === "working" ? "q-status working" : "q-status" }, status), evalProgress(r)),
      uidCell(r.uid, r.hotkey),
      hotkeyCell(r.hotkey),
      el("td", { class: "r when", title: fmtDateTime(r.updated_at) }, fmtRelative(r.updated_at)));
  };
  const idle = el("tr", {},
    el("td", {}, el("span", { class: "q-stage-badge" }, "-")),
    el("td", {}, el("span", { class: "q-status" }, "idle")),
    el("td", { class: "uid" }, "-"), el("td", { class: "hotkey" }, "-"), el("td", { class: "r when" }, "-"));
  mount($("pipeline-wrap"), el("div", { class: "data-table-wrap" },
    el("table", { class: "data-table q-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "stage"), el("th", {}, "state"), el("th", {}, "uid"), el("th", {}, "hotkey"), el("th", { class: "r" }, "updated"))),
      el("tbody", {}, rows.length ? rows.map(row) : idle))));
}

function renderHistory() {
  // finished duels only: a running eval is in the queue, a crashed one is in fails
  const rows = state.history.filter((e) => e.won != null);
  $("history-meta").textContent = `${rows.length} shown`;
  if (!rows.length) {
    mount($("history-wrap"), el("div", { class: "empty" }, "no completed duels yet."));
    return;
  }
  // each cell is the challenger's lead over the king, king − challenger (lower is closer to the
  // teacher), as a share of the king's score. A win needs both at least the lead that eval was
  // judged by (`need`, absolute); the column title carries the share the next eval needs.
  // a lead below FAR is a broken or far-off model: one muted bucket, the exact share in the tooltip
  const FAR = -50;
  const marginCell = (chal, king, need) => {
    if (chal == null || king == null) return el("span", { class: "muted-dash" }, "—");
    const lead = king - chal;
    const pct = (v) => `${((100 * v) / king).toFixed(2)}%`;
    const far = (100 * lead) / king < FAR;
    return el("span", {
      class: `margin-pct ${need != null && lead >= need ? "ok" : "bad"}${far ? " far" : ""}`,
      title: `${lead < 0 ? "−" : "+"}${pct(Math.abs(lead))} · challenger ${Number(chal).toFixed(6)} / king ${Number(king).toFixed(6)}${need == null ? "" : ` · needed ≥ ${pct(need)}`}`,
    }, far ? `< −${-FAR}%` : `${lead < 0 ? "−" : "+"}${pct(Math.abs(lead))}`);
  };
  const share = (current, row, king) =>
    current != null ? current : row != null && king ? row / king : null;
  const needed = (m) => (m == null ? "" : ` ≥ ${+(100 * m).toFixed(2)}%`);
  const repos = new Map(state.kings.map((k) => [k.id, k.hf_repo]));
  const kingCell = (e) => {
    const repo = repos.get(e.king_id);
    return el("span", { class: "model-cell" }, repo ? link(hubUrl(repo), e.king, { title: repo }) : e.king);
  };
  // a whole row opens its eval; the uid, hotkey and king links inside it go to their own pages
  const open = (ev, url) => {
    if (ev.target.closest("a")) return;
    if (ev.ctrlKey || ev.metaKey || ev.button === 1) window.open(url, "_blank", "noopener");
    else location.href = url;
  };
  const body = rows.map((e) => {
    const v = e.won ? "crowned" : "lost";
    const url = `eval.html?id=${e.id}`;
    const tr = el("tr", { class: e.won && e.king_id === state.kings.at(-2)?.id ? "open crowned-now" : "open", tabindex: "0", title: "open this eval" },
      el("td", { class: "when", title: fmtDateTime(e.finished_at) }, fmtRelative(e.finished_at)),
      uidCell(e.uid, e.hotkey),
      hotkeyCell(e.hotkey),
      el("td", { class: "model vs" }, kingCell(e)),
      el("td", { class: "center" }, marginCell(e.score_challenger, e.score_king, e.margin)),
      el("td", { class: "center" }, marginCell(e.dist_challenger, e.dist_king, e.dist_margin)),
      el("td", { class: "r" }, el("span", { class: `verdict-badge ${v}` }, v)));
    tr.addEventListener("click", (ev) => open(ev, url));
    tr.addEventListener("auxclick", (ev) => ev.button === 1 && open(ev, url));
    tr.addEventListener("keydown", (ev) => ev.key === "Enter" && open(ev, url));
    return tr;
  });
  mount($("history-wrap"), el("table", { class: "data-table" },
    el("thead", {}, el("tr", {},
      el("th", {}, "when"), el("th", {}, "uid"), el("th", {}, "hotkey"), el("th", {}, "vs king"),
      el("th", { class: "center", title: "(king KL − challenger KL) / king KL; a win needs at least this" }, `KL margin${needed(share(state.margins?.kl, rows[0].margin, rows[0].score_king))}`),
      el("th", { class: "center", title: "(king distance − challenger distance) / king distance; a win needs at least this" }, `distance margin${needed(share(state.margins?.dist, rows[0].dist_margin, rows[0].dist_king))}`),
      el("th", { class: "r" }, "result"))),
    el("tbody", {}, body)));
}

function renderFails() {
  const rows = state.fails;
  $("fails-meta").textContent = `${rows.length} shown`;
  if (!rows.length) {
    mount($("fails-wrap"), el("div", { class: "empty" }, "no failures."));
    return;
  }
  const body = rows.map((f) =>
    el("tr", {},
      el("td", { class: "when", title: fmtDateTime(f.at) }, fmtRelative(f.at)),
      uidCell(f.uid, f.hotkey),
      hotkeyCell(f.hotkey),
      el("td", { class: "fail-reason-cell" },
        el("span", { class: "fail-code" }, FAIL_CODE[f.stage] ?? f.stage.toLowerCase()),
        el("span", { class: "fail-reason", title: f.reason || "" }, f.reason || "—"))));
  mount($("fails-wrap"), el("table", { class: "data-table" },
    el("thead", {}, el("tr", {}, el("th", {}, "when"), el("th", {}, "uid"), el("th", {}, "hotkey"), el("th", {}, "reason"))),
    el("tbody", {}, body)));
}

mountHeader();
new ResizeObserver(() => state && renderKingsChart()).observe($("kings-chart"));
let registrationWidth = 0;
new ResizeObserver(() => {
  // width only: the chart sets its own height, which would otherwise re-trigger this observer
  if (!state || $("registration-chart").clientWidth === registrationWidth) return;
  registrationWidth = $("registration-chart").clientWidth;
  renderRegistration();
}).observe($("registration-chart"));

tick();
loadBenchmarks();
setInterval(() => { tick(); loadBenchmarks(); }, POLL_MS);
