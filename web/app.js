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

const score = (v) => (v == null ? "—" : Number(v).toFixed(4));
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

function renderKingsChart() {
  const box = $("kings-chart");
  // every judged model is a dot in the order it was judged, kings brass and losers grey. The y
  // axis is how much closer to the teacher a model is than genesis. The brass staircase is the bar
  // to beat: flat while a king holds, one step up at each coronation.
  mount($("kings-meta"), "closer to ", link(hubUrl(state.teacher), state.teacher), " than genesis");
  if (state.kings.length < 2) {
    mount(box, el("div", { class: "empty" }, "the staircase starts with the first coronation."));
    return;
  }
  const crowned = kingGains().map((k) => ({ ...k, crowned: true, evalId: k.eval_id, t: new Date(k.crowned_at).getTime(), v: k.gain }));
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
  for (const dot of dots.filter((_, i) => !points[i].crowned)) svg.append(dot);
  const steps = crowned.map((p, j) => (j ? `H${px(p)[0]}V${px(p)[1]}` : `M${px(p).join(",")}`));
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
        ? `divergence ${score(p.score)} · crowned ${fmtRelative(p.crowned_at)}${p.uid != null ? ` · uid ${p.uid}` : ""}`
        : `divergence ${score(p.score_challenger)} vs ${score(p.score_king)} · ${fmtRelative(p.finished_at)} · uid ${p.uid}`),
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
  const box = $("benchmarks-wrap");
  try {
    const res = await fetch("/api/benchmarks.json", { cache: "no-store" });
    if (res.status === 404) {
      if (benchmarkDocument !== null) throw new Error("benchmark feed unavailable");
      mount($("benchmark-charts"));
      $("benchmarks-meta").textContent = "awaiting results";
      mount(box, el("div", { class: "empty" }, "No external benchmark results published yet."));
      return;
    }
    if (!res.ok) throw new Error(`benchmark feed: ${res.status}`);
    const doc = await res.json();
    const { results, benchmarks } = doc;
    if (!Array.isArray(results)) throw new Error("benchmark feed needs a results array");
    const latestKing = Math.max(-1, ...results
      .filter((r) => Number.isFinite(r.score) && Number.isInteger(r.reign) && r.reign >= 0)
      .map((r) => r.reign));
    const latestModel = results.find((r) => r.reign === latestKing && Number.isFinite(r.score));
    const kingLabel = latestKing > 0 ? `latest benchmarked: ${latestModel.model}` : latestKing === 0 ? "latest benchmarked: genesis" : "awaiting benchmarked king";
    $("benchmarks-meta").textContent = `${results.length} results · ${kingLabel}`;
    const fingerprint = JSON.stringify(doc);
    if (fingerprint === benchmarkDocument) {
      return;
    }
    const names = benchmarks || [...new Set(results.map((r) => r.benchmark))];
    renderBenchmarkCharts(results, names);
    renderBenchmarkRows(results);
    benchmarkDocument = fingerprint;
  } catch (err) {
    if (benchmarkDocument !== null) {
      $("benchmarks-meta").textContent = "connection interrupted · retrying";
      return;
    }
    $("benchmarks-meta").textContent = "unavailable";
    mount($("benchmark-charts"));
    mount(box, el("div", { class: "empty" }, "Benchmark results are unavailable. Retrying shortly."));
    console.warn("benchmark fetch failed", err);
  } finally {
    benchmarksLoading = false;
  }
}

function renderBenchmarkCharts(results, names) {
  const hasScore = (r) => typeof r.score === "number" && Number.isFinite(r.score);
  mount($("benchmark-charts"), ...names.map((name) => {
    const rows = results.filter((r) => r.benchmark === name)
      .sort((a, b) => (a.reign ?? Infinity) - (b.reign ?? Infinity) || a.model.localeCompare(b.model));
    const scored = rows.filter(hasScore);
    const latest = scored.at(-1);
    const genesis = rows.find((r) => r.reign === 0);
    const genesisScore = genesis && hasScore(genesis) ? `${genesis.score.toFixed(2)}%` : "—";
    const card = el("article", { class: "benchmark-card" },
      el("div", { class: "benchmark-chart-head" }, el("h3", {}, name),
        el("div", { class: "benchmark-scores" },
          el("span", { class: "benchmark-latest" }, latest ? `${latest.score.toFixed(2)}%${latest.status === "complete" ? "" : " · partial"}` : "awaiting score"),
          el("span", { class: "benchmark-genesis", title: genesis && hasScore(genesis) ? `Genesis: ${genesisScore} · ${genesis.status}` : "No genesis score reported" }, `Genesis ${genesisScore}`))));
    if (!scored.length) {
      const last = rows.at(-1);
      card.append(el("div", { class: "chart" }, el("div", { class: "empty" }, ...(last ? [
        el("span", {}, last.model),
        el("span", {}, `${last.completed ?? 0} / ${last.samples ?? "—"} tasks`),
        el("span", {}, last.status),
      ] : ["No runs published yet."]))));
      return card;
    }
    const width = 240, height = 140;
    const pad = { left: 36, right: 24, top: 14, bottom: 28 };
    const x = (i) => pad.left + (rows.length === 1 ? 0.5 : i / (rows.length - 1)) * (width - pad.left - pad.right);
    const y = (value) => pad.top + (1 - value / 100) * (height - pad.top - pad.bottom);
    const label = (r) => r.reign === 0 ? "Genesis" : r.reign != null ? `King ${r.reign}` : r.model;
    const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${name}: score by model generation, 0 to 100 percent` });
    for (const value of [0, 50, 100]) {
      svg.append(svgEl("line", { x1: pad.left, x2: width - pad.right, y1: y(value), y2: y(value), class: "grid" }),
        svgEl("text", { x: pad.left - 8, y: y(value), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, `${value}%`));
    }
    // Missing scores break the line; stopped runs must not look like completed evaluations.
    rows.forEach((r, i) => {
      if (i % Math.max(1, Math.ceil(rows.length / 5)) === 0 || i === rows.length - 1) {
        svg.append(svgEl("text", { x: x(i), y: height - 12, class: "tick", "text-anchor": "middle" }, label(r)));
      }
      if (!hasScore(r)) return;
      const previous = rows[i - 1];
      if (previous && hasScore(previous)) {
        svg.append(svgEl("path", { d: `M${Math.round(x(i - 1))},${Math.round(y(previous.score))}L${Math.round(x(i))},${Math.round(y(r.score))}`,
          class: `king-line benchmark-line${r.status === "complete" && previous.status === "complete" ? "" : " partial"}` }));
      }
    });
    const tip = el("div", { class: "hero-chart-tip", hidden: true });
    const crosshair = svgEl("line", { y1: pad.top, y2: height - pad.bottom, class: "crosshair", visibility: "hidden" });
    svg.append(crosshair);
    const points = [];
    const hide = () => {
      tip.hidden = true;
      crosshair.setAttribute("visibility", "hidden");
      points.forEach((point) => point.classList.remove("active"));
    };
    const show = (r, i) => {
      points.forEach((point) => point.classList.remove("active"));
      points[i]?.classList.add("active");
      crosshair.setAttribute("x1", Math.round(x(i)) + 0.5);
      crosshair.setAttribute("x2", Math.round(x(i)) + 0.5);
      crosshair.removeAttribute("visibility");
      mount(tip, el("b", {}, r.model), el("div", { class: "hero-chart-tip-meta" }, `${hasScore(r) ? `${r.score.toFixed(2)}%` : "awaiting score"} · ${r.status}`),
        el("div", { class: "hero-chart-tip-meta" }, `${r.completed ?? "—"} / ${r.samples ?? "—"} tasks`));
      placeTip(tip, svg, width, x(i), pad.top);
    };
    rows.forEach((r, i) => {
      if (!hasScore(r)) return;
      const description = `${r.model}: ${r.score.toFixed(2)}% · ${r.status} · ${r.completed ?? "—"} / ${r.samples ?? "—"} tasks`;
      const point = svgEl("rect", { x: Math.round(x(i)) - 3, y: Math.round(y(r.score)) - 3, width: 7, height: 7, tabindex: "0", class: `king-dot benchmark-point${r.status === "complete" ? (r === latest ? " current" : "") : " partial"}`, "aria-label": description });
      point.append(svgEl("title", {}, description));
      points[i] = point;
      point.addEventListener("focus", () => show(r, i));
      point.addEventListener("click", () => show(r, i));
      point.addEventListener("blur", hide);
      svg.append(point);
    });
    svg.addEventListener("pointermove", (ev) => {
      const rect = svg.getBoundingClientRect();
      const at = ((ev.clientX - rect.left) / rect.width) * width;
      const i = rows.length === 1 ? 0 : Math.min(rows.length - 1, Math.max(0, Math.round((at - pad.left) / (width - pad.left - pad.right) * (rows.length - 1))));
      show(rows[i], i);
    });
    svg.addEventListener("pointerleave", hide);
    card.append(el("div", { class: "chart hero-chart-plot" }, svg, tip));
    return card;
  }));
}

function renderBenchmarkRows(results) {
  const box = $("benchmarks-wrap");
  if (!results.length) {
    mount(box, el("div", { class: "empty" }, "No external benchmark results published yet."));
    return;
  }
  mount(box, el("table", { class: "data-table" },
    el("thead", {}, el("tr", {},
      ...["benchmark", "model", "score", "metric", "tasks", "errors", "run", "evaluated"].map((name) =>
        el("th", { class: name === "score" ? "r" : "" }, name)))),
    el("tbody", {}, results.map((r) => {
      return el("tr", {},
        el("td", {}, r.benchmark || "—"), el("td", {}, r.model || "—"),
        el("td", { class: "r" }, typeof r.score === "number" && Number.isFinite(r.score) ? r.score.toFixed(2) : "—"),
        el("td", {}, r.metric || "—"),
        el("td", { class: "r" }, r.completed == null ? r.samples ?? "—" : `${r.completed} / ${r.samples ?? "—"}`),
        el("td", { class: "r" }, r.errors ?? "—"),
        el("td", {}, r.status || "—"),
        el("td", { class: "when" }, r.evaluated_at ? fmtDateTime(r.evaluated_at) : "—"));
    }))));
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
  const x = (t) => Math.round(PAD.left + ((t - t0) / Math.max(t1 - t0, 1)) * w);
  const y = (v) => Math.round(PAD.top + (1 - (v - (lo - pad)) / (hi - lo + 2 * pad)) * h);
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
  const d = points.map((p, i) => (i ? `H${x(p.t)}V${y(p.tao)}` : `M${x(p.t)},${y(p.tao)}`)).join("");
  svg.append(svgEl("path", { d, class: "reg-line" }));
  const square = (p, cls) => svgEl("rect", { x: x(p.t) - 3, y: y(p.tao) - 3, width: 7, height: 7, class: cls });
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
    crosshair.setAttribute("x1", x(p.t) + 0.5);
    crosshair.setAttribute("x2", x(p.t) + 0.5);
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
  // each cell is the challenger's margin over the king, king − challenger (lower is closer to the
  // teacher), to the digits the eval page shows: KL to 5, distance to 4. A win needs both at least
  // the margin the column title carries, the newest eval's.
  const marginCell = (chal, king, need, d) => {
    if (chal == null || king == null) return el("span", { class: "muted-dash" }, "—");
    const lead = king - chal;
    const fixed = (v) => Number(v).toFixed(d);
    return el("span", {
      class: `margin-pct ${need != null && lead >= need ? "ok" : "bad"}`,
      title: `challenger ${fixed(chal)} / king ${fixed(king)}${need == null ? "" : ` · needed ≥ ${fixed(need)}`}`,
    }, `${lead < 0 ? "−" : "+"}${fixed(Math.abs(lead))}`);
  };
  const needed = (v, d) => (v == null ? "" : ` ≥ ${Number(v).toFixed(d)}`);
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
      el("td", { class: "when", title: fmtDateTime(e.started_at) }, fmtRelative(e.started_at)),
      uidCell(e.uid, e.hotkey),
      hotkeyCell(e.hotkey),
      el("td", { class: "model vs" }, kingCell(e)),
      el("td", { class: "center" }, marginCell(e.score_challenger, e.score_king, e.margin, 5)),
      el("td", { class: "center" }, marginCell(e.dist_challenger, e.dist_king, e.dist_margin, 4)),
      el("td", { class: "r" }, el("span", { class: `verdict-badge ${v}` }, v)));
    tr.addEventListener("click", (ev) => open(ev, url));
    tr.addEventListener("auxclick", (ev) => ev.button === 1 && open(ev, url));
    tr.addEventListener("keydown", (ev) => ev.key === "Enter" && open(ev, url));
    return tr;
  });
  mount($("history-wrap"), el("table", { class: "data-table" },
    el("thead", {}, el("tr", {},
      el("th", {}, "when"), el("th", {}, "uid"), el("th", {}, "hotkey"), el("th", {}, "vs king"),
      el("th", { class: "center", title: "king KL − challenger KL; a win needs at least this" }, `KL margin${needed(rows[0].margin, 5)}`),
      el("th", { class: "center", title: "king distance − challenger distance; a win needs at least this" }, `distance margin${needed(rows[0].dist_margin, 4)}`),
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
