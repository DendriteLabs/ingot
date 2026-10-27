// The model page: the open round's best model from /api/state.json and /api/benchmarks.json, its
// headline figures, and the three figures of the benchmarks section. css/ingot.css styles this markup.

import { el, mount, link } from "./js/dom.js";
import { fmtRelative } from "./js/format.js";
import { $, mountHeader, svgEl } from "./js/page.js";
import { BENCHMARKS, SIZES_GB, TEACHER_RUN, density, ingotIndex, rankKings, roundRuns, scoreOf } from "./js/benchmarks.js";

const ROUND = "q2";
// the genesis run is Bonsai 2's own file; the links go to Prism's repo, not our mirror of it
const GENESIS = { label: "Bonsai 2", run: "Ingot-genesis-27B-Q2", repo: "prism-ml/Ternary-Bonsai-2-27B-gguf" };
const hubUrl = (repo) => `https://huggingface.co/${repo}`;
const pct = (v) => `${v.toFixed(1)}%`;
const signed = (v) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(1)}`;

let state = null;
let runs = null;
// all-time Hugging Face downloads per repo of the org's Ingot models, refreshed every ten minutes
let downloads = null;

async function loadDownloads() {
  try {
    const res = await fetch("https://huggingface.co/api/models?author=dendriteholdings&search=Ingot&limit=200&expand%5B%5D=downloadsAllTime");
    if (res.ok) downloads = Object.fromEntries((await res.json()).map((m) => [m.id, m.downloadsAllTime ?? 0]));
    render();
  } catch (err) {
    console.warn("downloads fetch failed", err);
  }
}

// the teacher's common files and ours; GGUF sizes from unsloth/Qwen3.8-27B-GGUF, decimal GB
const SIZES = [
  { label: "BF16", gb: 54.66, note: "the original" },
  { label: "Q8_0", gb: 29.05, note: "8-bit" },
  { label: "Q4_K_M", gb: 16.46, note: "the usual local pick" },
  { label: "IQ2_XXS", gb: 7.27, note: "standard 2-bit" },
];
const MACBOOK_GB = 16;

async function load() {
  try {
    const [s, b] = await Promise.all([fetch("/api/state.json", { cache: "no-store" }), fetch("/api/benchmarks.json", { cache: "no-store" })]);
    if (s.ok) state = await s.json();
    if (b.ok) runs = roundRuns((await b.json()).results ?? [], ROUND);
    render();
  } catch (err) {
    console.warn("feed fetch failed", err);
  }
}

// the round's best model by the external benchmarks (a full suite only), else the reigning king
function bestKing() {
  const kings = (state?.kings ?? []).filter((k) => k.quant === ROUND && k.submission_id);
  if (runs) {
    const best = rankKings(runs, BENCHMARKS, kings, GENESIS.run).find((row) => row.done === BENCHMARKS.length);
    if (best) return { king: best.king, ahead: best.ahead, benchmarked: true };
  }
  return kings.length ? { king: kings.at(-1), ahead: null, benchmarked: false } : null;
}

function render() {
  if (!state) return;
  $("link-github").href = state.links.github;
  $("foot-netuid").textContent = `Ingot · subnet ${state.netuid}`;
  $("m-teacher-short").textContent = state.teacher.split("/").pop();
  const best = bestKing();
  if (!best) return;
  const { king, ahead, benchmarked } = best;
  const index = runs ? ingotIndex(runs, king.name) : null;
  const dens = runs ? density(runs, king.name, SIZES_GB[ROUND]) : null;

  $("m-download").href = hubUrl(king.hf_repo);
  $("m-label").textContent = benchmarked ? "Best model" : "Reigning king";
  mount($("m-name"), link(hubUrl(king.hf_repo), king.name));
  mount($("m-repo"), link(hubUrl(king.hf_repo), king.hf_repo));
  mount($("m-line"), ...(ahead != null ? [el("b", {}, `${ahead} of ${BENCHMARKS.length}`), "benchmarks at or above ", link(hubUrl(GENESIS.repo), GENESIS.label)] : []));
  $("run-cmd").textContent = $("run-cmd").textContent.replaceAll(/dendriteholdings\/[A-Za-z0-9-]+/g, king.hf_repo).replaceAll(/Ingot-[A-Z]+-27B-PTQ1_0/g, `${king.name.replace(/-Q2$/, "")}-PTQ1_0`);

  $("s-index").textContent = index ? index.index.toFixed(1) : "—";
  $("s-ahead").textContent = ahead != null ? `${ahead} of ${BENCHMARKS.length}` : "—";
  $("s-density").textContent = dens ? dens.density.toFixed(3) : "—";

  renderSizes(king);
  renderKings(king);
  if (!runs) return;
  const models = [
    { tone: "teacher", label: "Qwen BF16", name: TEACHER_RUN, repo: state.teacher, size: SIZES_GB.teacher },
    { tone: "genesis", label: GENESIS.label, name: GENESIS.run, repo: GENESIS.repo, size: SIZES_GB[ROUND] },
    { tone: "king latest", label: `King ${king.name.split("-")[1]}`, name: king.name, repo: king.hf_repo, size: SIZES_GB[ROUND] },
  ];
  renderScores(models);
  const indexed = models.map((m) => ({ ...m, v: ingotIndex(runs, m.name) })).filter((m) => m.v);
  renderBars($("index-chart"), indexed, (m) => m.v.index, (m) => m.v.index.toFixed(1), 100,
    (m) => Object.entries(m.v.categories).map(([c, v]) => `${c} ${v.toFixed(0)}`).join(" · "), indexed.find((m) => m.tone === "teacher"));
  renderBars($("density-chart"), models.map((m) => ({ ...m, v: density(runs, m.name, m.size) })).filter((m) => m.v), (m) => m.v.density, (m) => m.v.density.toFixed(3), null,
    (m) => `mean ${m.v.mean.toFixed(1)}% · ${m.size} GB`);
}

// horizontal bars of file size, ours last, with the line a 16 GB MacBook draws
function renderSizes(king) {
  const rows = [...SIZES.map((r) => ({ ...r, tone: "other" })),
    { label: king.name.replace(/-Q2$/, ""), gb: SIZES_GB[ROUND], note: "1.75 bits per weight", tone: "king" }];
  const top = rows[0].gb;
  const at = (gb) => `${(100 * gb) / top}%`;
  mount($("size-chart"),
    // an overlay on the same columns as the rows, so the line lands on the tracks' scale
    el("div", { class: "size-overlay" }, el("span"), el("span", { class: "size-line-wrap" },
      el("i", { class: "size-line", style: `left: ${at(MACBOOK_GB)}` }, el("span", {}, "16 GB MacBook"))), el("span")),
    ...rows.map((r) => el("div", { class: `size-row ${r.tone}${r.gb > MACBOOK_GB ? " over" : ""}` },
      el("span", { class: "size-label" }, r.label, el("small", {}, r.note)),
      el("span", { class: "size-track" }, el("i", { style: `width: ${at(r.gb)}` })),
      el("span", { class: "size-value" }, `${r.gb.toFixed(r.gb < 10 ? 2 : 1)} GB`))));
}

// the newest kings as cards, the subnet's counts in the head
function renderKings(best) {
  const kings = (state.kings ?? []).filter((k) => k.quant === ROUND && k.submission_id);
  const total = downloads ? Object.values(downloads).reduce((a, b) => a + b, 0) : null;
  mount($("kings-meta"), `${kings.length} crowned · ${state.stats?.evaluated ?? state.history.length} assayed`,
    total ? ` · ${total.toLocaleString("en-US")} downloads · ` : " · ", link("dashboard.html", "dashboard →", { target: null, rel: null }));
  // the four best by the site's own rule, the summed points over Bonsai 2 across a full suite (the
  // hero's pick comes first); without the feed, the newest
  const scored = runs ? rankKings(runs, BENCHMARKS, kings, GENESIS.run).filter((row) => row.done === BENCHMARKS.length) : [];
  const shown = scored.length ? scored.slice(0, 4).map((row) => row.king) : kings.slice(-4).reverse();
  mount($("king-cards"), ...shown.map((k) => {
    const index = runs ? ingotIndex(runs, k.name) : null;
    const full = index && index.coverage === BENCHMARKS.length;
    const pending = index ? "benchmarking" : "queued";
    return el("a", { class: `king-card${k === shown[0] && scored.length ? " best" : ""}`, href: hubUrl(k.hf_repo), target: "_blank", rel: "noopener" },
      el("span", { class: "kc-numeral" }, k.name.split("-")[1]),
      el("span", { class: "kc-name" }, k.name),
      el("span", { class: "kc-meta" }, `crowned ${fmtRelative(k.crowned_at)}`),
      el("span", { class: "kc-stats" },
        el("span", {}, el("b", {}, full ? index.index.toFixed(1) : "—"), full ? "Index" : pending),
        el("span", {}, el("b", {}, downloads?.[k.hf_repo] != null ? downloads[k.hf_repo].toLocaleString("en-US") : "—"), "downloads")),
      k === shown[0] && scored.length ? el("span", { class: "kc-badge" }, "best") : null);
  }));
}

// one card per benchmark, the king's biggest leads over genesis first: the teacher, genesis and
// the king as bars from a shared floor
function renderScores(models) {
  const king = models.find((m) => m.tone.startsWith("king"));
  const lead = (name) => (scoreOf(runs, king.name, name) ?? -Infinity) - (scoreOf(runs, GENESIS.run, name) ?? 0);
  const ordered = BENCHMARKS.slice().sort((a, b) => lead(b[0]) - lead(a[0]));
  mount($("benchmark-charts"), ...ordered.map(([name, kind]) => {
    const rows = models.map((m) => ({ ...m, score: scoreOf(runs, m.name, name) }));
    const scores = rows.map((r) => r.score).filter((v) => v != null);
    const hi = Math.max(...scores, 1);
    const lo = Math.min(...scores, hi);
    const floor = Math.max(0, Math.floor(lo - Math.max(2, hi - lo)));
    const x = (score) => (100 * (score - floor)) / (hi - floor);
    const base = rows.find((r) => r.tone === "genesis")?.score;
    const bar = (r) => {
      const d = r.score != null && base != null && r.tone !== "genesis" ? r.score - base : null;
      return el("a", { class: `bench-row ${r.tone}${r.score == null ? " missing" : ""}`, href: hubUrl(r.repo), target: "_blank", rel: "noopener",
        title: r.score != null ? `${r.name}: ${pct(r.score)}${d == null ? "" : ` · ${signed(d)} points from ${GENESIS.label}`}` : "not run yet" },
      el("span", { class: "bench-label" }, r.label),
      el("span", { class: "bench-track" },
        base == null ? null : el("i", { class: "bench-ref", style: `left: ${x(base)}%` }),
        r.score == null ? null : el("i", { class: "bench-bar", style: `width: ${Math.max(0.5, x(r.score))}%` })),
      el("span", { class: "bench-value" },
        d == null ? null : el("span", { class: `bench-delta${d > 0 ? " up" : ""}` }, signed(d)),
        r.score == null ? "—" : pct(r.score)));
    };
    return el("article", { class: "benchmark-card" },
      el("div", { class: "benchmark-chart-head" }, el("h3", {}, name), el("span", { class: "benchmark-kind" }, kind)),
      el("div", { class: "bench-rows" }, ...rows.map(bar)));
  }));
}

// vertical bars on a dotted grid, in the charts' pixel look: a gradient body with a solid cap, the
// value above and the name below. With `ceiling`, the teacher is the dashed line the bars are
// measured against instead of a bar of its own.
function renderBars(box, items, value, label, max, note, ceiling) {
  const HEIGHT = 260;
  const PAD = { top: 28, bottom: 30, left: 40, right: 12 };
  const width = Math.max(box.clientWidth || 0, 240);
  const bars = ceiling ? items.filter((m) => m.tone !== "teacher") : items;
  const top = max ?? Math.max(...items.map(value)) * 1.12;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const y = (v) => Math.round(PAD.top + (1 - v / top) * plotH);
  const slot = (width - PAD.left - PAD.right) / bars.length;
  const bar = Math.round(Math.min(96, slot * 0.5));
  const svg = svgEl("svg", { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, role: "img", class: "vbars-svg" });
  const defs = svgEl("defs");
  for (const [id, tone] of [["vbar-gold", "fill-gold"], ["vbar-grey", "fill-grey"]]) {
    const g = svgEl("linearGradient", { id, x1: 0, y1: 0, x2: 0, y2: 1 });
    g.append(svgEl("stop", { offset: 0, class: `${tone} top` }), svgEl("stop", { offset: 1, class: `${tone} bottom` }));
    defs.append(g);
  }
  svg.append(defs);
  // faint grid lines at round steps; the ceiling, when there is one, is the top line
  const step = max ? max / 4 : [1, 2, 5].map((m) => m * 10 ** Math.floor(Math.log10(top / 4))).reduce((best, c) => (Math.abs(c - top / 4) < Math.abs(best - top / 4) ? c : best));
  for (let v = step; v <= top + 1e-9; v += step) {
    const isCeiling = ceiling && Math.abs(v - top) < 1e-9;
    svg.append(svgEl("line", { x1: PAD.left, y1: y(v) + 0.5, x2: width - PAD.right, y2: y(v) + 0.5, class: isCeiling ? "ceiling" : "grid" }));
    svg.append(svgEl("text", { x: PAD.left - 8, y: y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, max ? String(Math.round(v)) : v.toFixed(2)));
  }
  svg.append(svgEl("line", { x1: PAD.left, y1: HEIGHT - PAD.bottom + 0.5, x2: width - PAD.right, y2: HEIGHT - PAD.bottom + 0.5, class: "axis" }));
  if (ceiling) {
    svg.append(svgEl("text", { x: width - PAD.right, y: y(top) - 8, class: "vname", "text-anchor": "end" }, `${ceiling.label} · ${label(ceiling)}`));
  }
  bars.forEach((m, i) => {
    const cx = Math.round(PAD.left + slot * (i + 0.5));
    const x0 = cx - Math.floor(bar / 2);
    const yTop = y(value(m));
    const tone = m.tone.split(" ")[0];
    const group = svgEl("a", { href: hubUrl(m.repo), target: "_blank", rel: "noopener" });
    group.append(svgEl("title", {}, `${m.name}: ${label(m)} · ${note(m)}`));
    if (tone === "teacher") {
      group.append(svgEl("rect", { x: x0, y: yTop, width: bar, height: Math.max(HEIGHT - PAD.bottom - yTop, 1), class: "vbar teacher" }));
    } else {
      group.append(svgEl("rect", { x: x0, y: yTop, width: bar, height: Math.max(HEIGHT - PAD.bottom - yTop, 1), class: `vbar ${tone}`, fill: `url(#vbar-${tone === "king" ? "gold" : "grey"})` }));
      group.append(svgEl("rect", { x: x0, y: yTop, width: bar, height: 3, class: `vcap ${tone}` }));
    }
    group.append(svgEl("text", { x: cx, y: yTop - 9, class: `vval ${tone}`, "text-anchor": "middle" }, label(m)));
    group.append(svgEl("text", { x: cx, y: HEIGHT - 10, class: "vname", "text-anchor": "middle" }, m.label));
    svg.append(group);
  });
  mount(box, svg);
}

mountHeader();
loadDownloads();
setInterval(loadDownloads, 600000);
new ResizeObserver(() => runs && render()).observe($("panel-figures"));
load();
setInterval(load, 30000);
