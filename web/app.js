// Polls /api/state.json and renders every section; css/ingot.css styles exactly this markup.

import { el, mount, link } from "./js/dom.js";
import { fmtDateTime, fmtDuration, fmtRelative, shortHotkey } from "./js/format.js";
import { $, minerLink, mountHeader, placeTip, svgEl } from "./js/page.js";
import { BENCHMARKS, TEACHER_RUN, complete, protocolOf, rankKings as rank, roundRuns } from "./js/benchmarks.js";

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
  if (e.score_challenger == null) return "diverged, no finite score";
  const lead = 100 * (1 - Number(e.score_challenger) / Number(e.score_king));
  return `${Math.abs(lead).toFixed(1)}% ${lead < 0 ? "above" : "below"} ${e.king}`;
};
const hubUrl = (repo) => `https://huggingface.co/${repo}`;
const uidCell = (uid, hotkey) => el("td", { class: "uid" }, uid == null ? "—" : hotkey ? minerLink(state.netuid, hotkey, String(uid)) : String(uid));
const hotkeyCell = (hotkey, cls = "hotkey") => el("td", { class: cls }, hotkey ? minerLink(state.netuid, hotkey, shortHotkey(hotkey)) : "—");

let state = null;
let benchmarksLoading = false;
let benchmarkDocument = null;
// the selected round's benchmark runs (its protocol's), for the hero's best model; null until loaded
let benchRuns = null;

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
  $("foot-netuid").textContent = `Ingot · subnet ${state.netuid}`;
  renderHero();
  renderKingsChart();
  renderLineage();
  renderReign();
  renderRegistration();
  // a closed round queues nothing
  $("queue").hidden = !ROUNDS[selectedRound].open;
  renderPipeline();
  renderHistory();
  renderFails();
}

const ROUNDS = {
  q2: {
    label: "1.75-bit", open: true,
    genesis: { label: "Bonsai 2 Q2", name: "Bonsai-2-27B-Q2", versus: "Bonsai-2-27B-Q2", about: "Prism's ternary quantization (1.75 bpw)",
      repo: "dendriteholdings/Ingot-genesis-27B-Q2", run: "Ingot-genesis-27B-Q2" },
  },
  fp8: {
    label: "8-bit", open: false,
    genesis: { label: "Qwen FP8", name: "Qwen FP8", versus: "Qwen's own FP8", about: "Qwen's own quantization",
      repo: "dendriteholdings/Ingot-genesis-27B-FP8", run: "Ingot-genesis-27B-FP8" },
  },
};
let selectedRound = "q2";
// a backend from before the 2-bit round sends no quant: everything it knows is 8-bit
const inRound = (row) => (row.quant ?? "fp8") === selectedRound;

// each eval scores its own samples, so divergences from different evals do not compare. A king's
// divergence relative to genesis is chained instead: the king it beat times the challenger / king
// ratio measured in its coronation eval. Returns the selected round's kings with `rel` (1 =
// genesis) and `gain` (%).
function kingGains() {
  let rel = 1;
  return state.kings.filter(inRound).map((k) => {
    if (k.duel_challenger != null) rel *= Number(k.duel_challenger) / Number(k.duel_king);
    return { ...k, rel, gain: 100 * (1 - rel) };
  });
}

// the dashboard's head: the round tabs, and the round's best model (by the external benchmarks once
// their feed is in, a full suite only; until then the reigning king)
function renderHero() {
  document.querySelectorAll(".stamp-tab").forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.tab === selectedRound));
    tab.onclick = () => { selectedRound = tab.dataset.tab; benchRuns = null; render(); loadBenchmarks(); };
  });
  const { open, genesis } = ROUNDS[selectedRound];
  const best = benchRuns ? rankKings(benchRuns, BENCHMARKS).find((row) => row.done === BENCHMARKS.length) : null;
  const king = best?.king ?? kingGains().filter((k) => k.submission_id).at(-1);
  const teacher = state.teacher.split("/").pop();
  const [name, repo] = king ? [king.name, king.hf_repo] : [genesis.name, genesis.repo];
  const url = repo ? hubUrl(repo) : null;
  $("hero-label").textContent = `${open ? "Open round" : "Closed round"}${best ? " · best model" : " · reigning king"}`;
  mount($("hero-king"), url ? link(url, name) : name);
  mount($("hero-line"), ...(best
    ? [el("b", {}, `${best.ahead} of ${BENCHMARKS.length}`), "benchmarks at or above ", link(hubUrl(genesis.repo), genesis.versus)]
    : king
      ? [el("b", {}, `${king.gain.toFixed(1)}%`), "closer to ", link(hubUrl(state.teacher), teacher), " than ", link(hubUrl(genesis.repo), genesis.versus)]
      : []));
}

const rankKings = (runs, benchmarks) => rank(runs, benchmarks, state ? kingGains().filter((k) => k.submission_id) : [], ROUNDS[selectedRound].genesis.run);

// the kings in order with their chart value (`v`, the gain over genesis) and the step each took:
// how much lower its KL was than the king it beat, in that duel
function kingSteps() {
  const crowned = kingGains().map((k) => ({ ...k, crowned: true, evalId: k.eval_id, t: new Date(k.crowned_at).getTime(), v: k.gain }));
  crowned.forEach((k, j) => Object.assign(k, { beat: crowned[j - 1], step: k.duel_king != null ? 100 * (1 - Number(k.duel_challenger) / Number(k.duel_king)) : null }));
  return crowned;
}

const numeralOf = (k) => (k.name === "GENESIS" ? "genesis" : k.name.split("-")[1]);

function renderLineage() {
  const { genesis } = ROUNDS[selectedRound];
  const kings = kingSteps().filter((k) => k.submission_id);
  const current = kings.at(-1);
  const rows = kings.slice().reverse().map((k) => {
    const url = k.hf_repo ? hubUrl(k.hf_repo) : null;
    return el("li", { class: k === current ? "current" : "" },
      el("span", { class: "numeral" }, numeralOf(k)),
      el("span", { class: "who" },
        el("span", { class: "name" }, url ? link(url, k.name, { title: k.hf_repo }) : k.name),
        el("span", { class: "meta-line" }, `crowned ${fmtRelative(k.crowned_at)}`)),
      el("span", { class: "gain", title: `${k.gain.toFixed(2)}% closer to the original ${state.teacher} than ${genesis.versus}` }, `${k.gain.toFixed(1)}%`));
  });
  rows.push(el("li", { class: current ? "genesis" : "current" },
    el("span", { class: "numeral" }),
    el("span", { class: "who" },
      el("span", { class: "name" }, link(hubUrl(genesis.repo), genesis.name, { title: genesis.repo })),
      el("span", { class: "meta-line" }, `genesis · ${genesis.about}`)),
    el("span", { class: "gain baseline" }, "baseline")));
  mount($("lineage"), el("ol", { "aria-label": "The kings, newest first" }, ...rows));
}

function renderKingsChart() {
  const box = $("kings-chart");
  // every judged model is a dot in the order it was judged, kings brass and losers grey. The y
  // axis is how much closer to the teacher a model is than genesis. The brass staircase is the bar
  // to beat: flat while a king holds, one step up at each coronation.
  const { genesis } = ROUNDS[selectedRound];
  mount($("kings-meta"), "closer to the original ", link(hubUrl(state.teacher), state.teacher.split("/").pop()), " than ", link(hubUrl(genesis.repo), genesis.versus));
  const crowned = kingSteps();
  if (crowned.length < 2) {
    mount(box, el("div", { class: "empty" }, "the staircase starts with the first coronation."));
    return;
  }
  const byId = new Map(crowned.map((k) => [k.id, k]));
  // a diverged challenger's score is published as null: it sits on the floor with the far-off losers
  const lost = state.losers.filter((e) => byId.has(e.king_id)).map((e) => ({
    ...e, crowned: false, evalId: e.id, t: new Date(e.finished_at).getTime(),
    v: e.score_challenger == null ? -Infinity : 100 * (1 - byId.get(e.king_id).rel * (Number(e.score_challenger) / Number(e.score_king))),
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
        el("b", {}, Number.isFinite(p.v) ? `${p.v.toFixed(1)}%` : "diverged"),
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
  // the cards name the teacher and the kings from the state: nothing to draw before it is here
  if (benchmarksLoading || !state) return;
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
    const { genesis } = ROUNDS[selectedRound];
    mount($("benchmarks-meta"), el("b", {}, link(hubUrl(state.teacher), "Qwen BF16")), ": the unquantized teacher · ",
      el("b", {}, link(hubUrl(genesis.repo), genesis.name)), `: ${genesis.about}`);
    const fingerprint = JSON.stringify([doc, selectedRound, (state?.kings ?? []).map((k) => k.name)]);
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

// <defs> with a vertical gradient named `id` fading to transparent, for the fill under a chart's
// line: `tone` picks its colour (fill-gold, fill-grey in the stylesheet)
function gradient(id, tone = "fill-gold") {
  const fill = svgEl("linearGradient", { id, x1: 0, y1: 0, x2: 0, y2: 1 });
  fill.append(svgEl("stop", { offset: "0", class: `${tone} top` }), svgEl("stop", { offset: "1", class: `${tone} bottom` }));
  const defs = svgEl("defs");
  defs.append(fill);
  return defs;
}

// the king a visitor picked, per round; unset (every page load) means the best overall
const chosenKing = {};

// the chosen king's name in the section head; it opens the ranked kings as a listbox, by mouse or keyboard
function renderKingPicker(ranked, pick, redraw) {
  const total = BENCHMARKS.length;
  const { genesis } = ROUNDS[selectedRound];
  const best = Math.max(...ranked.filter((row) => row.done === total).map((row) => row.sum), 1);
  const signed = (v) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(1)}`;
  const options = ranked.map((row, i) => el("li", {
    role: "option", id: `king-option-${i}`, "aria-selected": String(row === pick),
    class: row.done === total ? null : "running",
  },
  el("span", { class: "kp-rank" }, row.done === total ? i + 1 : "·"),
  el("span", { class: "kp-name" }, `King ${numeralOf(row.king)}`),
  el("span", { class: "kp-bar" }, row.done === total ? el("i", { style: `width: ${Math.max(0, (100 * row.sum) / best)}%` }) : null),
  el("span", { class: "kp-score" }, row.done === total ? signed(row.sum) : `${row.done}/${total}`)));
  const pop = el("div", { class: "kp-pop", hidden: true },
    el("div", { class: "kp-head" }, `sum of points vs ${genesis.label}`),
    el("ul", { class: "kp-list", role: "listbox", id: "king-list", "aria-label": `Kings by their summed points over ${genesis.label}` }, ...options));
  const button = el("button", { class: "kp-button", type: "button", "aria-haspopup": "listbox", "aria-expanded": "false", "aria-controls": "king-list",
    title: "compare another king" }, `King ${numeralOf(pick.king)}`);
  let active = ranked.indexOf(pick);
  const show = (open) => {
    pop.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
    options.forEach((li, i) => li.classList.toggle("active", open && i === active));
  };
  const choose = (i) => { chosenKing[selectedRound] = ranked[i].king.name; show(false); redraw(); $("bench-picker").querySelector("button").focus(); };
  button.onclick = () => show(pop.hidden);
  button.onkeydown = (ev) => {
    const step = { ArrowDown: 1, ArrowUp: -1 }[ev.key];
    if (step) { ev.preventDefault(); active = pop.hidden ? active : (active + step + options.length) % options.length; show(true); }
    else if (ev.key === "Enter" && !pop.hidden) { ev.preventDefault(); choose(active); }
    else if (ev.key === "Escape") show(false);
  };
  options.forEach((li, i) => { li.onclick = () => choose(i); li.onpointermove = () => { active = i; show(true); }; });
  mount($("bench-picker"), button, pop);
}
// one listener for the page: a click outside the picker closes its list
document.addEventListener("click", (ev) => {
  const box = $("bench-picker");
  if (!box || box.contains(ev.target)) return;
  box.querySelector(".kp-pop")?.setAttribute("hidden", "");
  box.querySelector(".kp-button")?.setAttribute("aria-expanded", "false");
});

function renderBenchmarkCharts(results, benchmarks) {
  const runs = roundRuns(results, selectedRound);
  const { genesis } = ROUNDS[selectedRound];
  const ranked = rankKings(runs, benchmarks);
  if (benchRuns !== runs) {
    benchRuns = runs;
    renderHero();
  }
  const pick = ranked.find((row) => row.king.name === chosenKing[selectedRound]) ?? ranked[0];
  if (pick) renderKingPicker(ranked, pick, () => renderBenchmarkCharts(results, benchmarks));
  else mount($("bench-picker"));
  const models = [
    { tone: "teacher", label: "Qwen BF16", repo: state?.teacher, is: (r) => r.model === TEACHER_RUN },
    { tone: "genesis", label: genesis.label, repo: genesis.repo, is: (r) => r.model === genesis.run },
    ...(pick ? [{ tone: "king latest", label: `King ${numeralOf(pick.king)}`, repo: pick.king.hf_repo, is: (r) => r.model === pick.king.name }] : []),
  ];
  const pct = (v) => `${v.toFixed(1)}%`;
  const signed = (v) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(1)}`;
  mount($("benchmark-charts"), ...benchmarks.map(([name, kind]) => {
    const mine = runs.filter((r) => r.benchmark === name);
    const rows = models.map((m) => ({ ...m, run: mine.find(m.is) }));
    const scored = (row) => row.run && complete(row.run);
    const scores = rows.filter(scored).map((row) => row.run.score);
    const hi = Math.max(...scores);
    const lo = Math.min(...scores);
    // never below zero: a percentage axis that starts at -3% reads as a mistake
    const floor = Math.max(0, Math.floor(lo - Math.max(2, hi - lo)));
    const x = (score) => (100 * (score - floor)) / (hi - floor);
    const genesis = rows.find((row) => row.tone === "genesis" && scored(row))?.run.score;
    const delta = (row) => (scored(row) && genesis != null && row.tone !== "genesis" ? row.run.score - genesis : null);
    const bar = (row) => {
      const { tone, label, repo, run } = row;
      const pending = run?.status === "running" ? "running" : "queued";
      const d = delta(row);
      const tasksOf = (r) => `${r.completed ?? "—"} / ${r.samples ?? "—"} tasks`;
      return el(repo ? "a" : "div", {
        class: `bench-row ${tone}${scored(row) ? "" : " missing"}`,
        href: repo ? hubUrl(repo) : null, target: repo ? "_blank" : null, rel: repo ? "noopener" : null,
        title: scored(row) ? `${run.model}: ${pct(run.score)}${d == null ? "" : ` · ${signed(d)} points from ${genesis.label}`} · ${tasksOf(run)}`
          : pending === "running" && run.completed != null ? `running · ${tasksOf(run)}` : pending,
      },
      el("span", { class: "bench-label" }, label),
      el("span", { class: "bench-track" },
        genesis == null ? null : el("i", { class: "bench-ref", style: `left: ${x(genesis)}%` }),
        scored(row) ? el("i", { class: "bench-bar", style: `width: ${Math.max(0.5, x(run.score))}%` }) : null),
      el("span", { class: "bench-value" },
        d == null ? null : el("span", { class: `bench-delta${d > 0 ? " up" : ""}` }, signed(d)),
        scored(row) ? pct(run.score) : pending));
    };
    const axis = scores.length ? el("div", { class: "bench-axis" }, el("span"), el("span", {}, el("b", {}, `${floor}%`), el("b", {}, pct(hi))), el("span")) : null;
    return el("article", { class: "benchmark-card" },
      el("div", { class: "benchmark-chart-head" }, el("h3", {}, name), el("span", { class: "benchmark-kind" }, kind)),
      el("div", { class: "bench-rows" }, ...rows.map(bar), axis));
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
  // finished duels of the round only: a running eval is in the queue, a crashed one is in fails.
  // An eval's round is its king's.
  const kings = kingGains();
  const ids = new Set(kings.map((k) => k.id));
  const rows = state.history.filter((e) => e.won != null && ids.has(e.king_id));
  $("history-meta").textContent = `${ROUNDS[selectedRound].label} · ${rows.length} shown`;
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
    if (king == null) return el("span", { class: "muted-dash" }, "—");
    if (chal == null) return el("span", { class: "margin-pct bad far", title: "challenger diverged · no finite score" }, `< −${-FAR}%`);
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
    const tr = el("tr", { class: e.won && e.king_id === kings.at(-2)?.id ? "open crowned-now" : "open", tabindex: "0", title: "open this eval" },
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
  const rows = state.fails.filter(inRound);
  $("fails-meta").textContent = `${ROUNDS[selectedRound].label} · ${rows.length} shown`;
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
