// One eval's page: the duel with the king on both win measures, where in the network each model
// drifts from the teacher, every sample, and the raw artifacts. Reads api/evals/<id>.json.

import { el, mount } from "./js/dom.js";
import { fmtDateTime, fmtDuration, shortHotkey } from "./js/format.js";
import { $, minerLink, mountHeader, placeTip, svgEl } from "./js/page.js";
import { zip } from "./js/zip.js";

const id = new URLSearchParams(location.search).get("id");
const num = (v, d = 5) => (v == null ? "—" : Number(v).toFixed(d));
const signed = (v, d) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
const pct = (v) => (v == null ? "—" : `${(100 * v).toFixed(2)}%`);
// the three roles a sample record can hold, as the chart draws them
const SERIES = [["challenger", "challenger"], ["king", "king"], ["teacher_rerun", "teacher reread"]];
const WHAT = {
  "summary.json": "the eval's row: both scores, the margins, the noise floor, seed and timings",
  "samples.jsonl": "one record per sample: KL, top-1 agreement, distance and 1 − cos by layer for each model",
};

mountHeader();
load();

async function load() {
  const res = /^\d+$/.test(id ?? "") ? await fetch(`api/evals/${id}.json`, { cache: "no-store" }) : null;
  if (!res?.ok) {
    $("ev-title").textContent = "No such eval";
    $("ev-sub").textContent = id ? `nothing is recorded for eval ${id}.` : "open an eval from the history table.";
    for (const box of ["duel-wrap", "samples-wrap"]) mount($(box), el("div", { class: "empty" }, "—"));
    return;
  }
  const doc = await res.json();
  document.title = `INGOT · eval ${id}`;
  $("link-github").href = doc.links.github;
  $("foot-netuid").textContent = `Ingot · subnet ${doc.netuid}`;
  renderHead(doc);
  renderDuel(doc);
  renderSamples(doc);
  renderArtifacts(doc);
  let width = 0;
  new ResizeObserver(() => {
    // width only: the chart sets its own height
    if ($("layers-chart").clientWidth === width) return;
    width = $("layers-chart").clientWidth;
    renderLayers(doc);
  }).observe($("layers-chart"));
}

function renderHead(doc) {
  const e = doc.eval;
  $("ev-eyebrow").textContent = `eval ${e.id}${e.finished_at ? ` · finished ${fmtDateTime(e.finished_at)} UTC` : ""}`;
  $("ev-title").textContent = e.error ? "Crashed" : e.won ? `Crowned ${doc.crowned_as ?? "king"}` : e.won === false ? `Lost to ${doc.king}` : "Running";
  $("ev-title").classList.toggle("won", Boolean(e.won));
  mount($("ev-sub"),
    "uid ", doc.uid == null ? "—" : minerLink(doc.netuid, doc.hotkey, String(doc.uid)),
    " · ", minerLink(doc.netuid, doc.hotkey, shortHotkey(doc.hotkey)),
    ` · against ${doc.king} · ${e.sample_count} samples · teacher ${doc.teacher}`);
}

function renderDuel(doc) {
  const e = doc.eval;
  $("duel-meta").textContent = "a win needs both, each by its margin";
  if (e.error) {
    mount($("duel-wrap"), el("p", { class: "note" }, el("span", { class: "fail-code" }, "crashed"), e.error));
    return;
  }
  // lower is closer to the teacher, so the challenger's lead is king − challenger; the result says
  // by how much it cleared the margin, or how much more it needed
  const rows = [
    ["KL at the output", e.score_challenger, e.score_king, e.margin, 5],
    ["hidden-state distance", e.dist_challenger, e.dist_king, e.dist_margin, 4],
  ].map(([name, chal, king, need, d]) => {
    const lead = chal == null || king == null ? null : king - chal;
    const pass = lead != null && need != null && lead >= need;
    const vsKing = lead == null ? "—" : lead === 0 ? "even" : `${num(Math.abs(lead), d)} ${lead > 0 ? "better" : "worse"}`;
    const verdict = lead == null || need == null
      ? null
      : el("span", { class: `verdict-badge ${pass ? "crowned" : "lost"}` },
          `${pass ? "passed by" : "short by"} ${num(Math.abs(lead - need), d)}`);
    return el("tr", {},
      el("td", { class: "measure" }, name),
      el("td", { class: "r" }, num(chal, d)),
      el("td", { class: "r" }, num(king, d)),
      el("td", { class: `r ${pass ? "gain" : "short"}` }, vsKing),
      el("td", { class: "r" }, need == null ? "—" : `${num(need, d)} better`),
      el("td", { class: "r" }, verdict ?? "—"));
  });
  const secs = (s) => (s == null ? "—" : fmtDuration(s * 1000));
  mount($("duel-wrap"),
    el("div", { class: "data-table-wrap" }, el("table", { class: "data-table duel-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "measure"), el("th", { class: "r" }, "challenger"), el("th", { class: "r" }, "king"),
        el("th", { class: "r", title: "the challenger against the king; lower is closer to the teacher, so lower is better" }, "vs king"),
        el("th", { class: "r", title: "a win needs the challenger at least this much better than the king" }, "needed"),
        el("th", { class: "r" }, "result"))),
      el("tbody", {}, rows))),
    el("p", { class: "note" },
      `Lower is closer to the teacher. Noise floor, the teacher reading its own answers again: KL ${num(e.noise_kl)}, distance ${num(e.noise_dist, 4)}. `
      + `Run time: challenger ${secs(e.challenger_s)}, king ${secs(e.king_s)}. Seed ${e.seed}, dataset ${e.dataset_hash}.`));
}

// mean 1 − cos per layer, weighted by answer tokens as the evaluator weighs its means; hidden
// states cover every answer token but the last
function layerMeans(samples, role) {
  const sums = new Map();
  let weight = 0;
  for (const s of samples) {
    const run = s.runs[role];
    if (!run) continue;
    const w = Math.max(s.answer_tokens - 1, 1);
    weight += w;
    for (const [layer, v] of Object.entries(run.one_minus_cos_by_layer)) sums.set(layer, (sums.get(layer) ?? 0) + v * w);
  }
  return weight ? [...sums].map(([layer, v]) => ({ layer, v: v / weight })) : [];
}

function renderLayers(doc) {
  const box = $("layers-chart");
  const series = SERIES.map(([role, label]) => ({ role, label, points: layerMeans(doc.samples, role) })).filter((s) => s.points.length);
  mount($("layers-meta"), ...series.map((s) => el("span", { class: `legend ${s.role}` }, el("i"), s.label)));
  if (!series.length) {
    mount(box, el("div", { class: "empty" }, "no per-layer records for this eval."));
    return;
  }
  const layers = series[0].points.map((p) => p.layer);
  const PAD = { top: 20, right: 24, bottom: 28, left: 64 };
  const HEIGHT = 260;
  const width = Math.max(box.clientWidth || 0, 320);
  const w = width - PAD.left - PAD.right;
  const h = HEIGHT - PAD.top - PAD.bottom;
  const top = Math.max(...series.flatMap((s) => s.points.map((p) => p.v))) * 1.1;
  const x = (i) => Math.round(PAD.left + (i / Math.max(layers.length - 1, 1)) * w);
  const y = (v) => Math.round(PAD.top + (1 - v / top) * h);

  const svg = svgEl("svg", { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, role: "img" });
  svg.append(svgEl("rect", { x: PAD.left, y: PAD.top, width: w, height: h, fill: "transparent", "pointer-events": "all" }));
  for (const v of [0, top / 1.1]) {
    svg.append(svgEl("line", { x1: PAD.left, y1: y(v) + 0.5, x2: width - PAD.right, y2: y(v) + 0.5, class: "grid" }));
    svg.append(svgEl("text", { x: PAD.left - 8, y: y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, v.toFixed(5)));
  }
  layers.forEach((layer, i) => svg.append(svgEl("text", { x: x(i), y: HEIGHT - 8, class: "tick", "text-anchor": "middle" }, layer)));
  // the noise floor first, so the two models draw over it
  for (const s of [...series].reverse()) {
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.v)}`).join("");
    svg.append(svgEl("path", { d, class: `layer-line ${s.role}` }));
    s.points.forEach((p, i) => svg.append(svgEl("rect", { x: x(i) - 3, y: y(p.v) - 3, width: 7, height: 7, class: `layer-mark ${s.role}` })));
  }
  const crosshair = svgEl("line", { y1: PAD.top, y2: HEIGHT - PAD.bottom, class: "crosshair", visibility: "hidden" });
  svg.append(crosshair);
  const tip = el("div", { class: "hero-chart-tip", hidden: true });
  svg.addEventListener("pointermove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / Math.max(rect.width, 1)) * width;
    const i = Math.min(layers.length - 1, Math.max(0, Math.round(((px - PAD.left) / w) * (layers.length - 1))));
    crosshair.setAttribute("x1", x(i) + 0.5);
    crosshair.setAttribute("x2", x(i) + 0.5);
    crosshair.removeAttribute("visibility");
    mount(tip,
      el("div", { class: "hero-chart-tip-row" }, el("b", {}, `layer ${layers[i].slice(1)}`), el("span", { class: "lbl" }, "1 − cos to the teacher")),
      ...series.map((s) => el("div", { class: "hero-chart-tip-meta" }, `${s.label} ${s.points[i].v.toFixed(6)}`)));
    placeTip(tip, svg, width, x(i), PAD.top);
  });
  svg.addEventListener("pointerleave", () => {
    crosshair.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
  mount(box, el("div", { class: "hero-chart-plot" }, svg, tip));
}

function renderSamples(doc) {
  // worst for the challenger first: the samples where the king stayed closest to the teacher
  const rows = doc.samples
    .filter((s) => s.runs.challenger && s.runs.king)
    .map((s) => ({ ...s, lead: s.runs.king.kl - s.runs.challenger.kl }))
    .sort((a, b) => a.lead - b.lead);
  $("samples-meta").textContent = rows.length ? `${rows.length} samples · worst for the challenger first` : "";
  if (!rows.length) {
    const why = doc.artifacts.includes("samples.jsonl") ? "the sample artifact holds no records." : "no sample artifact for this eval.";
    mount($("samples-wrap"), el("div", { class: "empty" }, why));
    return;
  }
  const body = rows.map((s) => el("tr", {},
    el("td", { class: "sample" }, s.sample_id),
    el("td", { class: "r" }, String(s.prompt_tokens)),
    el("td", { class: "r" }, String(s.answer_tokens)),
    el("td", { class: "end", title: s.ended_with_eos ? "the teacher ended its answer" : "the answer hit the token limit" }, s.ended_with_eos ? "eos" : "cut"),
    el("td", { class: "r" }, num(s.runs.challenger.kl)),
    el("td", { class: "r" }, num(s.runs.king.kl)),
    el("td", { class: `r ${s.lead >= 0 ? "gain" : "short"}` }, signed(s.lead, 5)),
    el("td", { class: "r" }, pct(s.runs.challenger.top1_agreement)),
    el("td", { class: "r" }, pct(s.runs.king.top1_agreement))));
  mount($("samples-wrap"), el("table", { class: "data-table" },
    el("thead", {}, el("tr", {},
      el("th", {}, "sample"), el("th", { class: "r" }, "prompt"), el("th", { class: "r" }, "answer"), el("th", {}, "end"),
      el("th", { class: "r" }, "KL challenger"), el("th", { class: "r" }, "KL king"),
      el("th", { class: "r", title: "KL king − KL challenger" }, "lead"),
      el("th", { class: "r", title: "how often the model's top token is the teacher's" }, "top-1 challenger"),
      el("th", { class: "r" }, "top-1 king"))),
    el("tbody", {}, body)));
}

function renderArtifacts(doc) {
  $("artifacts-meta").textContent = `evals/${shortHotkey(doc.hotkey)}/${doc.eval.id}/`;
  if (!doc.artifacts.length) {
    mount($("artifacts-wrap"), el("div", { class: "empty" }, "no artifacts were uploaded for this eval."));
    return;
  }
  const all = el("button", { type: "button", class: "zip" }, "download all · zip");
  all.addEventListener("click", () => downloadAll(all, doc));
  mount($("artifacts-wrap"), el("div", { class: "actions" }, all), el("div", { class: "data-table-wrap" }, el("table", { class: "data-table" },
    el("thead", {}, el("tr", {}, el("th", {}, "file"), el("th", {}, "holds"), el("th", { class: "r" }, ""))),
    el("tbody", {}, doc.artifacts.map((name) => el("tr", {},
      el("td", { class: "artifact" }, name),
      el("td", { class: "holds" }, WHAT[name] ?? ""),
      el("td", { class: "r" }, el("a", { class: "download", href: `api/evals/${doc.eval.id}/${name}`, download: `eval-${doc.eval.id}-${name}` }, "download ↓"))))))));
}

async function downloadAll(button, doc) {
  button.disabled = true;
  button.textContent = "zipping…";
  try {
    const files = await Promise.all(doc.artifacts.map(async (name) => {
      const res = await fetch(`api/evals/${doc.eval.id}/${name}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`${name}: ${res.status}`);
      return { name, bytes: new Uint8Array(await res.arrayBuffer()) };
    }));
    const a = el("a", { href: URL.createObjectURL(zip(files)), download: `eval-${doc.eval.id}-artifacts.zip` });
    a.click();
    URL.revokeObjectURL(a.href);
    button.textContent = "download all · zip";
  } catch (err) {
    console.warn("artifact zip failed", err);
    button.textContent = "zip failed · retry";
  } finally {
    button.disabled = false;
  }
}
