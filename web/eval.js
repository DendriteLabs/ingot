// One eval's page: the duel with the king on both win measures, how each domain moved the lead,
// where in the network each model drifts from the teacher, every sample, and the raw artifacts.
// Reads api/evals/<id>.json.

import { el, mount } from "./js/dom.js";
import { fmtDateTime, fmtDuration, shortHotkey } from "./js/format.js";
import { $, minerLink, mountHeader, placeTip, svgEl } from "./js/page.js";
import { zip } from "./js/zip.js";

const id = new URLSearchParams(location.search).get("id");
const num = (v, d = 5) => (v == null ? "—" : Number(v).toFixed(d));
const signed = (v, d) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
const pct = (v) => (v == null ? "—" : `${(100 * v).toFixed(2)}%`);
// lower is closer to the teacher, so the challenger's lead is king − challenger, as a share of the king
const versus = (chal, king) => (chal === king ? "even" : `${Math.abs((100 * (king - chal)) / king).toFixed(2)}% ${chal < king ? "better" : "worse"}`);
const scored = (samples) => samples.filter((s) => s.runs.challenger && s.runs.king);
// what the layers chart and the samples table show: the picked group's samples, or every one
const shown = (doc) => (picked ? groupSamples(scored(doc.samples)).groups.get(picked) : scored(doc.samples));
// the three roles a sample record can hold, as the chart draws them
const SERIES = [["challenger", "challenger"], ["king", "king"], ["teacher_rerun", "teacher reread"]];
const WHAT = {
  "summary.json": "the eval's row: both scores, the margins, the noise floor, seed and timings",
  "samples.jsonl": "one record per sample: KL, top-1 agreement, distance and 1 − cos by layer for each model",
};
// records from before the evaluator wrote each sample's domain group by source dataset instead,
// the datasets with fewer samples than this folded into one row
const FOLD_BELOW = 8;
// the group the samples table is narrowed to, null for every sample
let picked = null;

mountHeader();
load();

async function load() {
  const res = /^\d+$/.test(id ?? "") ? await fetch(`api/evals/${id}.json`, { cache: "no-store" }) : null;
  if (!res?.ok) {
    $("ev-title").textContent = "No such eval";
    $("ev-sub").textContent = id ? `nothing is recorded for eval ${id}.` : "open an eval from the history table.";
    for (const box of ["duel-wrap", "domains-wrap", "samples-wrap"]) mount($(box), el("div", { class: "empty" }, "—"));
    return;
  }
  const doc = await res.json();
  document.title = `INGOT · eval ${id}`;
  $("link-github").href = doc.links.github;
  $("foot-netuid").textContent = `Ingot · subnet ${doc.netuid}`;
  renderHead(doc);
  renderDuel(doc);
  renderDomains(doc);
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
  // lower is closer to the teacher, so the challenger's lead is king − challenger, shown as a share
  // of the king's score; `need` is the absolute lead this eval required, the result says by how many
  // percentage points it cleared that, or fell short
  const rows = [
    ["KL at the output", e.score_challenger, e.score_king, e.margin],
    ["hidden-state distance", e.dist_challenger, e.dist_king, e.dist_margin],
  ].map(([name, chal, king, need]) => {
    const lead = chal == null || king == null ? null : king - chal;
    const share = (v) => (100 * v) / king;
    const pass = lead != null && need != null && lead >= need;
    const vsKing = lead == null ? "—" : versus(chal, king);
    const verdict = lead == null || need == null
      ? null
      : el("span", { class: `verdict-badge ${pass ? "crowned" : "lost"}` },
          `${pass ? "passed by" : "short by"} ${Math.abs(share(lead) - share(need)).toFixed(2)} pp`);
    return el("tr", {},
      el("td", { class: "measure" }, name),
      el("td", { class: "r" }, num(chal, 6)),
      el("td", { class: "r" }, num(king, 6)),
      el("td", { class: `r ${pass ? "gain" : "short"}` }, vsKing),
      el("td", { class: "r" }, need == null ? "—" : `${share(need).toFixed(2)}% better`),
      el("td", { class: "r" }, verdict ?? "—"));
  });
  const secs = (s) => (s == null ? "—" : fmtDuration(s * 1000));
  mount($("duel-wrap"),
    el("div", { class: "data-table-wrap" }, el("table", { class: "data-table duel-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, "measure"), el("th", { class: "r" }, "challenger"), el("th", { class: "r" }, "king"),
        el("th", { class: "r", title: "the challenger against the king, as a share of the king's score; lower is closer to the teacher, so lower is better" }, "vs king"),
        el("th", { class: "r", title: "a win needs the challenger at least this share of the king's score better" }, "needed"),
        el("th", { class: "r" }, "result"))),
      el("tbody", {}, rows))),
    el("p", { class: "note" },
      `Lower is closer to the teacher. Noise floor, the teacher reading its own answers again: KL ${num(e.noise_kl)}, distance ${num(e.noise_dist, 4)}. `
      + `Run time: challenger ${secs(e.challenger_s)}, king ${secs(e.king_s)}. Seed ${e.seed}, dataset ${e.dataset_hash}.`));
}

// the samples by domain; records from before domains by source dataset, with the rare datasets in
// one row
function groupSamples(samples) {
  const byDomain = samples.some((s) => s.domain);
  const groups = new Map();
  for (const s of samples) {
    const key = byDomain ? (s.domain || "—").replaceAll("_", " ") : s.sample_id.split("/")[0];
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(s);
  }
  const rare = byDomain ? [] : [...groups].filter(([, members]) => members.length < FOLD_BELOW);
  if (rare.length > 1) {
    for (const [key] of rare) groups.delete(key);
    groups.set(`${rare.length} other sources`, rare.flatMap(([, members]) => members));
  }
  return { byDomain, groups };
}

// token-weighted sums as the evaluator forms its scores: KL over every answer token, the distance
// over every answer token but the last
function totals(samples) {
  const t = { tokens: 0, klChal: 0, klKing: 0, states: 0, distChal: 0, distKing: 0 };
  for (const s of samples) {
    const n = s.answer_tokens;
    t.tokens += n;
    t.klChal += s.runs.challenger.kl * n;
    t.klKing += s.runs.king.kl * n;
    t.states += n - 1;
    t.distChal += s.runs.challenger.dist * (n - 1);
    t.distKing += s.runs.king.dist * (n - 1);
  }
  return t;
}

function renderDomains(doc) {
  const samples = scored(doc.samples);
  if (!samples.length) {
    mount($("domains-wrap"), el("div", { class: "empty" }, "no sample records for this eval."));
    return;
  }
  const { byDomain, groups } = groupSamples(samples);
  const all = totals(samples);
  // what a group adds to the challenger's KL lead, in points of the king's KL: the rows add up to
  // the duel's KL "vs king", so a row below zero is ground the challenger lost there
  const rows = [...groups]
    .map(([name, members]) => {
      const t = totals(members);
      return { name, count: members.length, t, adds: (100 * (t.klKing - t.klChal)) / all.klKing };
    })
    .sort((a, b) => b.t.tokens - a.t.tokens);
  const widest = Math.max(...rows.map((r) => Math.abs(r.adds)));
  $("domains-meta").textContent = byDomain ? "pick one to list its samples" : "by source dataset: these records predate domains";
  const trs = rows.map((r) => {
    const tr = el("tr", { class: "open", tabindex: "0", title: `list the ${r.name} samples` },
      el("td", { class: "domain" }, r.name),
      el("td", { class: "r c-count" }, String(r.count)),
      el("td", { class: "r c-share" }, pct(r.t.tokens / all.tokens)),
      el("td", { class: `r ${r.t.klChal <= r.t.klKing ? "gain" : "short"}`, title: `KL challenger ${num(r.t.klChal / r.t.tokens)}, king ${num(r.t.klKing / r.t.tokens)}` },
        versus(r.t.klChal, r.t.klKing)),
      el("td", { class: `r c-dist ${r.t.distChal <= r.t.distKing ? "gain" : "short"}`, title: `distance challenger ${num(r.t.distChal / r.t.states, 4)}, king ${num(r.t.distKing / r.t.states, 4)}` },
        versus(r.t.distChal, r.t.distKing)),
      el("td", { class: "c-bar" }, el("span", { class: "lead-bar" },
        el("i", { class: r.adds >= 0 ? "up" : "down", style: `width: ${widest ? (100 * Math.abs(r.adds)) / widest : 0}%` }))),
      el("td", { class: `r ${r.adds >= 0 ? "gain" : "short"}` }, `${signed(r.adds, 2)} pp`));
    tr.dataset.group = r.name;
    const pick = () => {
      picked = picked === r.name ? null : r.name;
      for (const row of trs) row.classList.toggle("picked", row.dataset.group === picked);
      renderLayers(doc);
      renderSamples(doc);
    };
    tr.addEventListener("click", pick);
    tr.addEventListener("keydown", (ev) => ev.key === "Enter" && pick());
    return tr;
  });
  mount($("domains-wrap"),
    el("div", { class: "data-table-wrap" }, el("table", { class: "data-table domains-table" },
      el("thead", {}, el("tr", {},
        el("th", {}, byDomain ? "domain" : "source"),
        el("th", { class: "r c-count" }, "samples"),
        el("th", { class: "r c-share", title: "its answer tokens as a share of all of them: the scores weigh every answer token the same" }, "weight"),
        el("th", { class: "r", title: "the challenger's KL against the king's on these samples; lower is closer to the teacher" }, "KL vs king"),
        el("th", { class: "r c-dist", title: "the challenger's hidden-state distance against the king's on these samples" }, "distance vs king"),
        el("th", { class: "c-bar" }, ""),
        el("th", { class: "r", title: "what these samples add to the challenger's KL lead, in points of the king's KL; the rows add up to the duel's KL vs king" }, "KL lead"))),
      el("tbody", {}, trs))),
    el("p", { class: "note" },
      "KL lead: what each row adds to the challenger's lead at the output, in percentage points of the king's KL. "
      + "The rows add up to the duel's KL result, so a row below zero is where the challenger lost ground to the king."));
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
  const series = SERIES.map(([role, label]) => ({ role, label, points: layerMeans(shown(doc), role) })).filter((s) => s.points.length);
  mount($("layers-meta"), picked ? el("b", {}, picked) : null, ...series.map((s) => el("span", { class: `legend ${s.role}` }, el("i"), s.label)));
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
  const samples = scored(doc.samples);
  // worst for the challenger first: the samples where the king stayed closest to the teacher
  const rows = shown(doc)
    .map((s) => ({ ...s, lead: s.runs.king.kl - s.runs.challenger.kl }))
    .sort((a, b) => a.lead - b.lead);
  if (picked) {
    const all = el("button", { type: "button", class: "show-all" }, "show all");
    all.addEventListener("click", () => $("domains-wrap").querySelector("tr.picked")?.click());
    mount($("samples-meta"), el("b", {}, picked), ` · ${rows.length} of ${samples.length} samples · worst for the challenger first · `, all);
  } else {
    $("samples-meta").textContent = rows.length ? `${rows.length} samples · worst for the challenger first` : "";
  }
  if (!rows.length) {
    const why = doc.artifacts.includes("samples.jsonl") ? "the sample artifact holds no records." : "no sample artifact for this eval.";
    mount($("samples-wrap"), el("div", { class: "empty" }, why));
    return;
  }
  const body = rows.map((s) => el("tr", {},
    el("td", { class: "sample", title: s.sample_id }, s.sample_id),
    el("td", { class: "r c-prompt" }, String(s.prompt_tokens)),
    el("td", { class: "r c-answer" }, String(s.answer_tokens),
      s.ended_with_eos ? null : el("span", { class: "cut", title: "the answer hit the token limit" }, " cut")),
    el("td", { class: "r c-chal" }, num(s.runs.challenger.kl)),
    el("td", { class: "r c-king" }, num(s.runs.king.kl)),
    el("td", { class: `r c-lead ${s.lead >= 0 ? "gain" : "short"}` }, signed(s.lead, 5)),
    el("td", { class: "r c-top1" }, pct(s.runs.challenger.top1_agreement)),
    el("td", { class: "r c-top1k" }, pct(s.runs.king.top1_agreement))));
  mount($("samples-wrap"), el("table", { class: "data-table samples-table" },
    el("thead", {}, el("tr", {},
      el("th", {}, "sample"),
      el("th", { class: "r c-prompt" }, "prompt"),
      el("th", { class: "r c-answer", title: "answer tokens; cut: the answer hit the token limit" }, "answer"),
      el("th", { class: "r c-chal" }, "KL challenger"),
      el("th", { class: "r c-king" }, "KL king"),
      el("th", { class: "r c-lead", title: "KL king − KL challenger" }, "lead"),
      el("th", { class: "r c-top1", title: "how often the model's top token is the teacher's" }, "top-1 challenger"),
      el("th", { class: "r c-top1k" }, "top-1 king"))),
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
