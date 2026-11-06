// The benchmark cards: /api/benchmarks.json and the noise, with a picker for the king compared.

import { el, mount } from "../dom.js";
import { $ } from "../page.js";
import {
  AA,
  aaEstimate,
  aaPendingNote,
  benchColumns,
  complete,
  kingNumber,
  loadNoise,
  noiseLine,
  protocolNote,
  rankKings as rank,
  roundRuns,
} from "../benchmarks.js";
import { ROUNDS, hubUrl, kingGains, numeralOf, store, suite } from "./store.js";

let benchmarksLoading = false;
let benchmarkDocument = null;
// the last benchmark feed read (/api/benchmarks.json), so switching rounds redraws without a fetch
let benchmarkFeed = null;
let noise = {};

// the round's kings with a run on its protocol, ranked
const rankKings = (runs, benchmarks) =>
  rank(
    runs,
    benchmarks,
    store.state ? kingGains().filter((k) => k.submission_id && runs.some((r) => r.model === k.name)) : [],
    suite().genesis,
  );

// the benchmark cards from the last feed read, unless that feed, the noise, the round and the kings are the
// ones already drawn; a round switch calls it at once instead of waiting for a fresh fetch
export function drawBenchmarks() {
  const { state } = store;
  if (!benchmarkFeed || !state) return;
  const fingerprint = JSON.stringify([benchmarkFeed, noise, store.round, state.kings.map((k) => k.name)]);
  if (fingerprint === benchmarkDocument) return;
  renderBenchmarkCharts(benchmarkFeed.results, suite().benchmarks);
  benchmarkDocument = fingerprint;
}

export async function loadBenchmarks() {
  const { state } = store;
  // the cards name the teacher and the kings from the state: nothing to draw before it is here
  if (benchmarksLoading || !state) return;
  benchmarksLoading = true;
  const box = $("benchmark-charts");
  try {
    const [res, n] = await Promise.all([fetch("/api/benchmarks.json", { cache: "no-store" }), loadNoise()]);
    noise = n;
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
    benchmarkFeed = doc;
    drawBenchmarks();
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

// the king a visitor picked, per round; unset (every page load) means the best overall
const chosenKing = {};

// the chosen king's name in the section head; it opens the benchmarked kings, by numeral, as a listbox
function renderKingPicker(ranked, pick, redraw, runs) {
  const { benchmarks, teacher } = suite();
  const total = benchmarks.length;
  const listed = ranked.slice().sort((a, b) => kingNumber(a.king.name) - kingNumber(b.king.name));
  const aaOf = (row) => (row.done === total ? aaEstimate(runs, row.king.name, teacher) : null);
  const options = listed.map((row, i) => {
    const aa = aaOf(row);
    return el(
      "li",
      {
        role: "option",
        id: `king-option-${i}`,
        "aria-selected": String(row === pick),
        class: row.done === total ? null : "running",
        title: aa
          ? `AA Intelligence Index, estimated: ${(100 * aa.share).toFixed(0)}% of the teacher's scores kept${aaPendingNote(aa)}`
          : `${row.done} of ${total} benchmarks done`,
      },
      el("span", { class: "kp-name" }, `King ${numeralOf(row.king)}`),
      el("span", { class: "kp-done" }, `${row.done}/${total}`),
      el("span", { class: "kp-bar" }, aa ? el("i", { style: `width: ${Math.min(100, (100 * aa.aa) / AA.teacher)}%` }) : null),
      el("span", { class: "kp-score" }, aa ? `≈${Math.round(aa.aa)}` : "—"),
    );
  });
  const pop = el(
    "div",
    { class: "kp-pop", hidden: true },
    el("div", { class: "kp-head" }, "benchmarks · AA index, est."),
    el("ul", { class: "kp-list", role: "listbox", id: "king-list", "aria-label": "The benchmarked kings by numeral" }, ...options),
  );
  const button = el(
    "button",
    {
      class: "kp-button",
      type: "button",
      "aria-haspopup": "listbox",
      "aria-expanded": "false",
      "aria-controls": "king-list",
      title: "compare another king",
    },
    `King ${numeralOf(pick.king)}`,
  );
  let active = listed.indexOf(pick);
  const show = (open) => {
    pop.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
    options.forEach((li, i) => li.classList.toggle("active", open && i === active));
  };
  const choose = (i) => {
    chosenKing[store.round] = listed[i].king.name;
    show(false);
    redraw();
    $("bench-picker").querySelector("button").focus();
  };
  button.onclick = () => show(pop.hidden);
  button.onkeydown = (ev) => {
    const step = { ArrowDown: 1, ArrowUp: -1 }[ev.key];
    if (step) {
      ev.preventDefault();
      active = pop.hidden ? active : (active + step + options.length) % options.length;
      show(true);
    } else if (ev.key === "Enter" && !pop.hidden) {
      ev.preventDefault();
      choose(active);
    } else if (ev.key === "Escape") show(false);
  };
  options.forEach((li, i) => {
    li.onclick = () => choose(i);
    li.onpointermove = () => {
      active = i;
      show(true);
    };
  });
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
  const { state } = store;
  const runs = roundRuns(results, store.round);
  const { genesis } = ROUNDS[store.round];
  const { teacher, genesis: genesisRun } = suite();
  const ranked = rankKings(runs, benchmarks);
  const pick = ranked.find((row) => row.king.name === chosenKing[store.round]) ?? ranked[0];
  if (pick) renderKingPicker(ranked, pick, () => renderBenchmarkCharts(results, benchmarks), runs);
  else mount($("bench-picker"));
  const models = [
    { tone: "teacher", label: "Qwen BF16", repo: state?.teacher, is: (r) => r.model === teacher },
    { tone: "genesis", label: genesis.label, repo: genesis.repo, is: (r) => r.model === genesisRun },
    ...(pick
      ? [{ tone: "king latest", label: `King ${numeralOf(pick.king)}`, repo: pick.king.hf_repo, is: (r) => r.model === pick.king.name }]
      : []),
  ];
  const pct = (v) => `${v.toFixed(1)}%`;
  const signed = (v) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(1)}`;
  // five or six benchmarks as one row of column cards; the 8-bit round's seven keep the 3 + 4 grid of bar cards
  const asColumns = benchmarks.length === 5 || benchmarks.length === 6;
  mount(
    $("benchmark-charts"),
    ...benchmarks.map(([name, kind]) => {
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
      const pendingOf = (row) => (row.run?.status === "running" ? "running" : "queued");
      const titleOf = (row) => {
        const { run } = row;
        const d = delta(row);
        const tasksOf = (r) => `${r.completed ?? "—"} / ${r.samples ?? "—"} tasks`;
        return scored(row)
          ? `${run.model}: ${pct(run.score)}${d == null ? "" : ` · ${signed(d)} points from ${ROUNDS[store.round].genesis.label}`} · ${tasksOf(run)}${run.borrowed ? " · its 1.75-bit round run" : ""}`
          : pendingOf(row) === "running" && run.completed != null
            ? `running · ${tasksOf(run)}`
            : pendingOf(row);
      };
      const bar = (row) => {
        const { tone, label, repo, run } = row;
        const pending = pendingOf(row);
        const d = delta(row);
        return el(
          repo ? "a" : "div",
          {
            class: `bench-row ${tone}${scored(row) ? "" : " missing"}`,
            href: repo ? hubUrl(repo) : null,
            target: repo ? "_blank" : null,
            rel: repo ? "noopener" : null,
            title: titleOf(row),
          },
          el("span", { class: "bench-label" }, label),
          el(
            "span",
            { class: "bench-track" },
            genesis == null ? null : el("i", { class: "bench-ref", style: `left: ${x(genesis)}%` }),
            scored(row) ? el("i", { class: "bench-bar", style: `width: ${Math.max(0.5, x(run.score))}%` }) : null,
          ),
          el(
            "span",
            { class: "bench-value" },
            d == null ? null : el("span", { class: `bench-delta${d > 0 ? " up" : ""}` }, signed(d)),
            scored(row) ? pct(run.score) : pending,
          ),
        );
      };
      const noisy = noiseLine(noise, name, rows.find((row) => row.tone === "genesis")?.run);
      if (asColumns) {
        return el(
          "article",
          { class: "benchmark-card" },
          el(
            "div",
            { class: "benchmark-chart-head" },
            el("h3", {}, name),
            el("div", { class: "benchmark-sub" }, el("span", { class: "benchmark-kind" }, kind), noisy),
          ),
          el(
            "div",
            { class: "bench-rows" },
            benchColumns(
              rows.map((row) => ({
                tone: row.tone,
                label: row.label,
                href: row.repo ? hubUrl(row.repo) : null,
                title: titleOf(row),
                score: scored(row) ? row.run.score : null,
                delta: delta(row),
                pending: pendingOf(row),
              })),
              genesis,
            ),
          ),
        );
      }
      return el(
        "article",
        { class: "benchmark-card" },
        el("div", { class: "benchmark-chart-head" }, el("h3", {}, name), el("span", { class: "benchmark-kind" }, kind)),
        el("div", { class: "bench-rows" }, noisy, ...rows.map(bar)),
      );
    }),
  );
  $("benchmark-charts").classList.toggle("cols", asColumns);
  $("benchmark-charts").style.setProperty("--cols", benchmarks.length);
  mount($("benchmarks-protocol"), protocolNote(runs, store.round));
}
