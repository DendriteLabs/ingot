// The external benchmark feed (/api/benchmarks.json) as both pages read it: the suite, each round's
// protocol, the teacher's run, and the figure built on the scores: the Artificial Analysis estimate.

import { el } from "./dom.js";

// each round's protocol, teacher and genesis runs, and suite; the 2-bit round's from King XVII on. The 8-bit
// round's teacher shows the 2-bit round's teacher runs (`teacherFrom`), so both rounds show one teacher; its own
// 8-bit run only where the 2-bit round has none (AutomationBench). MMMU is no longer in the suite.
export const ROUND_BENCH = {
  q2: {
    protocol: "t1-minp05-32k-150",
    teacher: "Qwen3.8-27B-BF16-t1minp05-32k",
    genesis: "Ingot-genesis-27B-Q2-t1minp05-32k",
    topK: 20,
    benchmarks: [
      ["Terminal-Bench 2.1", "terminal"],
      ["SWE-bench Multilingual", "coding"],
      ["GPQA Diamond", "science"],
      ["τ³-bench Banking", "tool use"],
      ["MATH-500", "math"],
      ["BBEH Mini", "reasoning"],
    ],
  },
  fp8: {
    protocol: "sampled-16k-150",
    ids: ["sampled-16k-150-v1"],
    teacher: "Qwen3.8-27B-BF16",
    genesis: "Ingot-genesis-27B-FP8",
    teacherFrom: "q2",
    benchmarks: [
      ["Terminal-Bench 2.1", "terminal"],
      ["SWE-bench Multilingual", "coding"],
      ["GPQA Diamond", "science"],
      ["τ³-bench Banking", "tool use"],
      ["MATH-500", "math"],
      ["AutomationBench", "automation"],
    ],
  },
};
export const complete = (r) => r.status === "complete" && typeof r.score === "number" && Number.isFinite(r.score);

// a round's runs: the feed's results on that round's protocol, a stopped run left out. With `teacherFrom`, the
// other round's teacher runs stand in for this round's teacher (named as it, `borrowed` set) on every benchmark
// that round has one for.
export function roundRuns(results, round) {
  const { protocol, ids, teacher, teacherFrom } = ROUND_BENCH[round];
  const ours = (id) => (ids ? ids.includes(id) : id?.startsWith(`${protocol}-`));
  const runs = results.filter((r) => ours(r.protocol?.id) && r.status !== "stopped");
  if (!teacherFrom) return runs;
  const borrowed = roundRuns(results, teacherFrom)
    .filter((r) => r.model === ROUND_BENCH[teacherFrom].teacher)
    .map((r) => ({ ...r, model: teacher, borrowed: true }));
  const covered = new Set(borrowed.map((r) => r.benchmark));
  return [...runs.filter((r) => !(r.model === teacher && covered.has(r.benchmark))), ...borrowed];
}

// a king's numeral as a number (Ingot-XVII-27B-Q2: 17), to order kings by
export function kingNumber(name) {
  const values = { I: 1, V: 5, X: 10, L: 50, C: 100 };
  const digits = [...(name.split("-")[1] ?? "")].map((c) => values[c] ?? 0);
  return digits.reduce((total, v, i) => total + (v < (digits[i + 1] ?? 0) ? -v : v), 0);
}

// a benchmark's models as columns from a floor at least 25 points under the lowest score
export function benchColumns(cols, genesis) {
  const scores = cols.map((c) => c.score).filter((v) => v != null);
  const hi = Math.max(...scores, 1);
  const lo = Math.min(...scores, hi);
  const floor = Math.max(0, Math.floor(lo - Math.max(25, hi - lo)));
  const y = (score) => (100 * (score - floor)) / (hi - floor);
  const signed = (v) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(1)}`;
  return el(
    "div",
    { class: "bench-cols" },
    ...cols.map((c) =>
      el(
        c.href ? "a" : "div",
        {
          class: `bench-col ${c.tone}${c.score == null ? " missing" : ""}`,
          title: c.title,
          href: c.href,
          target: c.href ? "_blank" : null,
          rel: c.href ? "noopener" : null,
        },
        el("span", { class: "bench-value" }, c.score == null ? c.pending : `${c.score.toFixed(1)}%`),
        el("span", { class: `bench-delta${c.delta > 0 ? " up" : ""}` }, c.delta == null ? " " : signed(c.delta)),
        el(
          "span",
          { class: "bench-track" },
          genesis == null ? null : el("i", { class: "bench-ref", style: `bottom: ${y(genesis)}%` }),
          c.score == null ? null : el("i", { class: "bench-bar", style: `height: ${Math.max(1, y(c.score))}%` }),
        ),
        el("span", { class: "bench-label" }, c.label),
      ),
    ),
  );
}

// the protocol under a round's benchmark cards, from the runs' own protocol records
export function protocolNote(all, round) {
  const { topK, benchmarks, teacherFrom } = ROUND_BENCH[round];
  // the round's own runs: a teacher run borrowed from another round ran on that round's settings
  const runs = all.filter((r) => !r.borrowed);
  const records = runs.map((r) => r.protocol).filter(Boolean);
  if (!records.length) return null;
  const p = records.find((r) => r.harbor_agent_timeout_multiplier) ?? records[0];
  const names = benchmarks.map(([name]) => name).filter((name) => runs.some((r) => r.benchmark === name && r.protocol?.agent_network));
  const sealed = names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names.at(-1)}` : names[0];
  const sampling = [
    `temperature ${p.temperature.toFixed(1)}`,
    `top-p ${p.top_p}`,
    p.min_p != null ? `min-p ${p.min_p}` : null,
    topK ? `top-k ${topK}` : null,
  ]
    .filter(Boolean)
    .join(", ");
  return el(
    "p",
    { class: "run-note quiet wide protocol-note info-src" },
    `${all.some((r) => r.borrowed) ? "The kings and genesis above ran" : "Every model above ran"} with the same settings: ${sampling}; replies up to `,
    `${p.max_tokens.toLocaleString("en-US")} tokens; agents get ${p.step_limit} steps; Harbor's time limits ×${p.harbor_timeout_multiplier}`,
    p.harbor_agent_timeout_multiplier
      ? `, Terminal-Bench's agent clock ×${p.harbor_agent_timeout_multiplier}, so a task ends at the step limit, not on the clock`
      : "",
    sealed ? `; on ${sealed} the agent has no internet, only the model (and τ³'s own runtime).` : ".",
    all.some((r) => r.borrowed)
      ? ` The teacher's scores are its ${ROUND_LABEL[teacherFrom]} round runs, the same as that round shows.`
      : "",
  );
}
const ROUND_LABEL = { q2: "1.75-bit", fp8: "8-bit" };

// run-to-run noise per benchmark in percent points, from /api/noise.json; {} while it is missing
export async function loadNoise() {
  try {
    const res = await fetch("/api/noise.json", { cache: "no-store" });
    return res.ok ? Object.fromEntries((await res.json()).benchmarks.map((b) => [b.benchmark, b.noise_pp])) : {};
  } catch (err) {
    console.warn("noise fetch failed", err);
    return {};
  }
}

// the line atop a benchmark card's box: the error of a score, per noise.json the larger of the run-to-run
// noise and the binomial error 100 * sqrt(p * (1 - p) / samples), taken at genesis's run, which the
// card's +/- figures are measured from
export function noiseLine(noise, name, run) {
  const p = run && complete(run) && run.samples ? run.score / 100 : null;
  const binomial = p == null ? null : 100 * Math.sqrt((p * (1 - p)) / run.samples);
  const errors = [noise[name], binomial].filter((v) => v != null);
  if (!errors.length) return null;
  return el(
    "div",
    {
      class: "benchmark-noise",
      title: `larger of run-to-run noise (${noise[name]?.toFixed(2) ?? "—"}%) and sampling error at genesis's score (${binomial?.toFixed(2) ?? "—"}%)`,
    },
    `±${Math.max(...errors).toFixed(2)}%`,
  );
}

export const scoreOf = (runs, model, benchmark) => {
  const run = runs.find((r) => r.benchmark === benchmark && r.model === model);
  return run && complete(run) ? run.score : null;
};

// Artificial Analysis Intelligence Index (artificialanalysis.ai, v4.3.2): ten
// evaluations Artificial Analysis runs itself, several with private test sets, so it cannot be run
// here. AA lists the teacher and the Mac-class models; it has not evaluated Bonsai 2 or any king.
export const AA = {
  version: "v4.3.2",
  url: "https://artificialanalysis.ai/methodology/intelligence-benchmarking",
  teacher: 34, // Qwen3.8 27B, reasoning (xhigh)
  // published scores; `estimate` marks the ones AA itself flags as estimates
  published: [
    {
      label: "Gemma 4 12B",
      aa: 14,
      estimate: true,
      href: "https://artificialanalysis.ai/models/releases/comparisons/gemma-4-12b-vs-qwen3-5-9b",
    },
    { label: "Qwen3.5 9B", aa: 11, href: "https://artificialanalysis.ai/models/qwen3-5-9b" },
    { label: "gpt-oss-20b", aa: 9, href: "https://artificialanalysis.ai/models/comparisons/qwen3-5-9b-vs-gpt-oss-20b" },
  ],
  // the estimate for a model AA has not run: the share of the teacher's scores it keeps on the four
  // benchmarks the Mac-class models also ran here (each capped at the teacher's, averaged within coding and
  // within reasoning, then over the two), raised to the
  // power that maps Qwen3.5 9B's and gpt-oss-20b's shares (0.55, 0.57) onto their AA scores (11, 9):
  // AA's harder suite falls off about with the square of the share
  shared: { coding: ["SWE-bench Multilingual", "Terminal-Bench 2.1"], reasoning: ["GPQA Diamond", "MATH-500"] },
  exponent: 2.1,
};
// The model needs a score on all four; a teacher benchmark without a score yet (a rerun replaces the
// finished row in the feed) is left out of its group, which then averages the rest, and `pending` names it.
export function aaEstimate(runs, model, teacher) {
  const values = [];
  const pending = [];
  for (const names of Object.values(AA.shared)) {
    const pairs = names.map((name) => [name, scoreOf(runs, model, name), scoreOf(runs, teacher, name)]);
    if (pairs.some(([, mine]) => mine == null)) return null;
    pending.push(...pairs.filter(([, , ref]) => !(ref > 0)).map(([name]) => name));
    const shares = pairs.filter(([, , ref]) => ref > 0).map(([, mine, ref]) => Math.min(mine / ref, 1));
    if (!shares.length) return null;
    values.push(shares.reduce((a, b) => a + b, 0) / shares.length);
  }
  const share = values.reduce((a, b) => a + b, 0) / values.length;
  return { aa: AA.teacher * share ** AA.exponent, share, pending };
}
// the estimate's note when a teacher run was left out
export const aaPendingNote = (est) =>
  est?.pending?.length ? ` (${est.pending.join(" and ")} left out until the teacher's run finishes)` : "";

// the files people download, in decimal GB
export const SIZES_GB = { q2: 5.95, fp8: 28.6, teacher: 54.4 };

// the round's kings, best first by the sum of their points over genesis: the +/- figures on the cards,
// added up. Each benchmark counts the points it moves, so MATH-500, where every model sits near 96,
// barely counts; a plain mean ranks the same but reads near 59% only because of it. A per-benchmark
// share of the gap to the teacher lets MATH's 1.4-point gap and its noise dominate. A king's benchmark
// run is named after it (2-bit and 8-bit kings share numerals, never names); a king still running
// ranks after every finished one, most done first.
export function rankKings(runs, benchmarks, kings, genesisRun) {
  const full = (row) => row.done === benchmarks.length;
  return kings
    .map((king, order) => {
      const pairs = benchmarks.map(([name]) => [scoreOf(runs, king.name, name), scoreOf(runs, genesisRun, name)]);
      return {
        king,
        order,
        done: pairs.filter(([mine]) => mine != null).length,
        ahead: pairs.filter(([mine, base]) => mine != null && base != null && mine >= base).length,
        sum: pairs.reduce((total, [mine, base]) => total + (mine != null && base != null ? mine - base : 0), 0),
      };
    })
    .sort((a, b) => full(b) - full(a) || (full(a) ? b.sum - a.sum : b.done - a.done || b.order - a.order));
}
