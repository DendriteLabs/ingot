// The external benchmark feed (/api/benchmarks.json) as both pages read it: the suite, each round's
// protocol, the teacher's run, and the figures built on the scores: the Artificial Analysis estimate
// and the intelligence density.

import { el } from "./dom.js";

export const BENCHMARKS = [
  ["SWE-bench Multilingual", "coding"], ["Terminal-Bench 2.1", "terminal"], ["AutomationBench", "automation"],
  ["GPQA Diamond", "science"], ["MATH-500", "math"], ["τ³-bench Banking", "tool use"], ["MMMU", "vision"],
];
export const PROTOCOL = "sampled-16k-150-v1";
// The 2-bit round's Terminal-Bench is on v2 (agent budget x24, verifier x2.0) so a task ends at the
// 150-step limit, not on the clock; the closed 8-bit round keeps its v1 runs
const PROTOCOLS = { q2: { "Terminal-Bench 2.1": "sampled-16k-150-v2" } };
export const protocolOf = (round, benchmark) => PROTOCOLS[round]?.[benchmark] ?? PROTOCOL;
export const TEACHER_RUN = "Qwen3.8-27B-BF16";
export const complete = (r) => r.status === "complete" && typeof r.score === "number" && Number.isFinite(r.score);

// a round's runs: the feed's results on that round's protocol
export const roundRuns = (results, round) => results.filter((r) => r.protocol?.id === protocolOf(round, r.benchmark));

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
  return el("div", { class: "benchmark-noise", title: `larger of run-to-run noise (${noise[name]?.toFixed(2) ?? "—"}%) and sampling error at genesis's score (${binomial?.toFixed(2) ?? "—"}%)` },
    `±${Math.max(...errors).toFixed(2)}%`);
}

export const scoreOf = (runs, model, benchmark) => {
  const run = runs.find((r) => r.benchmark === benchmark && r.model === model);
  return run && complete(run) ? run.score : null;
};

// Artificial Analysis Intelligence Index (artificialanalysis.ai, v4.3.2, read 2026-10-27): ten
// evaluations Artificial Analysis runs itself, several with private test sets, so it cannot be run
// here. AA lists the teacher and the Mac-class models; it has not evaluated Bonsai 2 or any king.
export const AA = {
  version: "v4.3.2",
  url: "https://artificialanalysis.ai/methodology/intelligence-benchmarking",
  teacher: 34, // Qwen3.8 27B, reasoning (xhigh)
  // published scores; `estimate` marks the ones AA itself flags as estimates
  published: [
    { label: "Gemma 4 12B", aa: 14, estimate: true, href: "https://artificialanalysis.ai/models/releases/comparisons/gemma-4-12b-vs-qwen3-5-9b" },
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
export function aaEstimate(runs, model) {
  const values = [];
  for (const names of Object.values(AA.shared)) {
    const shares = names.map((name) => [scoreOf(runs, model, name), scoreOf(runs, TEACHER_RUN, name)])
      .filter(([mine, teacher]) => mine != null && teacher > 0).map(([mine, teacher]) => Math.min(mine / teacher, 1));
    if (shares.length !== names.length) return null;
    values.push(shares.reduce((a, b) => a + b, 0) / shares.length);
  }
  const share = values.reduce((a, b) => a + b, 0) / values.length;
  return { aa: AA.teacher * share ** AA.exponent, share };
}

export const MAC_BOARD = {
  columns: [
    { kind: "Coding", name: "SWE-bench Multilingual" }, { kind: "Terminal", name: "Terminal-Bench 2.1" },
    { kind: "Science", name: "GPQA Diamond" }, { kind: "Math", name: "MATH-500" },
  ],
  rows: [
    { label: "Ingot-IV-27B-Q2", repo: "dendriteholdings/Ingot-IV-27B-Q2", gb: 5.95, king: true, scores: [61.0, 46.1, 77.3, 96.6] },
    // Bonsai 2 is the genesis run (its own file, the king's protocols), as everywhere on the site; the feed's
    { label: "Bonsai 2", repo: "prism-ml/Ternary-Bonsai-2-27B-gguf", gb: 5.95, scores: [58.3, 46.1, 69.7, 96.0] },
    // the usual 2-bit quant of the same teacher (Unsloth's UD-IQ2_XXS), on our pods
    { label: "Qwen3.8 27B IQ2_XXS", repo: "unsloth/Qwen3.8-27B-GGUF", gb: 7.27, scores: [19, 32.6, 54.5, 85.4] },
    { label: "gpt-oss-20b", repo: "openai/gpt-oss-20b", gb: 12.1, scores: [23, 19.1, 62.1, 89.2] },
    { label: "Qwen3.5 9B", repo: "Qwen/Qwen3.5-9B", gb: 5.7, scores: [29, 22.5, 57.1, 79.2] },
    { label: "Gemma 4 12B", repo: "google/gemma-4-12B-it-qat-q4_0-gguf", gb: 7.0, scores: [8, 10.1, 40.4, 95.2] },
  ],
  // the teacher on the same four, its feed scores (SWE-bench Multilingual on all 300; 76 on the subset)
  teacher: { label: "Qwen BF16", repo: "Qwen/Qwen3.8-27B", gb: 54.4, scores: [73.7, 64.0, 81.3, 97.4] },
};
// intelligence density on the four benchmarks every model on the board ran
export function boardDensity(row) {
  const mean = row.scores.reduce((a, b) => a + b, 0) / row.scores.length;
  return { density: -Math.log2(1 - mean / 100) / row.gb, mean };
}

// Intelligence density, Prism's figure: -log2(error rate) per GB, the error rate being one minus the
// mean benchmark score. Sizes are the files people download, in decimal GB.
export const SIZES_GB = { q2: 5.95, fp8: 28.6, teacher: 54.4 };
export function density(runs, model, sizeGb) {
  const scores = BENCHMARKS.map(([name]) => scoreOf(runs, model, name)).filter((v) => v != null);
  if (!scores.length) return null;
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  return { density: -Math.log2(1 - mean / 100) / sizeGb, mean, benchmarks: scores.length };
}

// the round's kings, best first by the sum of their points over genesis: the +/- figures on the cards,
// added up. Each benchmark counts the points it moves, so MATH-500, where every model sits near 96,
// barely counts; a plain mean ranks the same but reads near 59% only because of it. A per-benchmark
// share of the gap to the teacher lets MATH's 1.4-point gap and its noise dominate. A king's benchmark
// run is named after it (2-bit and 8-bit kings share numerals, never names); a king still running
// ranks after every finished one, most done first.
export function rankKings(runs, benchmarks, kings, genesisRun) {
  const full = (row) => row.done === benchmarks.length;
  return kings.map((king, order) => {
    const pairs = benchmarks.map(([name]) => [scoreOf(runs, king.name, name), scoreOf(runs, genesisRun, name)]);
    return {
      king, order,
      done: pairs.filter(([mine]) => mine != null).length,
      ahead: pairs.filter(([mine, base]) => mine != null && base != null && mine >= base).length,
      sum: pairs.reduce((total, [mine, base]) => total + (mine != null && base != null ? mine - base : 0), 0),
    };
  }).sort((a, b) => (full(b) - full(a)) || (full(a) ? b.sum - a.sum : b.done - a.done || b.order - a.order));
}
