// The external benchmark feed (/api/benchmarks.json) as both pages read it: the suite, each round's
// protocol, the teacher's run, and the two figures built on the scores: the Ingot Index and the
// intelligence density.

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

export const scoreOf = (runs, model, benchmark) => {
  const run = runs.find((r) => r.benchmark === benchmark && r.model === model);
  return run && complete(run) ? run.score : null;
};

// Ingot Index: how much of the teacher a model keeps, 100 = the teacher everywhere. Each benchmark's
// retention is the model's score over the teacher's, capped at 100% (a win on one benchmark cannot
// pay for a loss on another); a benchmark where the teacher scores under 10% counts in proportion;
// benchmarks weigh equally inside a category and the categories a third each. The same rule as the
// benchmark controller's `index` command without its KL penalty; a model missing benchmarks is
// scored on the ones it has, and `coverage` says how many.
const INDEX_CATEGORIES = {
  coding: ["SWE-bench Multilingual", "Terminal-Bench 2.1"],
  reasoning: ["GPQA Diamond", "MATH-500", "MMMU"],
  "tool use": ["τ³-bench Banking", "AutomationBench"],
};
export function ingotIndex(runs, model) {
  const categories = {};
  let covered = 0;
  for (const [category, names] of Object.entries(INDEX_CATEGORIES)) {
    let weight = 0, retained = 0;
    for (const name of names) {
      const mine = scoreOf(runs, model, name);
      const teacher = scoreOf(runs, TEACHER_RUN, name);
      if (mine == null || teacher == null || teacher <= 0) continue;
      const w = Math.min(1, teacher / 10);
      weight += w;
      retained += w * Math.min(mine / teacher, 1);
      covered++;
    }
    if (weight) categories[category] = (100 * retained) / weight;
  }
  const values = Object.values(categories);
  if (!values.length) return null;
  return { index: values.reduce((a, b) => a + b, 0) / values.length, categories, coverage: covered, total: BENCHMARKS.length };
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
