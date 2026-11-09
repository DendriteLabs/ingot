// What the dashboard's sections share: the last state read, the selected round and the helpers on both.

import { el } from "../dom.js";
import { shortHotkey } from "../format.js";
import { minerLink, svgEl } from "../page.js";
import { ROUND_BENCH } from "../benchmarks.js";

// `state` is the last /api/state.json read, `round` the round tab picked
export const store = { state: null, round: "q2" };

export const ROUNDS = {
  q2: {
    label: "1.75-bit",
    open: true,
    genesis: {
      label: "Bonsai 2 Q2",
      name: "Bonsai-2-27B-Q2",
      versus: "Bonsai-2-27B-Q2",
      about: "Prism's ternary quantization (1.75 bpw)",
      repo: "dendriteholdings/Ingot-genesis-27B-Q2",
    },
  },
  fp8: {
    label: "8-bit",
    open: false,
    genesis: {
      label: "Qwen FP8",
      name: "Qwen FP8",
      versus: "Qwen's own FP8",
      about: "Qwen's own quantization",
      repo: "dendriteholdings/Ingot-genesis-27B-FP8",
    },
  },
};
export const suite = () => ROUND_BENCH[store.round];
// a backend from before the 2-bit round sends no quant: everything it knows is 8-bit
export const inRound = (row) => (row.quant ?? "fp8") === store.round;

// a duel's [challenger, king] KL: under the current scoring rule when the dashboard has scored
// its records again (every eval's samples are public), else as the duel stored them
export const duelScores = (e) => {
  if (e.rescored_king != null) return [Number(e.rescored_challenger), Number(e.rescored_king)];
  const chal = e.duel_challenger ?? e.score_challenger;
  const king = e.duel_king ?? e.score_king;
  return [chal == null ? null : Number(chal), king == null ? null : Number(king)];
};

// a loser's KL against the king it faced, as a share of the king's: "4.7% above" or "0.3% below"
export const vsKing = (e) => {
  if (e.score_challenger == null) return "diverged, no finite score";
  const [chal, king] = duelScores(e);
  const lead = 100 * (1 - chal / king);
  return `${Math.abs(lead).toFixed(1)}% ${lead < 0 ? "above" : "below"} ${e.king}`;
};
export const hubUrl = (repo) => `https://huggingface.co/${repo}`;
export const uidCell = (uid, hotkey) =>
  el("td", { class: "uid" }, uid == null ? "—" : hotkey ? minerLink(store.state.netuid, hotkey, String(uid)) : String(uid));
export const hotkeyCell = (hotkey, cls = "hotkey") =>
  el("td", { class: cls }, hotkey ? minerLink(store.state.netuid, hotkey, shortHotkey(hotkey)) : "—");

// each eval scores its own samples, so divergences from different evals do not compare. A king's
// divergence relative to genesis is chained instead: the king it beat times the challenger / king
// ratio measured in its coronation eval, under the current scoring rule where the dashboard has
// rescored that eval. Returns the selected round's kings with `rel` (1 = genesis) and `gain` (%).
export function kingGains() {
  let rel = 1;
  return store.state.kings.filter(inRound).map((k) => {
    const [chal, king] = duelScores(k);
    if (chal != null) rel *= chal / king;
    return { ...k, rel, gain: 100 * (1 - rel) };
  });
}

// the kings in order with their chart value (`v`, the gain over genesis) and the step each took:
// how much lower its KL was than the king it beat, in that duel
export function kingSteps() {
  const crowned = kingGains().map((k) => ({ ...k, crowned: true, evalId: k.eval_id, t: new Date(k.crowned_at).getTime(), v: k.gain }));
  crowned.forEach((k, j) =>
    Object.assign(k, {
      beat: crowned[j - 1],
      step: k.duel_king != null ? 100 * (1 - duelScores(k)[0] / duelScores(k)[1]) : null,
    }),
  );
  return crowned;
}

export const numeralOf = (k) => (k.name === "GENESIS" ? "genesis" : k.name.split("-")[1]);

// <defs> with a vertical gradient named `id` fading to transparent, for the fill under a chart's
// line: `tone` picks its colour (fill-gold, fill-grey in the stylesheet)
export function gradient(id, tone = "fill-gold") {
  const fill = svgEl("linearGradient", { id, x1: 0, y1: 0, x2: 0, y2: 1 });
  fill.append(svgEl("stop", { offset: "0", class: `${tone} top` }), svgEl("stop", { offset: "1", class: `${tone} bottom` }));
  const defs = svgEl("defs");
  defs.append(fill);
  return defs;
}
