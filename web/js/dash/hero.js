// The dashboard's head: the round tabs and the round's best model.

import { el, mount, link } from "../dom.js";
import { $ } from "../page.js";
import { ROUNDS, hubUrl, kingGains, store } from "./store.js";

// the dashboard's head: the round tabs, and the round's best model: its reigning king, the closest to the
// teacher, with how much closer than genesis it is. Not the benchmark feed's, so a reload or a tab click
// shows the same thing. `onRound` redraws the page after a tab click.
export function renderHero(onRound) {
  const { state } = store;
  document.querySelectorAll(".stamp-tab").forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.tab === store.round));
    tab.onclick = () => {
      store.round = tab.dataset.tab;
      onRound();
    };
  });
  const { open, genesis } = ROUNDS[store.round];
  const king = kingGains()
    .filter((k) => k.submission_id)
    .at(-1);
  const teacher = state.teacher.split("/").pop();
  const [name, repo] = king ? [king.name, king.hf_repo] : [genesis.name, genesis.repo];
  const url = repo ? hubUrl(repo) : null;
  $("hero-label").textContent = `${open ? "Open round" : "Closed round"} · best model`;
  mount($("hero-king"), url ? link(url, name) : name);
  mount(
    $("hero-line"),
    ...(king
      ? [
          el("b", {}, `${king.gain.toFixed(1)}%`),
          "closer to ",
          link(hubUrl(state.teacher), teacher),
          " than ",
          link(hubUrl(genesis.repo), genesis.versus),
          " on eval's samples",
        ]
      : []),
  );
}
