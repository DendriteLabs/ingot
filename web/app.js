// Polls /api/state.json and renders every section from js/dash/; css/ingot.css styles exactly this markup.

import { $, mountHeader } from "./js/page.js";
import { drawBenchmarks, loadBenchmarks } from "./js/dash/bench.js";
import { renderHero } from "./js/dash/hero.js";
import { renderFails, renderHistory } from "./js/dash/history.js";
import { renderKingsChart, renderLineage } from "./js/dash/kings.js";
import { renderPipeline } from "./js/dash/queue.js";
import { renderRegistration, renderReign } from "./js/dash/reign.js";
import { ROUNDS, store } from "./js/dash/store.js";

const POLL_MS = 5000;
// an unchanged feed still redraws this often, so "2m ago" keeps up
const STALE_MS = 60000;
let feed = null;
let drawnAt = 0;

async function tick() {
  try {
    const res = await fetch("/api/state.json", { cache: "no-store" });
    if (!res.ok) return;
    // the same feed redraws nothing: a hover, a focus or a table scrolled sideways stays as it is
    const text = await res.text();
    if (text === feed && Date.now() - drawnAt < STALE_MS) return;
    store.state = JSON.parse(text);
    feed = text;
    drawnAt = Date.now();
    render();
    // the benchmark cards draw the kings from the state: redraw them once it is here or changed
    loadBenchmarks();
  } catch (err) {
    console.warn("state fetch failed", err);
  }
}

// a round tab redraws the page and the benchmark cards at once
function switchRound() {
  render();
  drawBenchmarks();
  loadBenchmarks();
}

function render() {
  const { state } = store;
  $("link-github").href = state.links.github;
  $("foot-netuid").textContent = `Ingot · subnet ${state.netuid}`;
  renderHero(switchRound);
  renderKingsChart();
  renderLineage();
  renderReign();
  renderRegistration();
  // a closed round queues nothing
  $("queue").hidden = !ROUNDS[store.round].open;
  renderPipeline();
  renderHistory();
  renderFails();
}

mountHeader();
new ResizeObserver(() => store.state && renderKingsChart()).observe($("kings-chart"));
let registrationWidth = 0;
new ResizeObserver(() => {
  // width only: the chart sets its own height, which would otherwise re-trigger this observer
  if (!store.state || $("registration-chart").clientWidth === registrationWidth) return;
  registrationWidth = $("registration-chart").clientWidth;
  renderRegistration();
}).observe($("registration-chart"));

tick();
loadBenchmarks();
setInterval(() => {
  tick();
  loadBenchmarks();
}, POLL_MS);
