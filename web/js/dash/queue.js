// The queue: every submission the validator is working on or has waiting.

import { el, mount } from "../dom.js";
import { fmtDateTime, fmtRelative } from "../format.js";
import { $ } from "../page.js";
import { hotkeyCell, store, uidCell } from "./store.js";

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

// a running eval's samples_done, written every 1/20th of its samples
function evalProgress(r) {
  if (r.eval_id == null || !r.sample_count) return null;
  if (!r.samples_done) return el("div", { class: "q-progress-label" }, "loading models");
  const pct = Math.min(100, (100 * r.samples_done) / r.sample_count);
  return el(
    "div",
    { class: "q-progress", title: `${pct.toFixed(0)}%` },
    el("div", { class: "q-progress-bar" }, el("div", { class: "q-progress-fill", style: `width: ${pct}%` })),
    el("span", { class: "q-progress-label" }, `${r.samples_done} / ${r.sample_count}`),
  );
}

export function renderPipeline() {
  const { state } = store;
  const rows = state.queue.rows;
  const working = rows.filter((r) => STAGES[r.state]?.[1] === "working").length;
  $("pipeline-meta").textContent = rows.length ? `${working} working · ${rows.length - working} queued` : "queue idle";
  const row = (r) => {
    const [stage, status] = STAGES[r.state] ?? [r.state.toLowerCase(), "queued"];
    return el(
      "tr",
      { class: status === "working" ? "working" : "" },
      el("td", {}, el("span", { class: "q-stage-badge" }, stage)),
      el("td", {}, el("span", { class: status === "working" ? "q-status working" : "q-status" }, status), evalProgress(r)),
      uidCell(r.uid, r.hotkey),
      hotkeyCell(r.hotkey),
      el("td", { class: "r when", title: fmtDateTime(r.updated_at) }, fmtRelative(r.updated_at)),
    );
  };
  const idle = el(
    "tr",
    {},
    el("td", {}, el("span", { class: "q-stage-badge" }, "-")),
    el("td", {}, el("span", { class: "q-status" }, "idle")),
    el("td", { class: "uid" }, "-"),
    el("td", { class: "hotkey" }, "-"),
    el("td", { class: "r when" }, "-"),
  );
  mount(
    $("pipeline-wrap"),
    el(
      "div",
      { class: "data-table-wrap" },
      el(
        "table",
        { class: "data-table q-table" },
        el(
          "thead",
          {},
          el(
            "tr",
            {},
            el("th", {}, "stage"),
            el("th", {}, "state"),
            el("th", {}, "uid"),
            el("th", {}, "hotkey"),
            el("th", { class: "r" }, "updated"),
          ),
        ),
        el("tbody", {}, rows.length ? rows.map(row) : idle),
      ),
    ),
  );
}
