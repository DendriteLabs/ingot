// The round's finished duels and its failures.

import { el, mount, link } from "../dom.js";
import { fmtDateTime, fmtRelative } from "../format.js";
import { $ } from "../page.js";
import { ROUNDS, hotkeyCell, hubUrl, inRound, kingGains, store, uidCell } from "./store.js";

const FAIL_CODE = { INVALID: "invalid", DUPLICATE: "duplicate", FAILED: "crashed", UPLOAD: "upload" };

export function renderHistory() {
  const { state } = store;
  // finished duels of the round only: a running eval is in the queue, a crashed one is in fails.
  // An eval's round is its king's.
  const kings = kingGains();
  const ids = new Set(kings.map((k) => k.id));
  const rows = state.history.filter((e) => e.won != null && ids.has(e.king_id));
  $("history-meta").textContent = `${rows.length} shown`;
  if (!rows.length) {
    mount($("history-wrap"), el("div", { class: "empty" }, "no completed duels yet."));
    return;
  }
  // each cell is the challenger's lead over the king, king − challenger (lower is closer to the
  // teacher), as a share of the king's score; the bar that eval was judged by (`need`, absolute) is
  // in the tooltip, the column title carries the share the next eval needs. A crowned row's values
  // are bold. A lead below FAR is a broken or far-off model: one bucket, the exact share in the tooltip
  const FAR = -50;
  const marginCell = (chal, king, need, won) => {
    if (king == null) return el("span", { class: "muted-dash" }, "—");
    if (chal == null) return el("span", { class: "margin-pct", title: "challenger diverged · no finite score" }, `< −${-FAR}%`);
    const lead = king - chal;
    const pct = (v) => `${((100 * v) / king).toFixed(2)}%`;
    const far = (100 * lead) / king < FAR;
    return el(
      "span",
      {
        class: `margin-pct${won ? " crowned" : ""}`,
        title: `${lead < 0 ? "−" : "+"}${pct(Math.abs(lead))} · challenger ${Number(chal).toFixed(6)} / king ${Number(king).toFixed(6)}${need == null ? "" : need < 0 ? ` · may lose at most ${pct(-need)}` : ` · needed ≥ ${pct(need)}`}`,
      },
      far ? `< −${-FAR}%` : `${lead < 0 ? "−" : "+"}${pct(Math.abs(lead))}`,
    );
  };
  const share = (current, row, king) => (current != null ? current : row != null && king ? row / king : null);
  const needed = (m) => (m == null ? "" : m < 0 ? ` ≥ −${+(-100 * m).toFixed(2)}%` : ` ≥ ${+(100 * m).toFixed(2)}%`);
  const repos = new Map(state.kings.map((k) => [k.id, k.hf_repo]));
  const kingCell = (e) => {
    const repo = repos.get(e.king_id);
    return el("span", { class: "model-cell" }, repo ? link(hubUrl(repo), e.king, { title: repo }) : e.king);
  };
  // a whole row opens its eval; the uid, hotkey and king links inside it go to their own pages
  const open = (ev, url) => {
    if (ev.target.closest("a")) return;
    if (ev.ctrlKey || ev.metaKey || ev.button === 1) window.open(url, "_blank", "noopener");
    else location.href = url;
  };
  const body = rows.map((e) => {
    const v = e.won ? "crowned" : "lost";
    const url = `eval.html?id=${e.id}`;
    const tr = el(
      "tr",
      { class: e.won && e.king_id === kings.at(-2)?.id ? "open crowned-now" : "open", tabindex: "0", title: "open this eval" },
      el("td", { class: "when", title: fmtDateTime(e.finished_at) }, fmtRelative(e.finished_at)),
      uidCell(e.uid, e.hotkey),
      hotkeyCell(e.hotkey),
      el("td", { class: "model vs" }, kingCell(e)),
      el("td", { class: "center" }, marginCell(e.score_challenger, e.score_king, e.margin, e.won)),
      el("td", { class: "r" }, el("span", { class: `verdict-badge ${v}` }, v)),
    );
    tr.addEventListener("click", (ev) => open(ev, url));
    tr.addEventListener("auxclick", (ev) => ev.button === 1 && open(ev, url));
    tr.addEventListener("keydown", (ev) => ev.key === "Enter" && open(ev, url));
    return tr;
  });
  mount(
    $("history-wrap"),
    el(
      "table",
      { class: "data-table" },
      el(
        "thead",
        {},
        el(
          "tr",
          {},
          el("th", {}, "when"),
          el("th", {}, "uid"),
          el("th", {}, "hotkey"),
          el("th", {}, "vs king"),
          el(
            "th",
            { class: "center", title: "(king KL − challenger KL) / king KL; a win needs at least this" },
            `KL margin${needed(share(state.margins?.kl, rows[0].margin, rows[0].score_king))}`,
          ),
          el("th", { class: "r" }, "result"),
        ),
      ),
      el("tbody", {}, body),
    ),
  );
}

export function renderFails() {
  const { state } = store;
  const rows = state.fails.filter(inRound);
  $("fails-meta").textContent = `${ROUNDS[store.round].label} · ${rows.length} shown`;
  if (!rows.length) {
    mount($("fails-wrap"), el("div", { class: "empty" }, "no failures."));
    return;
  }
  const body = rows.map((f) =>
    el(
      "tr",
      {},
      el("td", { class: "when", title: fmtDateTime(f.at) }, fmtRelative(f.at)),
      uidCell(f.uid, f.hotkey),
      hotkeyCell(f.hotkey),
      el(
        "td",
        { class: "fail-reason-cell" },
        el("span", { class: "fail-code" }, FAIL_CODE[f.stage] ?? f.stage.toLowerCase()),
        el("span", { class: "fail-reason", title: f.reason || "" }, f.reason || "—"),
      ),
    ),
  );
  mount(
    $("fails-wrap"),
    el(
      "table",
      { class: "data-table" },
      el("thead", {}, el("tr", {}, el("th", {}, "when"), el("th", {}, "uid"), el("th", {}, "hotkey"), el("th", {}, "reason"))),
      el("tbody", {}, body),
    ),
  );
}
