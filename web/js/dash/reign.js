// Weight holders and the registration burn.

import { el, mount, link } from "../dom.js";
import { fmtDateTime, fmtDuration, fmtRelative } from "../format.js";
import { $, placeTip, svgEl } from "../page.js";
import { gradient, hotkeyCell, hubUrl, store, uidCell } from "./store.js";

export function renderReign() {
  const { state } = store;
  const { reign } = state;
  $("reign-meta").textContent = `${reign.slots} slots · ${Math.round(reign.share * 100)}% each · ${reign.reign_days} days`;
  const now = Date.now();
  // the emission as five squares of one share each, a king's shares lit from the left
  const shares = (lit) =>
    el(
      "span",
      { class: "rc-slots", title: `${lit} of ${reign.slots} shares` },
      ...Array.from({ length: reign.slots }, (_, i) => el("i", { class: i < lit ? "on" : "" })),
    );
  const rows = reign.kings.map((k) => {
    const url = k.hf_repo ? hubUrl(k.hf_repo) : null;
    const slots = shares(Math.round(k.weight / reign.share));
    return el(
      "tr",
      { class: k.name === state.king?.name ? "reign-current" : "" },
      el("td", { class: "rc-model" }, url ? link(url, k.name, { title: k.hf_repo }) : k.name),
      uidCell(k.uid, k.hotkey),
      hotkeyCell(k.hotkey, "rc-hotkey"),
      el("td", { class: "when", title: fmtDateTime(k.crowned_at) }, fmtRelative(k.crowned_at)),
      el(
        "td",
        {
          class: "when",
          title: `earns ${Math.round(k.weight * 100)}% until ${fmtDateTime(k.ends_at)}, or until ${reign.slots} newer kings are crowned`,
        },
        `in ${fmtDuration(new Date(k.ends_at).getTime() - now)}`,
      ),
      el("td", { class: "r rc-weight" }, slots, el("span", {}, `${Math.round(k.weight * 100)}%`)),
    );
  });
  // one row per slot: a slot no king holds stays as an empty row; its share burns
  const dash = () => el("span", { class: "muted-dash" }, "—");
  for (let i = rows.length; i < reign.slots; i++) {
    rows.push(
      el(
        "tr",
        { title: "open slot: its share burns" },
        el("td", {}, dash()),
        el("td", {}, dash()),
        el("td", {}, dash()),
        el("td", {}, dash()),
        el("td", {}, dash()),
        el("td", { class: "r rc-weight" }, shares(0), dash()),
      ),
    );
  }
  mount(
    $("reign-wrap"),
    el(
      "div",
      { class: "data-table-wrap" },
      el(
        "table",
        { class: "data-table reign-table" },
        el(
          "thead",
          {},
          el(
            "tr",
            {},
            el("th", {}, "model"),
            el("th", {}, "uid"),
            el("th", {}, "hotkey"),
            el("th", {}, "crowned"),
            el("th", {}, "incentive ends"),
            el("th", { class: "r" }, "weight"),
          ),
        ),
        el("tbody", {}, rows),
      ),
    ),
  );
}

// the registration burn over the last 24 hours as a step line: each sample holds until the next one
export function renderRegistration() {
  const { state } = store;
  const box = $("registration-chart");
  const points = state.registration.map(([t, block, tao]) => ({ t: t * 1000, block, tao }));
  const latest = points.at(-1);
  $("registration-meta").textContent = latest ? `last 24h · now ${latest.tao.toFixed(4)} TAO` : "last 24h";
  if (points.length < 2) {
    mount(box, el("div", { class: "empty" }, "registration prices appear after the first samples."));
    return;
  }
  // beside weight holders the chart takes the table's height (the table, not its stretched panel)
  const table = $("reign-wrap").querySelector("table");
  const beside = matchMedia("(min-width: 1280px)").matches && table ? table.offsetHeight : 0;
  const HEIGHT = Math.max(220, beside);
  const PAD = { top: 20, right: 20, bottom: 26, left: 52 };
  const width = Math.max(box.clientWidth || 0, 320);
  const w = width - PAD.left - PAD.right;
  const h = HEIGHT - PAD.top - PAD.bottom;
  const t0 = points[0].t;
  const t1 = points.at(-1).t;
  const lo = Math.min(...points.map((p) => p.tao));
  const hi = Math.max(...points.map((p) => p.tao));
  const pad = Math.max((hi - lo) * 0.1, hi * 0.02);
  const x = (t) => PAD.left + ((t - t0) / Math.max(t1 - t0, 1)) * w;
  const y = (v) => PAD.top + (1 - (v - (lo - pad)) / (hi - lo + 2 * pad)) * h;
  const when = (t) => new Date(t).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });

  const svg = svgEl("svg", { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, role: "img" });
  svg.append(svgEl("rect", { x: PAD.left, y: PAD.top, width: w, height: h, fill: "transparent", "pointer-events": "all" }));
  for (const v of [lo, hi]) {
    svg.append(svgEl("line", { x1: PAD.left, y1: y(v) + 0.5, x2: width - PAD.right, y2: y(v) + 0.5, class: "grid" }));
    svg.append(
      svgEl("text", { x: PAD.left - 8, y: y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, v.toFixed(3)),
    );
  }
  for (const [t, anchor] of [
    [t0, "start"],
    [(t0 + t1) / 2, "middle"],
    [t1, "end"],
  ]) {
    svg.append(svgEl("text", { x: x(t), y: HEIGHT - 8, class: "tick", "text-anchor": anchor }, when(t)));
  }
  // a sample every 25 blocks: joined straight, a price step is steep enough to read as one
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.tao).toFixed(1)}`).join("");
  svg.append(gradient("registration-fill", "fill-grey"));
  svg.append(svgEl("path", { d: `${d}V${HEIGHT - PAD.bottom}H${x(points[0].t)}Z`, class: "area", fill: "url(#registration-fill)" }));
  svg.append(svgEl("path", { d, class: "reg-line" }));
  const square = (p, cls) => svgEl("rect", { x: Math.round(x(p.t)) - 3, y: Math.round(y(p.tao)) - 3, width: 7, height: 7, class: cls });
  svg.append(square(latest, "reg-now"));
  const active = square(latest, "reg-active");
  active.setAttribute("visibility", "hidden");
  const crosshair = svgEl("line", { y1: PAD.top, y2: HEIGHT - PAD.bottom, class: "crosshair", visibility: "hidden" });
  svg.append(crosshair, active);

  const tip = el("div", { class: "hero-chart-tip", hidden: true });
  svg.addEventListener("pointermove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const t = t0 + ((((ev.clientX - rect.left) / Math.max(rect.width, 1)) * width - PAD.left) / w) * (t1 - t0);
    let p = points[0];
    for (const q of points) if (q.t <= t) p = q;
    crosshair.setAttribute("x1", Math.round(x(p.t)) + 0.5);
    crosshair.setAttribute("x2", Math.round(x(p.t)) + 0.5);
    crosshair.removeAttribute("visibility");
    active.setAttribute("x", x(p.t) - 3);
    active.setAttribute("y", y(p.tao) - 3);
    active.removeAttribute("visibility");
    mount(
      tip,
      el(
        "div",
        { class: "hero-chart-tip-row" },
        el("b", {}, `${p.tao.toFixed(4)} TAO`),
        el("span", { class: "lbl" }, `block ${p.block.toLocaleString()}`),
      ),
      el("div", { class: "hero-chart-tip-meta" }, fmtDateTime(p.t)),
    );
    placeTip(tip, svg, width, x(p.t), PAD.top);
  });
  svg.addEventListener("pointerleave", () => {
    crosshair.setAttribute("visibility", "hidden");
    active.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
  mount(box, el("div", { class: "hero-chart-plot" }, svg, tip));
}
