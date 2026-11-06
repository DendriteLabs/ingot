// The kings: the lineage list and the chart of every judged model against genesis.

import { el, mount, link } from "../dom.js";
import { fmtRelative } from "../format.js";
import { $, placeTip, svgEl } from "../page.js";
import { ROUNDS, gradient, hubUrl, kingSteps, numeralOf, store, vsKing } from "./store.js";

export function renderLineage() {
  const { state } = store;
  const { genesis } = ROUNDS[store.round];
  const kings = kingSteps().filter((k) => k.submission_id);
  const current = kings.at(-1);
  const rows = kings
    .slice()
    .reverse()
    .map((k) => {
      const url = k.hf_repo ? hubUrl(k.hf_repo) : null;
      return el(
        "li",
        { class: k === current ? "current" : "" },
        el("span", { class: "numeral" }, numeralOf(k)),
        el(
          "span",
          { class: "who" },
          el("span", { class: "name" }, url ? link(url, k.name, { title: k.hf_repo }) : k.name),
          el("span", { class: "meta-line" }, `crowned ${fmtRelative(k.crowned_at)}`),
        ),
        el(
          "span",
          {
            class: "gain",
            title: `${k.gain.toFixed(2)}% closer to the original ${state.teacher} than ${genesis.versus} on eval's samples`,
          },
          `${k.gain.toFixed(1)}%`,
        ),
      );
    });
  rows.push(
    el(
      "li",
      { class: current ? "genesis" : "current" },
      el("span", { class: "numeral" }),
      el(
        "span",
        { class: "who" },
        el("span", { class: "name" }, link(hubUrl(genesis.repo), genesis.name, { title: genesis.repo })),
        el("span", { class: "meta-line" }, `genesis · ${genesis.about}`),
      ),
      el("span", { class: "gain baseline" }, "baseline"),
    ),
  );
  mount($("lineage"), el("ol", { "aria-label": "The kings, newest first" }, ...rows));
}

export function renderKingsChart() {
  const { state } = store;
  const box = $("kings-chart");
  // every judged model is a dot in the order it was judged, kings brass and losers grey. The y
  // axis is how much closer to the teacher a model is than genesis. The brass staircase is the bar
  // to beat: flat while a king holds, one step up at each coronation.
  const { genesis } = ROUNDS[store.round];
  mount(
    $("kings-meta"),
    "closer to the original ",
    link(hubUrl(state.teacher), state.teacher.split("/").pop()),
    " than ",
    link(hubUrl(genesis.repo), genesis.versus),
    " on eval's samples",
  );
  const crowned = kingSteps();
  if (crowned.length < 2) {
    mount(box, el("div", { class: "empty" }, "the staircase starts with the first coronation."));
    return;
  }
  const byId = new Map(crowned.map((k) => [k.id, k]));
  // a diverged challenger's score is published as null: it sits on the floor with the far-off losers
  const lost = state.losers
    .filter((e) => byId.has(e.king_id))
    .map((e) => ({
      ...e,
      crowned: false,
      evalId: e.id,
      t: new Date(e.finished_at).getTime(),
      v: e.score_challenger == null ? -Infinity : 100 * (1 - byId.get(e.king_id).rel * (Number(e.score_challenger) / Number(e.score_king))),
    }));
  const points = [...crowned, ...lost].sort((a, b) => a.t - b.t).map((p, i) => Object.assign(p, { i }));
  const PAD = { top: 18, right: 24, bottom: 28, left: 48 };
  const HEIGHT = 280;
  const width = Math.max(box.clientWidth || 0, 320);
  const w = width - PAD.left - PAD.right;
  const h = HEIGHT - PAD.top - PAD.bottom;
  const last = points.length - 1;
  const current = crowned.at(-1);
  const vTop = Math.max(current.v, 1);
  const vMax = Math.max(vTop, ...points.map((p) => p.v)) * 1.08;
  // a loser can be far worse than genesis; the floor is capped so the kings keep the room
  const vMin = Math.max(Math.min(0, ...points.map((p) => p.v)), -vTop * 0.3);
  const x = (i) => PAD.left + (i / last) * w;
  const y = (v) => PAD.top + (1 - (Math.max(v, vMin) - vMin) / (vMax - vMin)) * h;
  const numeral = (p) => (p.name === "GENESIS" ? "genesis" : p.name.split("-")[1]);

  const svg = svgEl("svg", { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, role: "img" });
  svg.append(svgEl("rect", { x: PAD.left, y: PAD.top, width: w, height: h, fill: "transparent", "pointer-events": "all" }));
  for (const v of [0, current.v]) {
    svg.append(svgEl("line", { x1: PAD.left, y1: y(v), x2: width - PAD.right, y2: y(v), class: "grid" }));
    svg.append(
      svgEl("text", { x: PAD.left - 8, y: y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, `${v.toFixed(0)}%`),
    );
  }
  // a numeral under each king while they fit, the current king always
  const labelled = [];
  for (const p of crowned) {
    if (labelled.length && x(p.i) - x(labelled.at(-1)) < 40) {
      if (p !== current) continue;
      labelled.pop();
    }
    labelled.push(p.i);
  }
  for (const i of labelled) {
    svg.append(svgEl("text", { x: x(i), y: HEIGHT - 8, class: "tick", "text-anchor": "middle" }, numeral(points[i])));
  }
  // square marks and steps on whole pixels: the chart's pixel look is in its shapes, not its type
  const px = (p) => [Math.round(x(p.i)), Math.round(y(p.v))];
  const mark = (p, size, cls) => {
    const [cx, cy] = px(p);
    const half = Math.floor(size / 2);
    return svgEl("rect", { x: cx - half, y: cy - half, width: size, height: size, class: cls });
  };
  const dots = points.map((p) =>
    p === current ? mark(p, 11, "king-dot current") : p.crowned ? mark(p, 7, "king-dot") : mark(p, 3, "king-dot lost"),
  );
  const steps = crowned.map((p, j) => (j ? `H${px(p)[0]}V${px(p)[1]}` : `M${px(p).join(",")}`));
  // the gain over genesis, as a faint gold under the staircase down to the genesis line
  svg.append(gradient("kings-fill"));
  svg.append(
    svgEl("path", {
      d: `${steps.join("")}H${Math.round(x(last))}V${Math.round(y(0))}H${px(crowned[0])[0]}Z`,
      class: "area",
      fill: "url(#kings-fill)",
    }),
  );
  for (const dot of dots.filter((_, i) => !points[i].crowned)) svg.append(dot);
  svg.append(svgEl("path", { d: `${steps.join("")}H${Math.round(x(last))}`, class: "king-line" }));
  for (const dot of dots.filter((_, i) => points[i].crowned)) svg.append(dot);

  const crosshair = svgEl("line", { y1: PAD.top, y2: HEIGHT - PAD.bottom, class: "crosshair", visibility: "hidden" });
  svg.append(crosshair);
  const tip = el("div", { class: "hero-chart-tip", hidden: true });
  let hover = -1;
  svg.addEventListener("pointermove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / Math.max(rect.width, 1)) * width;
    const i = Math.min(last, Math.max(0, Math.round(((px - PAD.left) / w) * last)));
    dots[hover]?.classList.remove("active");
    dots[i].classList.add("active");
    hover = i;
    const p = points[i];
    crosshair.setAttribute("x1", Math.round(x(i)) + 0.5);
    crosshair.setAttribute("x2", Math.round(x(i)) + 0.5);
    crosshair.removeAttribute("visibility");
    mount(
      tip,
      el(
        "div",
        { class: "hero-chart-tip-row" },
        el("span", { class: `swatch ${p.crowned ? "k-win" : "k-loss"}` }),
        el("b", {}, Number.isFinite(p.v) ? `${p.v.toFixed(1)}%` : "diverged"),
        el("span", { class: "lbl" }, p.crowned ? p.name : `lost to ${p.king}`),
      ),
      el(
        "div",
        { class: "hero-chart-tip-meta" },
        p.crowned
          ? `${p.step != null ? `KL ${p.step.toFixed(1)}% below ${p.beat.name} · ` : ""}crowned ${fmtRelative(p.crowned_at)}${p.uid != null ? ` · uid ${p.uid}` : ""}`
          : `KL ${vsKing(p)} · ${fmtRelative(p.finished_at)} · uid ${p.uid}`,
      ),
      p.evalId != null ? el("div", { class: "hero-chart-tip-meta" }, `click to open eval ${p.evalId}`) : null,
    );
    svg.style.cursor = p.evalId != null ? "pointer" : "";
    placeTip(tip, svg, width, x(i), PAD.top);
  });
  svg.addEventListener("click", (ev) => {
    const id = points[hover]?.evalId;
    if (id == null) return;
    const url = `eval.html?id=${id}`;
    if (ev.ctrlKey || ev.metaKey) window.open(url, "_blank", "noopener");
    else location.href = url;
  });
  svg.addEventListener("pointerleave", () => {
    dots[hover]?.classList.remove("active");
    hover = -1;
    crosshair.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
  mount(box, el("div", { class: "hero-chart-plot" }, svg, tip));
}
