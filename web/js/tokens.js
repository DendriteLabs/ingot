// The token section of the model page: /data/kl-demo.json, the teacher's answers to held-out
// Terminal-Bench 2.1 and SWE-bench Multilingual steps, read by each model token by token.

import { el, mount } from "./dom.js";

const MODELS = {
  xxvii: { label: "King XXVII", note: "5.95 GB", tone: "king" },
  genesis: { label: "Bonsai 2", note: "5.95 GB", tone: "other" },
};
const kingOf = (models) => models.find((m) => MODELS[m].tone === "king");
const SPECIAL = new Set(["<think>", "</think>", "<tool_call>", "</tool_call>"]);
// a token's shade on a log scale: KL 0.03 or less none, 1 or more full
const FLOOR = Math.log10(0.03);
const shade = (kl) => Math.min(1, Math.max(0, (Math.log10(Math.max(kl, 1e-6)) - FLOOR) / -FLOOR));
const visible = (piece) => piece.replaceAll(" ", "␣").replaceAll("\n", "↵").replaceAll("\t", "⇥");
const pct = (p) => (p >= 0.995 ? "100%" : p < 0.001 ? "<0.1%" : `${(100 * p).toFixed(p < 0.1 ? 1 : 0)}%`);

let data = null;
let view = { passage: 0, model: null, at: null };

export async function renderTokens() {
  try {
    const res = await fetch("/data/kl-demo.json");
    if (!res.ok) return;
    data = await res.json();
  } catch (err) {
    console.warn("token demo fetch failed", err);
    return;
  }
  document.getElementById("tokens").hidden = false;
  document.getElementById("tok-demo-note").textContent =
    `${data.passages.length} steps · ${data.models.map((m) => MODELS[m].label).join(" and ")} on the same answers`;
  renderSummary();
  draw();
}

// per domain each model's mean KL per token over the steps, both read on the same answers
function renderSummary() {
  const king = kingOf(data.models);
  mount(document.getElementById("tokens-figures"), ...data.summary.map((row) => {
    const top = Math.max(...data.models.map((m) => row.kl[m]));
    const best = row.kl[king];
    return el("figure", {},
      el("figcaption", {}, el("h3", {}, row.domain), el("p", {}, `${row.n} steps · KL to the teacher per token, lower is closer`)),
      el("div", { class: "kl-rows" }, ...data.models.map((m) => el("div", { class: `size-row ${MODELS[m].tone}` },
        el("span", { class: "size-label" }, MODELS[m].label, el("small", {}, m === king ? MODELS[m].note : `${MODELS[m].note} · +${(100 * (row.kl[m] / best - 1)).toFixed(0)}%`)),
        el("span", { class: "size-track" }, el("i", { style: `width: ${(100 * row.kl[m]) / top}%` })),
        el("span", { class: "size-value" }, row.kl[m].toFixed(3))))));
  }));
}

function draw() {
  const p = data.passages[view.passage];
  view.model ??= kingOf(data.models);
  const model = p.models[view.model];
  if (view.at == null) view.at = model.kl.indexOf(Math.max(...model.kl));
  const piece = (id) => data.vocab[id] ?? "";

  mount(document.getElementById("tokens-picker"),
    el("div", { class: "tok-pick" }, ...data.passages.map((q, i) => el("button", {
      class: i === view.passage ? "on" : null, title: q.domain, "aria-pressed": String(i === view.passage),
    }, `${q.domain.startsWith("Terminal") ? "TB" : "SWE"} ${1 + data.passages.slice(0, i).filter((r) => r.domain === q.domain).length}`))),
    el("div", { class: "tok-models" }, ...data.models.map((m) => el("button", {
      class: `${m === view.model ? "on" : ""} ${MODELS[m].tone}`, "data-model": m, "aria-pressed": String(m === view.model),
    }, MODELS[m].label, el("small", { title: "mean KL per token in this passage" }, `KL ${p.models[m].kl_window.toFixed(3)}`)))));
  document.querySelectorAll("#tokens-picker .tok-pick button").forEach((b, i) => b.addEventListener("click", () => { view = { ...view, passage: i, at: null }; draw(); }));
  document.querySelectorAll("#tokens-picker .tok-models button").forEach((b) => b.addEventListener("click", () => { view = { ...view, model: b.dataset.model }; draw(); }));

  mount(document.getElementById("tokens-context"),
    el("div", { class: "tok-ctx-head" }, `${p.domain} · ${p.prompt_tokens.toLocaleString("en-US")} tokens of context · `,
      `tokens ${p.window[0] + 1}–${p.window[1]} of the teacher's ${p.answer_tokens.toLocaleString("en-US")}-token answer`),
    el("details", {}, el("summary", {}, p.last.role === "tool" ? "The output it answers" : "The message it answers"), el("pre", {}, p.last.content)));

  const stream = document.getElementById("tokens-stream");
  mount(stream, ...p.tokens.map((id, i) => {
    const text = piece(id);
    const special = SPECIAL.has(text);
    return el("span", {
      class: `tok${special ? " mark" : ""}${i === view.at ? " at" : ""}`, "data-i": String(i),
      style: `--k: ${shade(model.kl[i]).toFixed(2)}`,
    }, special ? text : text.includes("\n") ? text.replaceAll("\n", "↵\n") : text);
  }));
  stream.onmouseover = (e) => { const i = e.target.dataset?.i; if (i != null) inspect(+i); };
  stream.onclick = stream.onmouseover;
  stream.onkeydown = (e) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (!step) return;
    e.preventDefault();
    inspect(Math.min(p.tokens.length - 1, Math.max(0, view.at + step)));
  };
  inspect(view.at);
}

function inspect(i) {
  const p = data.passages[view.passage];
  const actual = p.tokens[i];
  view.at = i;
  document.querySelectorAll("#tokens-stream .tok.at").forEach((n) => n.classList.remove("at"));
  document.querySelector(`#tokens-stream .tok[data-i="${i}"]`)?.classList.add("at");
  const column = (title, o, tone) => el("div", { class: `tok-col ${tone}` }, el("h4", {}, title),
    ...o.top[i].map((id, j) => el("div", { class: `tok-row${id === actual ? " actual" : ""}` },
      el("code", {}, visible(data.vocab[id] ?? "")),
      el("span", { class: "tok-bar" }, el("i", { style: `width: ${100 * o.top_p[i][j]}%` })),
      el("span", { class: "tok-p" }, pct(o.top_p[i][j])))));
  mount(document.getElementById("tokens-inspect"),
    el("div", { class: "tok-head" }, "The teacher wrote ", el("code", {}, visible(data.vocab[actual] ?? "")),
      el("span", {}, `the teacher gave it ${pct(p.teacher.p[i])}`),
      ...data.models.map((m) => el("span", { class: MODELS[m].tone }, `${MODELS[m].label} ${pct(p.models[m].p[i])} · KL ${p.models[m].kl[i].toFixed(3)}`))),
    el("div", { class: "tok-cols" }, column("Teacher", p.teacher, "teacher"),
      ...data.models.map((m) => column(MODELS[m].label, p.models[m], MODELS[m].tone))));
}
