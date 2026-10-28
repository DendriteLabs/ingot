// What every page shares: the header, theme switch, llms.txt copy, taomarketcap links and the
// two chart helpers.

import { link } from "./dom.js";

export const $ = (id) => document.getElementById(id);

export const svgEl = (tag, attrs = {}, text) => {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
};

// a chart tooltip beside chart x `at`, flipped to the left near the right edge
export function placeTip(tip, svg, width, at, top) {
  const rect = svg.getBoundingClientRect();
  const cssX = (at / width) * rect.width;
  tip.hidden = false;
  tip.style.left = `${Math.max(cssX + 12 + tip.offsetWidth > rect.width ? cssX - tip.offsetWidth - 12 : cssX + 12, 0)}px`;
  tip.style.top = `${top}px`;
}

// a uid or hotkey links to that hotkey's miner page on taomarketcap
export const minerLink = (netuid, hotkey, text) =>
  link(`https://taomarketcap.com/subnets/${netuid}/miners?query=${encodeURIComponent(hotkey)}`, text, { title: `${hotkey} on taomarketcap` });

const HEADER = `
<div class="row">
  <a class="brand" href="./" aria-label="Ingot">
    <img class="logo on-dark" src="brand/ingot-white.svg" alt="INGOT" width="134" height="24">
    <img class="logo on-light" src="brand/ingot-black.svg" alt="" width="134" height="24">
  </a>
  <nav class="links">
    <a class="page-link" href="./" data-page="index">Model</a>
    <a class="page-link" href="dashboard.html" data-page="dashboard">Dashboard</a>
    <button type="button" class="llms" id="hero-llms-btn" aria-label="Copy llms.txt to clipboard" title="Copy llms.txt">
      <span class="hero-llms-label">llms.txt</span>
      <svg width="11" height="11" viewBox="0 0 12 12" fill="none" aria-hidden="true">
        <rect x="3.5" y="3.5" width="7" height="7" stroke="currentColor" stroke-width="1.2"/>
        <path d="M2.5 8.5H1.5V1.5H8.5V2.5" stroke="currentColor" stroke-width="1.2"/>
      </svg>
    </button>
    <a class="icon" id="link-github" href="#" target="_blank" rel="noopener" aria-label="GitHub" title="GitHub">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.4 7.86 10.93.58.1.79-.25.79-.55v-1.96c-3.2.7-3.88-1.54-3.88-1.54-.52-1.32-1.27-1.68-1.27-1.68-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.03 1.76 2.7 1.25 3.36.96.1-.74.4-1.25.73-1.54-2.55-.29-5.24-1.27-5.24-5.66 0-1.25.45-2.28 1.18-3.08-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.18 1.18.92-.26 1.91-.39 2.89-.4.98.01 1.97.14 2.89.4 2.2-1.49 3.17-1.18 3.17-1.18.63 1.58.24 2.75.12 3.04.74.8 1.18 1.83 1.18 3.08 0 4.4-2.7 5.36-5.27 5.65.41.36.78 1.06.78 2.13v3.16c0 .31.21.66.79.55C20.21 21.4 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5z"/></svg>
    </a>
    <a class="icon social" href="https://discord.com/invite/bittensor" target="_blank" rel="noopener" aria-label="Discord" title="Bittensor Discord">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/></svg>
    </a>
    <a class="icon social" href="https://x.com/IngotSubnet/" target="_blank" rel="noopener" aria-label="X" title="@IngotSubnet on X">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
    </a>
    <button type="button" class="icon" id="theme-btn" aria-label="Switch to light theme" title="Switch to light theme">
      <svg width="14" height="14" viewBox="0 0 14 14" shape-rendering="crispEdges" aria-hidden="true">
        <rect x="1.5" y="1.5" width="11" height="11" fill="none" stroke="currentColor"/><rect x="2" y="2" width="5" height="10" fill="currentColor"/>
      </svg>
    </button>
  </nav>
</div>
`;

const FOOTER = `
<div class="row foot-cols">
  <div class="foot-brand">
    <img class="logo on-dark" src="brand/ingot-white.svg" alt="INGOT" width="112" height="20">
    <img class="logo on-light" src="brand/ingot-black.svg" alt="" width="112" height="20">
    <p>Open models, cast small.</p>
  </div>
  <nav aria-label="Model"><h4>Model</h4>
    <a href="./">The best model</a>
    <a href="https://huggingface.co/dendriteholdings" target="_blank" rel="noopener">Every king on Hugging Face</a>
    <a href="https://github.com/DendriteLabs/llama.cpp/tree/dendrite" target="_blank" rel="noopener">llama.cpp fork</a>
  </nav>
  <nav aria-label="Competition"><h4>Competition</h4>
    <a href="dashboard.html">Dashboard</a>
    <a href="llms.txt">llms.txt</a>
    <a id="foot-github" href="#" target="_blank" rel="noopener">Run a miner</a>
  </nav>
  <nav aria-label="Community"><h4>Community</h4>
    <a href="https://discord.com/invite/bittensor" target="_blank" rel="noopener">Discord</a>
    <a href="https://x.com/IngotSubnet/" target="_blank" rel="noopener">X</a>
    <a href="https://taomarketcap.com/subnets/97" target="_blank" rel="noopener">Subnet 97 on taomarketcap</a>
  </nav>
</div>
<div class="row foot-base"><span id="foot-netuid">Ingot</span><span>Refined. Assayed. Stamped.</span></div>
`;

// the header into <header id="top">; dark is the default theme, a choice of light is remembered
// per browser and applied before paint by each page's head
export function mountHeader() {
  $("top").innerHTML = HEADER;
  if ($("foot")) $("foot").innerHTML = FOOTER;
  // the GitHub links follow the state's, which the pages set on the header's
  new MutationObserver(() => { if ($("foot-github")) $("foot-github").href = $("link-github").href; })
    .observe($("link-github"), { attributes: true, attributeFilter: ["href"] });
  // the current page's link is underlined; an eval page belongs to the dashboard
  const page = (location.pathname.split("/").pop() || "index.html").replace(".html", "");
  for (const a of document.querySelectorAll(".page-link")) {
    if (a.dataset.page === (page === "eval" ? "dashboard" : page)) a.setAttribute("aria-current", "page");
  }
  const showTheme = () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    $("theme-btn").title = $("theme-btn").ariaLabel = `Switch to ${next} theme`;
  };
  $("theme-btn").addEventListener("click", () => {
    const light = document.documentElement.dataset.theme !== "light";
    if (light) document.documentElement.dataset.theme = "light";
    else delete document.documentElement.dataset.theme;
    try { localStorage.setItem("ingot-theme", light ? "light" : "dark"); } catch { /* private mode: the choice lasts this page view */ }
    showTheme();
  });
  showTheme();
  $("hero-llms-btn").addEventListener("click", async () => {
    const button = $("hero-llms-btn");
    const label = button.querySelector(".hero-llms-label");
    button.disabled = true;
    try {
      await navigator.clipboard.writeText(await (await fetch("llms.txt", { cache: "no-store" })).text());
      button.classList.add("copied");
      label.textContent = "copied";
      setTimeout(() => { button.classList.remove("copied"); label.textContent = "llms.txt"; }, 1600);
    } catch {
      window.open("llms.txt", "_blank", "noopener");
    } finally {
      button.disabled = false;
    }
  });
}
