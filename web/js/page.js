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
    <button type="button" class="icon" id="theme-btn" aria-label="Switch to light theme" title="Switch to light theme">
      <svg width="14" height="14" viewBox="0 0 14 14" shape-rendering="crispEdges" aria-hidden="true">
        <rect x="1.5" y="1.5" width="11" height="11" fill="none" stroke="currentColor"/><rect x="2" y="2" width="5" height="10" fill="currentColor"/>
      </svg>
    </button>
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
  </nav>
</div>
`;

// the header into <header id="top">; dark is the default theme, a choice of light is remembered
// per browser and applied before paint by each page's head
export function mountHeader() {
  $("top").innerHTML = HEADER;
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
