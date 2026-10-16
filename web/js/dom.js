// The three DOM helpers the page uses: build an element, replace a node's children, an outbound link.

export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  append(node, children);
  return node;
}

export function mount(node, ...children) {
  node.replaceChildren();
  append(node, children);
}

export function link(href, text, props = {}) {
  return el("a", { href, target: "_blank", rel: "noopener", ...props }, text);
}

function append(node, children) {
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
}
