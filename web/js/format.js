// Text for hotkeys, times and durations, as the tables show them.

export const shortHotkey = (hk) => (!hk ? "—" : `${hk.slice(0, 6)}…${hk.slice(-4)}`);

export function fmtRelative(when) {
  if (!when) return "—";
  const s = Math.floor((Date.now() - new Date(when).getTime()) / 1000);
  if (s < 0) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  return h < 48 ? `${h}h ago` : `${Math.floor(h / 24)}d ago`;
}

// UTC, as mm/dd hh:mm
export function fmtDateTime(when) {
  if (!when) return "—";
  const d = new Date(when);
  const two = (n) => String(n).padStart(2, "0");
  return `${two(d.getUTCMonth() + 1)}/${two(d.getUTCDate())} ${two(d.getUTCHours())}:${two(d.getUTCMinutes())}`;
}

export function fmtDuration(ms) {
  if (ms == null || !Number.isFinite(ms) || ms < 0) return "—";
  const m = Math.round(ms / 60000);
  if (m < 60) return `${Math.max(1, m)}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return m % 60 ? `${h}h ${m % 60}m` : `${h}h`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
}
