"""Where an eval's time goes: seconds per stage over the eval's samples, and how many samples are in
each stage right now. Only the log reads it; it measures and never changes what it measures."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Generator


class Stages:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.longest: dict[str, float] = {}
        self.active: dict[str, int] = {}  # also every stage name, in the order first entered
        self.amounts: dict[str, tuple[float, int]] = {}
        self.samples: list[tuple[float, str, int, int]] = []

    def reset(self) -> None:
        with self.lock:
            self.started = time.monotonic()
            for d in (self.seconds, self.calls, self.longest, self.active, self.amounts):
                d.clear()
            self.samples.clear()

    @contextlib.contextmanager
    def timed(self, name: str) -> Generator[None]:
        start = time.monotonic()
        with self.lock:
            self.active[name] = self.active.get(name, 0) + 1
        try:
            yield
        finally:
            took = time.monotonic() - start
            with self.lock:
                self.active[name] = max(self.active.get(name, 1) - 1, 0)
                self.seconds[name] = self.seconds.get(name, 0.0) + took
                self.calls[name] = self.calls.get(name, 0) + 1
                self.longest[name] = max(self.longest.get(name, 0.0), took)

    def add(self, name: str, amount: float) -> None:
        with self.lock:
            total, n = self.amounts.get(name, (0.0, 0))
            self.amounts[name] = (total + amount, n + 1)

    def sample(
        self, sample_id: str, seconds: float, prompt_tokens: int, answer_tokens: int
    ) -> None:
        with self.lock:
            self.samples.append((seconds, sample_id, prompt_tokens, answer_tokens))

    def done(self) -> int:
        with self.lock:
            return len(self.samples)

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def line(self) -> str:
        """Per stage: mean and longest seconds of the calls that left it, and how many are in it."""
        with self.lock:
            parts = []
            for name, now in self.active.items():
                calls = self.calls.get(name, 0)
                timed = (
                    f" {self.seconds[name] / calls:.1f}s mean, {self.longest[name]:.0f}s max"
                    if calls
                    else ""
                )
                parts.append(f"{name}{timed} ({now} in it)")
            parts += [f"{name} {t:.0f} ({t / n:.0f} each)" for name, (t, n) in self.amounts.items()]
        return "; ".join(parts) or "nothing timed yet"

    def slowest(self, n: int) -> str:
        with self.lock:
            worst = sorted(self.samples, reverse=True)[:n]
        return "; ".join(
            f"{sid} {s:.0f}s ({p} prompt + {a} answer tokens)" for s, sid, p, a in worst
        )


# One eval runs at a time; run() resets it at the start of each.
stages = Stages()
