"""A group admits requests in line order, forced reads first: a long request is not overtaken by
short ones that would fit sooner, which once left the long prompts running at the end of an eval."""

from __future__ import annotations

import threading
import time

from eval_service.servers import Admission

CAPS = [100]


def queue(admission: Admission, admitted: list[str], name: str, need: int, *, first: bool = False):
    """Start a request that waits for its turn; returns once it is in line."""
    in_line = len(admission.line)

    def run() -> None:
        admission.enter(need, CAPS, first)
        admitted.append(name)

    threading.Thread(target=run, daemon=True).start()
    while len(admission.line) == in_line and name not in admitted:
        time.sleep(0.001)


def settle() -> None:
    time.sleep(0.05)


def test_a_long_request_is_not_overtaken_by_a_short_one():
    admission, admitted = Admission(1), []
    held = admission.enter(60, CAPS, first=False)
    queue(admission, admitted, "long", 80)
    queue(admission, admitted, "short", 30)
    settle()
    assert admitted == [], "the short one fits now, but waits behind the long one"

    admission.leave(held, 60)
    settle()
    assert admitted == ["long"], "80 + 30 do not fit together"
    admission.leave(0, 80)
    settle()
    assert admitted == ["long", "short"]


def test_a_forced_read_goes_ahead_of_new_answers():
    admission, admitted = Admission(1), []
    held = admission.enter(100, CAPS, first=False)
    queue(admission, admitted, "answer", 50)
    queue(admission, admitted, "reread", 50, first=True)
    admission.leave(held, 100)
    settle()
    assert admitted == ["reread", "answer"]
    assert admission.line == [] and admission.used == [100]
