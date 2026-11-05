"""A group admits requests in line order, forced reads first: a long request is not overtaken by
short ones that would fit sooner. A request booked below its whole size books the rest once it
is due for it. A prefetch (the next eval's) takes only a server the running eval has left."""

from __future__ import annotations

import threading
import time

from eval_service.servers import Admission

CAPS = [100]


def queue(
    admission: Admission,
    admitted: list[str],
    name: str,
    need: int,
    *,
    first: bool = False,
    prefetch: bool = False,
):
    """Start a request that waits for its turn; returns once it is in line."""
    lines = len(admission.line) + len(admission.prefetch_line)

    def run() -> None:
        admission.enter(need, CAPS, first, prefetch)
        admitted.append(name)

    threading.Thread(target=run, daemon=True).start()
    while len(admission.line) + len(admission.prefetch_line) == lines and name not in admitted:
        time.sleep(0.001)


def settle() -> None:
    time.sleep(0.05)


def test_a_long_request_is_not_overtaken_by_a_short_one():
    admission, admitted = Admission(1), []
    held = admission.enter(60, CAPS, first=False, prefetch=False)
    queue(admission, admitted, "long", 80)
    queue(admission, admitted, "short", 30)
    settle()
    assert admitted == [], "the short one fits now, but waits behind the long one"

    admission.leave(held)
    settle()
    assert admitted == ["long"], "80 + 30 do not fit together"
    admission.leave(next(iter(admission.admitted)))
    settle()
    assert admitted == ["long", "short"]


def test_a_forced_read_goes_ahead_of_new_answers():
    admission, admitted = Admission(1), []
    held = admission.enter(100, CAPS, first=False, prefetch=False)
    queue(admission, admitted, "answer", 50)
    queue(admission, admitted, "reread", 50, first=True)
    admission.leave(held)
    settle()
    assert admitted == ["reread", "answer"]
    assert admission.line == [] and admission.used == [100]


def test_a_request_books_the_rest_once_due():
    admission, admitted = Admission(1), []
    held = admission.enter(40, CAPS, first=False, prefetch=False, grow=50, grow_after_s=10.0)
    queue(admission, admitted, "next", 50)
    settle()
    assert admitted == ["next"], "40 + 50 fit"
    admission.rebook(now=time.monotonic() + 11)
    assert admission.used == [140] and held.need == 90 and held.grow == 0
    admission.leave(held)
    assert admission.used == [50]


def test_a_prefetch_takes_only_a_server_the_eval_has_left():
    admission, admitted = Admission(1), []
    held = admission.enter(10, CAPS, first=False, prefetch=False)
    queue(admission, admitted, "ahead", 10, prefetch=True)
    settle()
    assert admitted == [], "room enough, but the eval still generates there"
    admission.leave(held)
    settle()
    assert admitted == ["ahead"] and admission.ahead == [1] and admission.running == [0]
    admission.promote()
    assert admission.running == [1] and admission.ahead == [0], "the eval's own request now"
    queue(admission, admitted, "next ahead", 10, prefetch=True)
    settle()
    assert admitted == ["ahead"], "the next eval's prefetch waits for it like for any other"
