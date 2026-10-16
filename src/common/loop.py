"""The loop every service runs."""

from __future__ import annotations

import signal
import sys
import time
from collections.abc import Callable

from loguru import logger as log


def forever(
    step: Callable[[], bool], *, idle_s: float, on_stop: Callable[[], None] | None = None
) -> None:
    """Call `step` forever, sleeping `idle_s` when it finds nothing; `on_stop` runs once on exit."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        while True:
            try:
                busy = step()
            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("step failed; retrying in {}s", idle_s)
                busy = False
            if not busy:
                time.sleep(idle_s)
    finally:
        if on_stop is not None:
            on_stop()
