"""The win rule when a model diverged: a non-finite score never wins, and a diverged king cannot
hold its crown against a challenger that did not diverge."""

from __future__ import annotations

import math

from eval_service.run import DIVERGED, Distance, won

KING = Distance(kl=0.10, dist=0.40)
WORSE = Distance(kl=0.20, dist=0.50)


def test_a_diverged_challenger_loses():
    assert not won(DIVERGED, KING)
    assert not won(Distance(kl=math.nan, dist=0.1), KING)


def test_a_diverged_king_loses_to_any_challenger_that_did_not():
    assert won(WORSE, DIVERGED)
    assert won(WORSE, Distance(kl=0.1, dist=math.nan))
    assert not won(DIVERGED, DIVERGED)
