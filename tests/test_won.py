"""The win rule when a model diverged: a non-finite score never wins, and a diverged king cannot
hold its crown against a challenger that did not diverge."""

from __future__ import annotations

import math

from eval_service.run import DIVERGED, Score, won

KING = Score(kl=0.10)
WORSE = Score(kl=0.20)


def test_a_diverged_challenger_loses():
    assert not won(DIVERGED, KING)
    assert not won(Score(kl=math.nan), KING)


def test_a_diverged_king_loses_to_any_challenger_that_did_not():
    assert won(WORSE, DIVERGED)
    assert won(WORSE, Score(kl=math.nan))
    assert not won(DIVERGED, DIVERGED)
