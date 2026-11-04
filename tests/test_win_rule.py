"""The win rule: the challenger must beat the king on KL by the margin, a share of the king's KL."""

from __future__ import annotations

import pytest

from eval_service import config
from eval_service.run import Score, won

KING = Score(kl=0.1)
KL = config.KL_MARGIN * KING.kl


@pytest.mark.parametrize(
    ("challenger", "expected"),
    [
        (Score(0.1 - 2 * KL), True),
        (Score(0.0), True),
        (Score(0.1 - KL), True),
        (Score(0.1 - KL / 2), False),
        (Score(0.2), False),
        (KING, False),
    ],
)
def test_win_rule(challenger: Score, expected: bool):
    assert won(challenger, KING) is expected


def test_the_margin_is_three_percent():
    assert config.KL_MARGIN == 0.03
