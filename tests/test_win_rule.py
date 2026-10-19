"""The win rule: the challenger must beat the king on KL and on the hidden-state distance, each by
its own margin; a big gain on one never buys a loss on the other."""

from __future__ import annotations

import pytest

from eval_service import config
from eval_service.run import Distance, won

KING = Distance(kl=0.1, dist=0.5)
KL, DIST = config.KL_MARGIN * KING.kl, config.DIST_MARGIN * KING.dist


@pytest.mark.parametrize(
    ("challenger", "expected"),
    [
        (Distance(0.1 - 2 * KL, 0.5 - 2 * DIST), True),
        (Distance(0.0, 0.0), True),
        (Distance(0.1 - 2 * KL, 0.5), False),
        (Distance(0.1, 0.5 - 2 * DIST), False),
        (Distance(0.0, 0.6), False),
        (Distance(0.2, 0.0), False),
        (Distance(0.1 - KL / 2, 0.0), False),
        (Distance(0.0, 0.5 - DIST / 2), False),
        (KING, False),
    ],
)
def test_win_rule(challenger: Distance, expected: bool):
    assert won(challenger, KING) is expected
