"""The win rule: the challenger must beat the king on KL by the margin and may not be further from
the teacher on the hidden-state distance than the slack; a distance gain never buys a KL loss."""

from __future__ import annotations

import pytest

from eval_service import config
from eval_service.run import Distance, won

KING = Distance(kl=0.1, dist=0.5)
KL, SLACK = config.KL_MARGIN * KING.kl, config.DIST_SLACK * KING.dist


@pytest.mark.parametrize(
    ("challenger", "expected"),
    [
        (Distance(0.1 - 2 * KL, 0.5 - 0.1), True),
        (Distance(0.0, 0.0), True),
        (Distance(0.1 - 2 * KL, 0.5), True),
        (Distance(0.1 - 2 * KL, 0.5 + SLACK), True),
        (Distance(0.1 - 2 * KL, 0.5 + SLACK + 0.001), False),
        (Distance(0.1, 0.5 - 0.1), False),
        (Distance(0.0, 0.6), False),
        (Distance(0.2, 0.0), False),
        (Distance(0.1 - KL / 2, 0.0), False),
        (KING, False),
    ],
)
def test_win_rule(challenger: Distance, expected: bool):
    assert won(challenger, KING) is expected
