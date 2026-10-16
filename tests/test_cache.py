"""Which downloaded models each box deletes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from common import states
from common.cache import doomed
from eval_service.main import POLICY as EVAL
from model_dedup.main import POLICY as DEDUP
from model_validation.main import POLICY as VALIDATION

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _rows(*spec: tuple[str, str, int]) -> list[dict]:
    return [
        {"model_hash": h, "state": st, "updated_at": NOW - timedelta(minutes=m)}
        for h, st, m in spec
    ]


@pytest.mark.parametrize(
    ("policy", "rows", "keep", "expected"),
    [
        (
            VALIDATION,
            _rows(
                ("a", states.INVALID, 1),
                ("b", states.FAILED, 2),
                ("c", states.INVALID, 3),
                ("d", states.INVALID, 4),
                ("e", states.VALID, 500),
                ("f", states.VALIDATING, 500),
                ("g", states.UNIQUE, 500),
            ),
            set(),
            {"d"},
        ),
        (
            DEDUP,
            _rows(
                *((f"d{i}", states.DUPLICATE, i) for i in range(5)),
                ("u1", states.UNIQUE, 14),
                ("u2", states.UNIQUE, 15),
                ("l", states.LOST, 20),
                ("v", states.VALID, 500),
                ("x", states.INVALID, 500),
            ),
            set(),
            {"d3", "d4", "u2", "l"},
        ),
        (
            EVAL,
            _rows(
                *((f"f{i}", states.FAILED, i) for i in range(4)),
                ("lost", states.LOST, 0),
                ("won", states.WON, 9),
                ("ev", states.EVALUATING, 9),
                ("k1", states.CROWNED, 1),
                ("k2", states.CROWNED, 2),
                ("k3", states.CROWNED, 3),
            ),
            {"k1", "k2"},
            {"f3", "lost", "k3"},
        ),
    ],
)
def test_doomed(policy, rows, keep, expected):
    present = [r["model_hash"] for r in rows] + ["unknown"]
    assert doomed(present, rows, policy, keep, NOW) == expected


def test_only_files_on_disk_count_towards_the_newest():
    rows = _rows(*((f"d{i}", states.DUPLICATE, i) for i in range(5)))
    assert doomed(["d0", "d3", "d4"], rows, DEDUP, set(), NOW) == set()
