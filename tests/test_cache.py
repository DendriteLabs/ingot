"""Which downloaded models the shared cache deletes: validation, dedup and eval sweep one folder by
one policy, so a model is never deleted while its submission waits or runs anywhere."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from common import states
from common.cache import POLICY, doomed

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _rows(*spec: tuple[str, str, int]) -> list[dict]:
    return [
        {"model_hash": h, "state": st, "updated_at": NOW - timedelta(minutes=m)}
        for h, st, m in spec
    ]


def test_a_model_in_the_pipeline_is_kept_however_long_it_waits():
    waiting = (
        states.SUBMITTED,
        states.VALIDATING,
        states.VALID,
        states.DEDUPING,
        states.UNIQUE,
        states.EVALUATING,
        states.WON,
        states.CROWNING,
    )
    rows = _rows(*((f"m{i}", st, 10_000) for i, st in enumerate(waiting)))
    assert doomed([r["model_hash"] for r in rows], rows, POLICY, set(), NOW) == set()


def test_finished_models_go_and_the_newest_rejected_duplicates_and_kings_stay():
    rows = _rows(
        ("i1", states.INVALID, 1),
        ("f2", states.FAILED, 2),
        ("i3", states.INVALID, 3),
        ("f4", states.FAILED, 4),
        *((f"d{i}", states.DUPLICATE, i) for i in range(5)),
        ("lost", states.LOST, 0),
        ("k1", states.CROWNED, 1),
        ("k2", states.CROWNED, 2),
        ("k3", states.CROWNED, 3),
    )
    present = [r["model_hash"] for r in rows] + ["unknown"]
    assert doomed(present, rows, POLICY, {"k1", "k2"}, NOW) == {"f4", "d3", "d4", "lost", "k3"}


def test_only_files_on_disk_count_towards_the_newest():
    rows = _rows(*((f"d{i}", states.DUPLICATE, i) for i in range(5)))
    assert doomed(["d0", "d3", "d4"], rows, POLICY, set(), NOW) == set()
