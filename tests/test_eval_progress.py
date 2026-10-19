"""A running eval reports samples_done every 1/PROGRESS_CHUNKS of its samples, ending at the total.
test_pipeline checks that the dashboard's queue shows it."""

from __future__ import annotations

import threading

from eval_service import config
from eval_service.run import _Evaluation


def counter(total: int) -> tuple[_Evaluation, list[int]]:
    reported: list[int] = []
    evaluation = object.__new__(_Evaluation)  # no servers or GPUs: only the counting
    evaluation.total, evaluation.done, evaluation.lock = total, 0, threading.Lock()
    evaluation.on_progress = reported.append
    return evaluation, reported


def test_progress_is_reported_once_per_chunk_and_at_the_end():
    evaluation, reported = counter(1080)
    for _ in range(1080):
        evaluation._count()
    assert reported == [54 * k for k in range(1, config.PROGRESS_CHUNKS + 1)]


def test_fewer_samples_than_chunks_report_every_sample():
    evaluation, reported = counter(7)
    for _ in range(7):
        evaluation._count()
    assert reported == list(range(1, 8))
