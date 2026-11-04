"""The eval's score averages per-sample means: a long answer weighs no more than a short one."""

from __future__ import annotations

from dataclasses import dataclass

from eval_service.run import per_sample


@dataclass
class Sums:
    kl: float
    predictions: int


def test_per_sample_means_weigh_every_sample_once():
    long = Sums(kl=0.2 * 4096, predictions=4096)
    short = Sums(kl=0.1 * 100, predictions=100)
    assert abs(per_sample([long, short]).kl - 0.15) < 1e-12


def test_a_one_token_answer_counts():
    assert per_sample([Sums(kl=0.3, predictions=1)]).kl == 0.3
