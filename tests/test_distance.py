"""The eval's score averages per-sample means: a long answer weighs no more than a short one."""

from __future__ import annotations

from dataclasses import dataclass

from eval_service.run import per_sample


@dataclass
class Sums:
    kl: float
    predictions: int
    layer_mean_dist: float
    activations: int


def test_per_sample_means_weigh_every_sample_once():
    long = Sums(kl=0.2 * 4096, predictions=4096, layer_mean_dist=0.3 * 4095, activations=4095)
    short = Sums(kl=0.1 * 100, predictions=100, layer_mean_dist=0.5 * 99, activations=99)
    d = per_sample([long, short])
    assert abs(d.kl - 0.15) < 1e-12 and abs(d.dist - 0.4) < 1e-12


def test_a_one_token_answer_has_no_activations():
    d = per_sample([Sums(kl=0.3, predictions=1, layer_mean_dist=0.0, activations=0)])
    assert d.kl == 0.3 and d.dist == 0.0
