"""The eval's score: within a domain every answer token weighs the same, and the domains weigh by
their shares of the samples."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from eval_service.run import by_domain


@dataclass
class Sums:
    kl: float
    predictions: int


def test_tokens_of_one_domain_weigh_the_same():
    long = Sums(kl=0.2 * 4000, predictions=4000)
    short = Sums(kl=0.1 * 1000, predictions=1000)
    assert abs(by_domain([("swe", long), ("swe", short)]).kl - 0.18) < 1e-12


def test_domains_weigh_by_their_sample_counts():
    swe = [("swe", Sums(kl=0.2 * 4000, predictions=4000))] * 3
    math = [("math", Sums(kl=0.1 * 10, predictions=10))]
    assert abs(by_domain(swe + math).kl - (0.75 * 0.2 + 0.25 * 0.1)) < 1e-12


def test_a_one_token_answer_counts():
    assert by_domain([("math", Sums(kl=0.3, predictions=1))]).kl == 0.3


def test_the_dashboard_rescores_records_as_the_eval_scores_sums():
    from dashboard.state import token_weighted

    def record(domain, n, king, chal):
        runs = {"king": {"kl": king}, "challenger": {"kl": chal}}
        return {"domain": domain, "answer_tokens": n, "runs": runs}

    scored = [
        record("swe", 4000, 0.2, 0.1),
        record("swe", 1000, 0.1, 0.3),
        record("math", 10, 0.5, 0.4),
    ]
    diverged = record("math", 10, float("inf"), 0.4)

    def sums(role):
        return [
            (r["domain"], Sums(r["runs"][role]["kl"] * r["answer_tokens"], r["answer_tokens"]))
            for r in scored
        ]

    king, chal = by_domain(sums("king")), by_domain(sums("challenger"))
    assert token_weighted([*scored, diverged]) == pytest.approx((king.kl, chal.kl))
    assert token_weighted([diverged]) is None
