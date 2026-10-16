"""The incentive rule: newest five kings, 20% each, for five days; the rest burns."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from common.reign import King, name, weights

NOW = datetime(2026, 10, 14, tzinfo=UTC)


def king(uid: int, days_ago: float) -> King:
    return King(uid=uid, hotkey=f"hk-{uid}", crowned_at=NOW - timedelta(days=days_ago))


def registered(*uids: int) -> dict[int, str]:
    return {uid: f"hk-{uid}" for uid in uids}


def pay(kings: Sequence[King], hotkeys: dict[int, str], *, share: float = 0.2) -> dict[int, float]:
    return weights(kings, NOW, hotkeys, slots=5, reign_days=5.0, share=share, burn_uid=0)


def test_one_king_gets_a_fifth_and_the_rest_burns():
    assert pay([king(7, 1)], registered(7)) == {7: 0.2, 0: 0.8}


def test_five_fresh_kings_split_everything():
    kings = [king(uid, days_ago=uid * 0.5) for uid in range(1, 6)]
    assert pay(kings, registered(1, 2, 3, 4, 5)) == {uid: 0.2 for uid in range(1, 6)}


def test_a_sixth_king_pushes_the_oldest_out():
    kings = [king(uid, days_ago=uid * 0.5) for uid in range(1, 7)]
    paid = pay(kings, registered(*range(1, 7)))
    assert 6 not in paid and paid[1] == 0.2 and 0 not in paid


def test_a_reign_ends_after_five_days_even_in_the_top_five():
    kings = [king(1, days_ago=5.5), king(2, days_ago=1)]
    assert pay(kings, registered(1, 2)) == {2: 0.2, 0: 0.8}


def test_genesis_is_never_paid():
    genesis = King(uid=None, hotkey=None, crowned_at=NOW - timedelta(days=1))
    assert pay([genesis, king(3, 1)], registered(3)) == {3: 0.2, 0: 0.8}


def test_a_deregistered_king_burns_its_share():
    assert pay([king(1, 1), king(2, 1)], {1: "hk-1", 2: "someone-else"}) == {1: 0.2, 0: 0.8}


def test_shares_beyond_the_emission_are_refused():
    with pytest.raises(ValueError):
        pay([], {}, share=0.3)


def test_king_names():
    assert [name(n, "27B-FP8") for n in (None, 1, 4, 9, 14, 129)] == [
        "GENESIS",
        "Ingot-I-27B-FP8",
        "Ingot-IV-27B-FP8",
        "Ingot-IX-27B-FP8",
        "Ingot-XIV-27B-FP8",
        "Ingot-CXXIX-27B-FP8",
    ]
    assert name(2, "35B-FP4") == "Ingot-II-35B-FP4"
