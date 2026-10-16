"""The chain reader's registration price samples: spaced by PRICE_EVERY_BLOCKS whatever the call
rate, spacing kept across a restart, and nothing written when the chain has no burn to report."""

from __future__ import annotations

from types import SimpleNamespace

from chain_reader import config, prices
from common import db


class FakeSubtensor:
    def __init__(self, burns: dict[int, int | None]) -> None:
        self.burns = burns

    def recycle(self, netuid: int, block: int | None = None) -> SimpleNamespace | None:
        rao = self.burns.get(block)
        return None if rao is None else SimpleNamespace(rao=rao)


def sampled(conn) -> list[tuple[int, int]]:
    rows = conn.execute("SELECT block, burn_rao FROM registration_prices ORDER BY block")
    return [(r["block"], r["burn_rao"]) for r in rows]


def test_one_sample_every_price_interval(conn):
    every = config.PRICE_EVERY_BLOCKS
    chain = FakeSubtensor({b: 90_000_000 + b for b in range(1000, 1000 + 3 * every + 1)})
    for block in range(1000, 1000 + 3 * every + 1):
        prices.sample(conn, chain, 97, block)
    assert sampled(conn) == [(1000 + i * every, 90_001_000 + i * every) for i in range(4)]

    # a restarted reader picks the spacing up from the table
    assert not prices.sample(conn, chain, 97, 1000 + 3 * every + 1)


def test_no_burn_no_row(conn):
    assert not prices.sample(conn, FakeSubtensor({}), 97, 5000)
    assert db.one(conn, "SELECT count(*) AS n FROM registration_prices")["n"] == 0
