"""Registration price samples for the dashboard: one row every PRICE_EVERY_BLOCKS. The last
sampled block comes from the table, so a restart neither doubles nor skips a sample."""

from __future__ import annotations

from typing import Any

from chain_reader import config
from common import chain, db


def sample(conn: db.Conn, subtensor: Any, netuid: int, block: int) -> bool:
    """Record the registration burn at `block` if the last sample is far enough back."""
    last = db.one(conn, "SELECT max(block) AS block FROM registration_prices")["block"]
    if last is not None and block - last < config.PRICE_EVERY_BLOCKS:
        return False
    burn = chain.registration_burn_rao(subtensor, netuid, block)
    if burn is None:
        return False
    conn.execute(
        "INSERT INTO registration_prices (block, burn_rao) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (block, burn),
    )
    return True
