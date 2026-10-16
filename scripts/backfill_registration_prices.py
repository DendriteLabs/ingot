"""Fill registration_prices with the last day of registration burns, read from an archive node.

    uv run scripts/backfill_registration_prices.py

The chain reader samples the burn every PRICE_EVERY_BLOCKS from now on, but a normal node keeps
only a few hundred blocks of state, so the day before it started has to come from an archive
node. Blocks already in the table are skipped, so a rerun only fills gaps.
"""

from __future__ import annotations

import argparse

from chain_reader import config
from common import chain, db
from common.config import settings

BLOCKS_PER_DAY = 7200


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--network", default="archive")
    args = parser.parse_args()

    s = settings()
    conn = db.connect()
    subtensor = chain.connect(args.network)
    head = chain.current_block(subtensor)
    have = {r["block"] for r in conn.execute("SELECT block FROM registration_prices")}
    todo = [
        b for b in range(head, head - BLOCKS_PER_DAY, -config.PRICE_EVERY_BLOCKS) if b not in have
    ]
    print(f"netuid {s.netuid}: {len(todo)} blocks to read")
    for done, block in enumerate(todo, start=1):
        burn = chain.registration_burn_rao(subtensor, s.netuid, block)
        if burn is not None:
            conn.execute(
                "INSERT INTO registration_prices (block, burn_rao, at) VALUES (%s, %s, %s)"
                " ON CONFLICT DO NOTHING",
                (block, burn, chain.block_time(subtensor, block)),
            )
        if done % 50 == 0 or done == len(todo):
            print(f"{done}/{len(todo)} (block {block})", flush=True)


if __name__ == "__main__":
    main()
