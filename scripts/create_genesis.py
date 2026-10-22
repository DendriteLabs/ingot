"""Insert a quant's genesis king so its first challenger has something to beat.

    uv run scripts/create_genesis.py [--quant q2]

Reads the pair (the open one by default) and its genesis from chain.toml. Inserts that genesis,
or points the pair's existing genesis row at it while no challenger of the pair has been crowned.
"""

from __future__ import annotations

import argparse

from common import db, models
from common.config import chain_spec


def main() -> None:
    spec = chain_spec()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", default=spec.quant)
    quant = spec.pair(spec.family, parser.parse_args().quant)
    pair = (quant.family, quant.name)
    uri = models.genesis_uri(quant)
    conn = db.connect()
    if conn.execute(
        "SELECT 1 FROM kings WHERE submission_id IS NOT NULL AND family = %s AND quant = %s",
        pair,
    ).fetchone():
        print(f"a {quant.name} challenger has been crowned; genesis is history, nothing to do")
        return
    if conn.execute(
        "UPDATE kings SET model_uri = %s, hf_repo = %s"
        " WHERE submission_id IS NULL AND family = %s AND quant = %s RETURNING id",
        (uri, quant.genesis_repo, *pair),
    ).fetchone():
        print(f"{quant.name} genesis king updated: {uri}")
        return
    conn.execute(
        "INSERT INTO kings (model_uri, hf_repo, family, quant) VALUES (%s, %s, %s, %s)",
        (uri, quant.genesis_repo, *pair),
    )
    print(f"{quant.name} genesis king: {uri}")


if __name__ == "__main__":
    main()
