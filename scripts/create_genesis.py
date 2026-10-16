"""Insert the genesis king so the first challenger has something to beat.

    uv run scripts/create_genesis.py

Reads the genesis repo from chain.toml. Inserts genesis into an empty kings table, or points the
existing genesis row at it while no challenger has been crowned.
"""

from __future__ import annotations

from common import db
from common.config import chain_spec


def main() -> None:
    spec = chain_spec()
    uri = f"{spec.genesis_repo}@{spec.genesis_revision}"
    conn = db.connect()
    if conn.execute("SELECT 1 FROM kings WHERE submission_id IS NOT NULL LIMIT 1").fetchone():
        print("a challenger has been crowned; genesis is history, nothing to do")
        return
    if conn.execute(
        "UPDATE kings SET model_uri = %s, hf_repo = %s WHERE submission_id IS NULL RETURNING id",
        (uri, spec.genesis_repo),
    ).fetchone():
        print(f"genesis king updated: {uri}")
        return
    conn.execute("INSERT INTO kings (model_uri, hf_repo) VALUES (%s, %s)", (uri, spec.genesis_repo))
    print(f"genesis king: {uri}")


if __name__ == "__main__":
    main()
