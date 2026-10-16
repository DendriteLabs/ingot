"""The kings row lands first, so the next eval already faces the new king while the upload
runs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from loguru import logger as log

from common import db, models, reign, states
from common.config import ChainSpec, Settings, chain_spec, settings
from common.loop import forever
from promoter import config

Fetch = Callable[[str, Path], Path]
Publish = Callable[[Path, str, str], str]


def run_once(
    conn: db.Conn,
    *,
    worker: str,
    s: Settings,
    spec: ChainSpec,
    fetch: Fetch = models.fetch,
    publish: Publish = models.publish,
) -> bool:
    db.heartbeat(conn, worker, queued=states.WON)
    row = db.claim(
        conn, worker=worker, queued=states.WON, running=states.CROWNING, lease_s=config.LEASE_S
    )
    if row is None:
        return False
    try:
        king = db.one(
            conn,
            """
            INSERT INTO kings (submission_id, hotkey, uid, model_uri) VALUES (%s, %s, %s, %s)
            ON CONFLICT (submission_id) DO UPDATE SET hotkey = EXCLUDED.hotkey
            RETURNING id, hf_repo
            """,
            (row["id"], row["hotkey"], row["uid"], row["model_uri"]),
        )
        ordinal = db.one(
            conn,
            "SELECT count(*) AS n FROM kings WHERE submission_id IS NOT NULL AND id <= %s",
            (king["id"],),
        )["n"]
        title = reign.name(ordinal, spec.king_name(row["family"], row["quant"]))
        if not king["hf_repo"] and s.king_hf_namespace:
            repo_id = f"{s.king_hf_namespace}/{title}"
            publish(fetch(row["model_uri"], s.model_cache_dir), repo_id, card(row, title))
            conn.execute("UPDATE kings SET hf_repo = %s WHERE id = %s", (repo_id, king["id"]))
        elif not king["hf_repo"]:
            log.warning("INGOT_KING_HF_NAMESPACE is empty; {} is not mirrored", title)
    except Exception as exc:
        log.exception("crowning {} failed", row["id"])
        db.fail(
            conn,
            row,
            back_to=states.WON,
            error=f"crowning failed: {exc}",
            max_attempts=config.MAX_ATTEMPTS,
        )
        return True
    db.finish(conn, row["id"], states.CROWNED)
    log.info("crowned {} from submission {} (hotkey {})", title, row["id"], row["hotkey"])
    return True


def card(row: db.Row, title: str) -> str:
    return (
        f"# {title}\n\n"
        "A public mirror of a model that won on Ingot, a Bittensor subnet where miners quantize a "
        "fixed teacher model and the validator crowns the copy that stays closest to it.\n\n"
        f"- Submitted by miner hotkey `{row['hotkey']}` (uid {row['uid']})\n"
        f"- Crowned from submission `{row['id']}`\n\n"
        "The weights are the miner's upload, byte for byte. Credit belongs to the miner.\n"
    )


def main() -> None:
    s = settings()
    conn = db.connect()
    worker = db.register_worker(conn, "promoter")
    log.info("promoter started — mirroring kings to {}", s.king_hf_namespace or "(nowhere)")
    forever(
        lambda: run_once(conn, worker=worker, s=s, spec=chain_spec()),
        idle_s=config.IDLE_S,
        on_stop=lambda: db.offline(conn, worker),
    )
