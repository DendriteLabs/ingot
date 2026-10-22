from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from loguru import logger as log

from common import chain, db, reign
from common.config import Settings, settings
from common.loop import forever
from weight_setter import config


class Chain(Protocol):
    def block(self) -> int: ...
    def hotkey_by_uid(self, netuid: int) -> dict[int, str]: ...
    def set_weights(
        self, *, netuid: int, uids: list[int], weights: list[float]
    ) -> tuple[bool, str]: ...


def run_once(
    conn: db.Conn, client: Chain, *, worker: str, s: Settings, now: datetime | None = None
) -> bool:
    """Set weights if the rate limit allows. Returns whether a set_weights call was made."""
    db.heartbeat(conn, worker)
    block = client.block()
    last = db.one(conn, "SELECT max(block) AS block FROM weight_sets WHERE ok")["block"]
    if last is not None and block - last < config.SET_RATE_BLOCKS:
        return False
    rows = conn.execute(
        "SELECT id, uid, hotkey, crowned_at FROM kings WHERE submission_id IS NOT NULL"
        " ORDER BY id DESC LIMIT %s",
        (reign.SLOTS,),
    )
    rows = rows.fetchall()
    kings = [reign.King(r["uid"], r["hotkey"], r["crowned_at"]) for r in rows]
    by_uid = reign.weights(kings, now or datetime.now(UTC), client.hotkey_by_uid(s.netuid))
    uids = sorted(by_uid)
    weights = [by_uid[uid] for uid in uids]
    ok, message = client.set_weights(netuid=s.netuid, uids=uids, weights=weights)
    conn.execute(
        "INSERT INTO weight_sets (block, uids, weights, kings, ok, error)"
        " VALUES (%s, %s, %s, %s, %s, %s)",
        (block, uids, weights, [r["id"] for r in rows], ok, None if ok else message),
    )
    log.log(
        "INFO" if ok else "ERROR",
        "set_weights at block {}: {} → {}",
        block,
        dict(zip(uids, weights, strict=True)),
        message or "ok",
    )
    return True


def main() -> None:
    s = settings()
    link = db.Link()
    client = chain.WeightClient(
        coldkey=s.wallet_coldkey,
        hotkey=s.wallet_hotkey,
        wallet_path=s.wallet_path,
        network=s.network,
    )
    worker = db.register_worker(link.conn(), "weight_setter")
    log.info("weight setter started — netuid={} every {} blocks", s.netuid, config.SET_RATE_BLOCKS)
    # always sleep: a set_weights call is followed by SET_RATE_BLOCKS of nothing to do
    forever(
        lambda: run_once(link.conn(), client, worker=worker, s=s) and False,
        idle_s=config.POLL_S,
        on_stop=lambda: db.offline_quietly(link, worker),
    )
