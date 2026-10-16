"""The hotkey ledger: who may not submit again, and the metagraph snapshot that catches swaps."""

from __future__ import annotations

from typing import Any

from loguru import logger as log
from psycopg.types.json import Jsonb

from common.chain import SwapEvent
from common.db import Conn


def is_used(conn: Conn, hotkey: str) -> bool:
    return (
        conn.execute("SELECT 1 FROM used_hotkeys WHERE hotkey = %s", (hotkey,)).fetchone()
        is not None
    )


def mark_used(conn: Conn, hotkey: str, source: str, detail: dict[str, Any] | None = None) -> bool:
    """Bar a hotkey. Returns False when it was already barred."""
    return (
        conn.execute(
            "INSERT INTO used_hotkeys (hotkey, source, detail) VALUES (%s, %s, %s)"
            " ON CONFLICT (hotkey) DO NOTHING",
            (hotkey, source, Jsonb(detail or {})),
        ).rowcount
        == 1
    )


def record_legacy(conn: Conn, commits: list[tuple[str, int, str]], start_block: int) -> int:
    """Bar every hotkey that committed anything at or before start_block."""
    barred = 0
    for hotkey, block, _payload in commits:
        if block <= start_block and mark_used(conn, hotkey, "backfill", {"block": block}):
            barred += 1
    log.info("backfill: {} hotkeys barred (commits at or before block {})", barred, start_block)
    return barred


def uid_state(conn: Conn) -> dict[int, tuple[str, int | None]]:
    rows = conn.execute("SELECT uid, hotkey, registration_block FROM neurons").fetchall()
    return {row["uid"]: (row["hotkey"], row["registration_block"]) for row in rows}


def record_swaps(conn: Conn, swaps: list[SwapEvent], netuid: int, block: int) -> int:
    barred = 0
    for swap in swaps:
        detail = {**swap.detail(), "netuid": netuid, "detected_at_block": block}
        if mark_used(conn, swap.new_hotkey, "swap", detail):
            barred += 1
            log.warning("hotkey swap: uid {} {} -> {}", swap.uid, swap.old_hotkey, swap.new_hotkey)
    return barred


def refresh_neurons(
    conn: Conn, snapshot: list[tuple[int, str, int]], owners: dict[str, str]
) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO neurons (uid, hotkey, coldkey, registration_block)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (uid) DO UPDATE SET hotkey = EXCLUDED.hotkey, coldkey = EXCLUDED.coldkey,
                registration_block = EXCLUDED.registration_block, updated_at = now()
            """,
            [(uid, hotkey, owners.get(hotkey), reg_block) for uid, hotkey, reg_block in snapshot],
        )
