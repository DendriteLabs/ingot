"""Apply on-chain private-store signals to private_registrations rows.

RevealedCommitments re-yields every signal on every scan, so everything here is idempotent
and a signal that cannot be applied yet (ready before credentials) is retried next tick.
"""

from __future__ import annotations

from loguru import logger as log

from chain_reader import config
from common import guard
from common.chain import PrivateSignal
from common.config import chain_spec
from common.db import Conn
from private_store.contracts import (
    parse_activation,
    parse_ready_signal,
    registration_id,
)


def handle(conn: Conn, signals: list[PrivateSignal]) -> int:
    applied = 0
    for signal in signals:
        _record(conn, signal)
        try:
            apply = _apply_activate if signal.kind == "activate" else _apply_ready
            applied += apply(conn, signal)
        except Exception as exc:
            log.warning("{} signal from {} rejected: {}", signal.kind, signal.hotkey, exc)
    return applied


def _record(conn: Conn, s: PrivateSignal) -> None:
    conn.execute(
        """
        INSERT INTO commits (netuid, hotkey, block, uid, coldkey, kind, payload, block_hash)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (netuid, hotkey, block) DO NOTHING
        """,
        (s.netuid, s.hotkey, s.block, s.uid, s.coldkey, s.kind, s.payload, s.block_hash),
    )


def _rid(signal: PrivateSignal) -> str:
    generation = chain_spec().chain_generation
    return registration_id(netuid=signal.netuid, hotkey=signal.hotkey, chain_generation=generation)


def _apply_activate(conn: Conn, signal: PrivateSignal) -> int:
    if signal.uid is None or guard.is_used(conn, signal.hotkey):
        return 0
    activation = parse_activation(signal.payload)
    spec = chain_spec()
    spec.pair(activation.family, activation.quant)
    if spec.closed(activation.family, activation.quant):
        raise ValueError(f"quant {activation.quant} is closed; open is {spec.quant}")
    pubkey = activation.pubkey.hex()
    rid = _rid(signal)
    inserted = conn.execute(
        """
        INSERT INTO private_registrations
            (netuid, uid, hotkey, coldkey, registration_id, activation_block, submission_pubkey,
             family, quant)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (registration_id) DO NOTHING
        RETURNING id
        """,
        (
            signal.netuid,
            signal.uid,
            signal.hotkey,
            signal.coldkey,
            rid,
            signal.block,
            pubkey,
            activation.family,
            activation.quant,
        ),
    ).fetchone()
    if inserted is not None:
        log.info("activated registration {} hotkey {}", rid, signal.hotkey)
        return 1
    # a later activate re-keys an unfinished attempt, or claims attempt n+1 after a failure
    attempt = conn.execute(
        """
        UPDATE private_registrations
        SET submission_pubkey = %(pubkey)s, attempt_count = attempt_count + 1, state = 'ACTIVATED',
            activation_block = %(block)s, credential_expires_at = NULL, model_prefix = NULL,
            family = %(family)s, quant = %(quant)s,
            coldkey = COALESCE(%(coldkey)s, coldkey), updated_at = now()
        WHERE registration_id = %(rid)s
          AND ( (state IN ('ACTIVATED', 'CREDENTIALED') AND submission_pubkey <> %(pubkey)s)
                OR (state = 'SUBMITTED' AND submission_id IS NULL)
                OR state = 'FAILED' )
          AND %(block)s > activation_block
          AND attempt_count < %(max_attempts)s + extra_attempts
        RETURNING attempt_count
        """,
        {
            "rid": rid,
            "pubkey": pubkey,
            "block": signal.block,
            "max_attempts": config.MAX_UPLOAD_ATTEMPTS,
            "coldkey": signal.coldkey,
            "family": activation.family,
            "quant": activation.quant,
        },
    ).fetchone()
    if attempt is None:
        return 0
    log.info("re-keyed registration {} — attempt {}", rid, attempt["attempt_count"])
    return 1


def _apply_ready(conn: Conn, signal: PrivateSignal) -> int:
    model_digest = parse_ready_signal(signal.payload)
    rid = _rid(signal)
    updated = conn.execute(
        """
        UPDATE private_registrations
        SET state = 'READY', model_digest = %s, ready_block = %s, ready_block_hash = %s,
            updated_at = now()
        WHERE registration_id = %s AND state = 'CREDENTIALED' AND %s > activation_block
        RETURNING id
        """,
        (model_digest, signal.block, signal.block_hash, rid, signal.block),
    ).fetchone()
    if updated is not None:
        log.info("ready for registration {} model digest {}", rid, model_digest)
    return int(updated is not None)
