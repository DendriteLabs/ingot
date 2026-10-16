"""Everything the page shows, in one document. Reign and weights come from common.reign, the
same rule the weight setter applies, so the page cannot disagree with the chain."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from common import db, reign, states
from common.config import ChainSpec, Settings
from dashboard import config
from model_validation import config as validation_config

ACTIVE = [
    states.SUBMITTED,
    states.VALIDATING,
    states.VALID,
    states.DEDUPING,
    states.UNIQUE,
    states.EVALUATING,
    states.WON,
    states.CROWNING,
]


def build(conn: db.Conn, *, s: Settings, spec: ChainSpec, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    kings = _kings(conn, spec)
    names = {k["id"]: k["name"] for k in kings}
    return {
        "updated_at": now,
        "netuid": s.netuid,
        "teacher": spec.teacher_repo,
        "genesis": spec.genesis_repo,
        "links": {
            "github": config.GITHUB_URL,
            "dataset": f"https://huggingface.co/datasets/{spec.dataset_repo}"
            if spec.dataset_repo
            else None,
            "huggingface": f"https://huggingface.co/{s.king_hf_namespace}"
            if s.king_hf_namespace
            else None,
        },
        "king": kings[-1] if kings else None,
        "kings": kings,
        "stats": _stats(conn),
        "reign": _reign(conn, kings, now),
        "queue": _queue(conn),
        "history": _history(conn, names),
        "losers": _losers(conn, names),
        "fails": _fails(conn),
        "registration": _registration(conn, now),
    }


def _kings(conn: db.Conn, spec: ChainSpec) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT k.id, k.submission_id, k.hotkey, k.uid, k.hf_repo, k.crowned_at,
               s.family, s.quant,
               CASE WHEN k.submission_id IS NOT NULL
                    THEN row_number() OVER (PARTITION BY k.submission_id IS NULL ORDER BY k.id)
               END AS ordinal,
               COALESCE(w.score_challenger, first.score_king) AS score,
               w.score_challenger AS duel_challenger, w.score_king AS duel_king, w.id AS eval_id
        FROM kings k
        LEFT JOIN submissions s ON s.id = k.submission_id
        LEFT JOIN LATERAL (
            SELECT e.id, e.score_challenger, e.score_king FROM evals e
            WHERE e.submission_id = k.submission_id AND e.won ORDER BY e.id DESC LIMIT 1
        ) w ON TRUE
        LEFT JOIN LATERAL (
            SELECT e.score_king FROM evals e
            WHERE e.king_id = k.id AND e.score_king IS NOT NULL ORDER BY e.id LIMIT 1
        ) first ON TRUE
        ORDER BY k.id
        """
    ).fetchall()
    return [
        {
            **row,
            "name": reign.name(
                row["ordinal"],
                spec.king_name(row["family"], row["quant"]) if row["ordinal"] else "",
            ),
        }
        for row in rows
    ]


# the files eval_service uploads for every eval, under evals/<hotkey>/<eval id>/, and their types
ARTIFACTS = {"summary.json": "application/json", "samples.jsonl": "application/x-ndjson"}


def eval_detail(conn: db.Conn, eval_id: int, *, s: Settings, spec: ChainSpec) -> dict | None:
    """One eval's page: its row, who ran it against which king, and its per-sample records from
    the local copy of the artifacts bucket. None when there is no such eval."""
    row = conn.execute(
        "SELECT e.*, s.hotkey, s.uid FROM evals e JOIN submissions s ON s.id = e.submission_id"
        " WHERE e.id = %s",
        (eval_id,),
    ).fetchone()
    if row is None:
        return None
    kings = _kings(conn, spec)
    names = {k["id"]: k["name"] for k in kings}
    crowned = next((k["name"] for k in kings if k["submission_id"] == row["submission_id"]), None)
    folder = artifact_dir(s, row["hotkey"], eval_id)
    samples = folder / "samples.jsonl"
    return {
        "eval": {k: v for k, v in row.items() if k not in ("hotkey", "uid")},
        "hotkey": row["hotkey"],
        "uid": row["uid"],
        "king": names.get(row["king_id"], "?"),
        "crowned_as": crowned,
        "teacher": spec.teacher_repo,
        "netuid": s.netuid,
        "links": {"github": config.GITHUB_URL},
        "samples": [json.loads(line) for line in samples.read_text().splitlines() if line]
        if samples.exists()
        else [],
        "artifacts": [name for name in ARTIFACTS if (folder / name).exists()],
    }


def artifact_dir(s: Settings, hotkey: str, eval_id: int) -> Path:
    return s.artifacts_dir / "evals" / hotkey / str(eval_id)


def _stats(conn: db.Conn) -> dict[str, int]:
    return db.one(
        conn,
        """
        SELECT (SELECT count(*) FROM evals WHERE won IS NOT NULL) AS evaluated,
               (SELECT count(*) FROM kings WHERE submission_id IS NOT NULL) AS crowned,
               (SELECT count(*) FROM submissions
                WHERE state IN ('SUBMITTED', 'VALIDATING', 'VALID', 'DEDUPING', 'UNIQUE',
                                'EVALUATING')) AS queued
        """,
    )


def _reign(conn: db.Conn, kings: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    hotkeys = {
        r["uid"]: r["hotkey"] for r in conn.execute("SELECT uid, hotkey FROM neurons").fetchall()
    }
    crowned = [k for k in kings if k["uid"] is not None]
    by_rule = [reign.King(k["uid"], k["hotkey"], k["crowned_at"]) for k in crowned]
    paid = set(reign.paid(by_rule, now))
    rows = []
    for king, rule in zip(crowned, by_rule, strict=True):
        # a reigning king whose uid now belongs to another hotkey is paid nothing; not listed
        if rule not in paid or hotkeys.get(king["uid"]) != king["hotkey"]:
            continue
        rows.append(
            {
                **{key: king[key] for key in ("name", "uid", "hotkey", "hf_repo")},
                "crowned_at": king["crowned_at"],
                "ends_at": reign.ends_at(king["crowned_at"]),
                "weight": reign.SHARE,
            }
        )
    weights = reign.weights(by_rule, now, hotkeys)
    return {
        "slots": reign.SLOTS,
        "share": reign.SHARE,
        "reign_days": reign.REIGN_DAYS,
        "kings": sorted(rows, key=lambda r: r["crowned_at"], reverse=True),
        "burn": {"uid": reign.BURN_UID, "weight": weights.get(reign.BURN_UID, 0.0)},
    }


def _queue(conn: db.Conn) -> dict[str, Any]:
    counts = conn.execute(
        "SELECT state, count(*) AS n FROM submissions WHERE state = ANY(%s) GROUP BY state",
        (ACTIVE,),
    ).fetchall()
    rows = conn.execute(
        """
        SELECT id, hotkey, uid, state, created_at, updated_at FROM submissions
        WHERE state = ANY(%s)
        ORDER BY state = 'EVALUATING' DESC, created_at
        LIMIT %s
        """,
        (ACTIVE, config.QUEUE_LIMIT),
    ).fetchall()
    return {
        "counts": {state: 0 for state in ACTIVE} | {r["state"]: r["n"] for r in counts},
        "rows": rows,
    }


def _history(conn: db.Conn, names: dict[int, str]) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT e.id, e.started_at, e.finished_at, e.king_id, e.sample_count, e.margin,
               e.dist_margin, e.score_challenger, e.score_king, e.dist_challenger, e.dist_king,
               e.won, e.error, s.hotkey, s.uid
        FROM evals e JOIN submissions s ON s.id = e.submission_id
        ORDER BY e.id DESC LIMIT %s
        """,
        (config.HISTORY_LIMIT,),
    ).fetchall()
    return [{**row, "king": names.get(row["king_id"], "?")} for row in rows]


def _losers(conn: db.Conn, names: dict[int, str]) -> list[dict[str, Any]]:
    """Every scored duel the challenger lost, oldest first, for the kings chart."""
    rows = conn.execute(
        """
        SELECT * FROM (
            SELECT e.id, e.finished_at, e.king_id, e.score_challenger, e.score_king, s.uid
            FROM evals e JOIN submissions s ON s.id = e.submission_id
            WHERE e.won = FALSE AND e.score_challenger IS NOT NULL AND e.score_king IS NOT NULL
            ORDER BY e.id DESC LIMIT %s
        ) l ORDER BY id
        """,
        (config.LOSERS_LIMIT,),
    ).fetchall()
    return [{**row, "king": names.get(row["king_id"], "?")} for row in rows]


def _registration(conn: db.Conn, now: datetime) -> list[tuple[float, int, float]]:
    """(unix seconds, block, TAO) for every sample in the chart's window, oldest first."""
    rows = conn.execute(
        "SELECT at, block, burn_rao FROM registration_prices WHERE at > %s ORDER BY at",
        (now - timedelta(hours=config.REGISTRATION_HOURS),),
    ).fetchall()
    return [(r["at"].timestamp(), r["block"], r["burn_rao"] / 1e9) for r in rows]


def _fails(conn: db.Conn) -> list[dict[str, Any]]:
    return conn.execute(
        """
        SELECT f.*,
               GREATEST(%s - (SELECT count(*) FROM submissions s
                              WHERE s.hotkey = f.hotkey AND s.state = 'INVALID'), 0)
                   AS invalid_left,
               u.hotkey IS NOT NULL AS barred,
               u.source AS barred_by,
               u.created_at AS barred_at
        FROM (
            SELECT hotkey, uid, state AS stage, error AS reason, updated_at AS at
            FROM submissions WHERE state IN ('INVALID', 'DUPLICATE', 'FAILED')
            UNION ALL
            SELECT hotkey, uid, 'UPLOAD' AS stage, fault_message AS reason, updated_at AS at
            FROM private_registrations WHERE state = 'FAILED'
        ) f LEFT JOIN used_hotkeys u ON u.hotkey = f.hotkey
        ORDER BY f.at DESC LIMIT %s
        """,
        (validation_config.MAX_INVALID, config.FAILS_LIMIT),
    ).fetchall()
