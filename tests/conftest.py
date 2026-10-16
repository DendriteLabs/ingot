"""Pipeline tests run against a real Postgres (docker compose up -d). Each session gets its own
database built from schema.sql; each test starts with empty tables. The chain, R2, Hugging
Face and the GPU are faked at the boundary of the service under test."""

from __future__ import annotations

import base64
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import psycopg
import pytest
from psycopg import sql

from common import db
from common.config import REPO_ROOT, ChainSpec, Settings, chain_spec


def pg_settings(db: str, *, model_cache_dir: Path = Path("/nowhere")) -> Settings:
    """Settings pointing at the docker-compose Postgres (or INGOT_POSTGRES_* in the env)."""
    return Settings(
        _env_file=None,
        postgres_user=os.environ.get("INGOT_POSTGRES_USER", "ingot"),
        postgres_password=os.environ.get("INGOT_POSTGRES_PASSWORD", "ingot"),
        postgres_host=os.environ.get("INGOT_POSTGRES_HOST", "127.0.0.1"),
        postgres_port=int(os.environ.get("INGOT_POSTGRES_PORT", "5432")),
        postgres_db=db,
        model_cache_dir=model_cache_dir,
        eval_work_dir=model_cache_dir / "eval",
        king_hf_namespace="ingot-test",
    )


@pytest.fixture(scope="session")
def test_db() -> Iterator[str]:
    admin_url = pg_settings("postgres").database_url
    try:
        admin = psycopg.connect(admin_url, autocommit=True)
    except psycopg.OperationalError as exc:
        pytest.skip(f"no Postgres at {admin_url} ({exc}); run `docker compose up -d`")
    name = f"ingot_test_{uuid.uuid4().hex[:8]}"
    admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    with psycopg.connect(pg_settings(name).database_url, autocommit=True) as conn:
        conn.execute((REPO_ROOT / "schema.sql").read_bytes())
    yield name
    admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
    admin.close()


@pytest.fixture
def s(test_db: str, tmp_path: Path) -> Settings:
    return pg_settings(test_db, model_cache_dir=tmp_path)


@pytest.fixture
def conn(s: Settings) -> Iterator[db.Conn]:
    conn = db.connect(s.database_url)
    yield conn
    conn.execute(
        "TRUNCATE submissions, private_registrations, evals, dedups, kings, weight_sets,"
        " used_hotkeys, neurons, workers, events, commits, registration_prices"
        " RESTART IDENTITY CASCADE"
    )
    conn.close()


@pytest.fixture
def spec() -> ChainSpec:
    return chain_spec()


def submit(conn: db.Conn, *, hotkey: str = "hk-1", uid: int = 7, block: int = 100) -> uuid.UUID:
    """Insert a verified private upload, as the access controller would."""
    digest = uuid.uuid4().hex * 2
    row = db.one(
        conn,
        "INSERT INTO submissions"
        " (hotkey, uid, model_uri, model_hash, block, block_hash, family, quant)"
        " VALUES (%s, %s, %s, %s, %s, %s, 'qwen38_27b', 'fp8') RETURNING id",
        (hotkey, uid, f"s3://private/models/{hotkey}@sha256:{digest}", digest, block, digest[:16]),
    )
    return row["id"]


def crown_genesis(conn: db.Conn) -> None:
    conn.execute(
        "INSERT INTO kings (model_uri, hf_repo) VALUES ('org/genesis@main', 'org/genesis')"
    )


def state(conn: db.Conn, submission_id: uuid.UUID) -> str:
    return db.one(conn, "SELECT state FROM submissions WHERE id = %s", (submission_id,))["state"]


def fingerprint(error: np.ndarray, *, tensors_hash: str = "") -> dict:
    """A one-tensor fingerprint whose sketch is `error`, standing in for the GPU fingerprint."""
    return {
        "version": "test",
        "key_id": "test",
        "tensors_hash": tensors_hash or uuid.uuid4().hex,
        "rel_to_teacher": 0.02,
        "identity_frac": 1.0,
        "rescale_frac": 0.0,
        "residual": {"identity": True, "scale": 1.0, "off_identity": 0.0, "perm_like": 1.0},
        "tensors": [
            {
                "name": "t",
                "k": error.shape[0],
                "s": base64.b64encode(error.astype(np.float32).tobytes()).decode(),
            }
        ],
    }


def quantization_error(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(64, 64))
