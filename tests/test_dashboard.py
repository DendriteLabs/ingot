"""The dashboard document after a real pipeline run: the king, the reign the weight setter
would pay, the eval history and the failures, and that it serialises."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from common import db, states
from dashboard import state
from dashboard.main import Publisher, _json_default
from eval_service import config as eval_config
from model_validation import config as validation_config
from model_validation import main as model_validation
from promoter import main as promoter
from tests.conftest import crown_genesis, submit
from tests.test_pipeline import FAMILY, WORKER, dedup, duel, evaluate, fetch, valid
from tests.test_private_upload import HOTKEY, FakeS3, make_deps, submit_through

FILES = ["config.json", "model.safetensors.index.json", "model.safetensors"]


def test_state_after_a_coronation(conn, s, spec):
    crown_genesis(conn)
    conn.execute("INSERT INTO workers (name, service) VALUES (%s, %s)", (WORKER, "test"))
    winner = submit(conn, hotkey="hk-win", uid=7)
    model_validation.run_once(
        conn,
        worker=WORKER,
        list_files=lambda _uri: FILES,
        load=lambda _family: FAMILY,
        s=s,
        spec=spec,
        fetch=fetch,
        check=valid,
    )
    dedup(conn, s)
    evaluate(conn, s, duel(0.1, 0.5))
    promoter.run_once(conn, worker=WORKER, s=s, spec=spec, fetch=fetch, publish=lambda *_: "url")
    loser = submit(conn, hotkey="hk-lose", uid=8)
    model_validation.run_once(
        conn,
        worker=WORKER,
        list_files=lambda _uri: FILES,
        load=lambda _family: FAMILY,
        s=s,
        spec=spec,
        fetch=fetch,
        check=valid,
    )
    dedup(conn, s)
    evaluate(conn, s, duel(0.9, 0.1))
    submit(conn, hotkey="hk-bad", uid=9)
    model_validation.run_once(
        conn,
        worker=WORKER,
        list_files=lambda _uri: FILES,
        load=lambda _family: FAMILY,
        s=s,
        spec=spec,
        fetch=fetch,
        check=lambda *_a, **_k: "no fp8",
    )
    conn.execute("INSERT INTO neurons (uid, hotkey) VALUES (7, 'hk-win'), (8, 'hk-lose')")

    doc = state.build(conn, s=s, spec=spec)

    king = doc["king"]
    assert doc["links"]["dataset"] == f"https://huggingface.co/datasets/{spec.dataset_repo}"
    assert (king["name"], king["uid"], king["score"]) == ("Ingot-I-27B-FP8", 7, 0.1)
    won = db.one(conn, "SELECT id FROM evals WHERE won")["id"]
    assert king["eval_id"] == won and doc["kings"][0]["eval_id"] is None, "a king links its duel"
    assert [k["name"] for k in doc["kings"]] == ["GENESIS", "Ingot-I-27B-FP8"]
    assert doc["kings"][0]["score"] == 0.5, "genesis is scored by the first eval against it"
    assert [(k["name"], k["weight"]) for k in doc["reign"]["kings"]] == [("Ingot-I-27B-FP8", 0.2)]
    assert doc["reign"]["burn"] == {"uid": 0, "weight": 0.8}
    assert doc["stats"] == {"evaluated": 2, "crowned": 1, "queued": 0}
    assert [(e["hotkey"], e["won"], e["king"], e["dist_challenger"]) for e in doc["history"]] == [
        ("hk-lose", False, "Ingot-I-27B-FP8", 0.9),
        ("hk-win", True, "GENESIS", 0.1),
    ]
    assert all(
        e["dist_margin"] == pytest.approx(-eval_config.DIST_SLACK * e["dist_king"])
        for e in doc["history"]
    )
    assert doc["margins"] == {"kl": eval_config.KL_MARGIN, "dist": -eval_config.DIST_SLACK}
    assert [(e["uid"], e["king"], e["score_challenger"]) for e in doc["losers"]] == [
        (8, "Ingot-I-27B-FP8", 0.9)
    ]
    assert [(f["hotkey"], f["stage"], f["reason"], f["quant"]) for f in doc["fails"]] == [
        ("hk-bad", "INVALID", "no fp8", "fp8")
    ]
    assert (
        db.one(conn, "SELECT state FROM submissions WHERE id = %s", (winner,))["state"] == "CROWNED"
    )
    assert db.one(conn, "SELECT state FROM submissions WHERE id = %s", (loser,))["state"] == "LOST"
    json.dumps(doc, default=_json_default)

    conn.execute("UPDATE neurons SET hotkey = 'hk-new' WHERE uid = 7")
    doc = state.build(conn, s=s, spec=spec)
    assert doc["reign"]["kings"] == [], "a king whose uid changed hands holds no weight"
    assert doc["reign"]["burn"] == {"uid": 0, "weight": 1.0}


def test_registration_feed_is_the_last_day(conn, s, spec):
    now = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    samples = [
        (1, now - timedelta(hours=25), 1),  # outside REGISTRATION_HOURS: dropped
        (2, now - timedelta(hours=23), 2),
        (3, now - timedelta(minutes=5), 3),
    ]
    conn.cursor().executemany(
        "INSERT INTO registration_prices (block, at, burn_rao) VALUES (%s, %s, %s)",
        [(block, at, rao * 1_000_000_000) for block, at, rao in samples],
    )
    doc = state.build(conn, s=s, spec=spec, now=now)
    assert [(block, tao) for _, block, tao in doc["registration"]] == [(2, 2.0), (3, 3.0)]


def test_eval_page_reads_its_artifacts(conn, s, spec, tmp_path):
    crown_genesis(conn)
    genesis = db.one(conn, "SELECT id FROM kings")["id"]
    sid = submit(conn, hotkey="hk-1", uid=7)
    eval_id = db.one(
        conn,
        "INSERT INTO evals (submission_id, king_id, teacher, dataset_hash, seed, sample_count,"
        " margin, score_challenger, score_king, won) VALUES (%s, %s, 't', 'd', 'x', 2, 0.1,"
        " 0.2, 0.5, false) RETURNING id",
        (sid, genesis),
    )["id"]
    s = s.model_copy(update={"artifacts_dir": tmp_path})
    bare = state.eval_detail(conn, eval_id, s=s, spec=spec)
    assert bare is not None and bare["artifacts"] == [] and bare["samples"] == [], (
        "nothing uploaded"
    )

    folder = state.artifact_dir(s, "hk-1", eval_id)
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text("{}")
    (folder / "samples.jsonl").write_text('{"sample_id": "a"}\n{"sample_id": "b"}\n')
    doc = state.eval_detail(conn, eval_id, s=s, spec=spec)
    assert doc is not None
    assert (doc["hotkey"], doc["uid"], doc["king"], doc["crowned_as"]) == (
        "hk-1",
        7,
        "GENESIS",
        None,
    )
    assert doc["eval"]["score_challenger"] == 0.2
    assert [r["sample_id"] for r in doc["samples"]] == ["a", "b"]
    assert doc["artifacts"] == ["summary.json", "samples.jsonl"]
    assert state.eval_detail(conn, eval_id + 1, s=s, spec=spec) is None


def _eval(conn, sid, king_id, *, won, minutes_ago=5):
    return db.one(
        conn,
        "INSERT INTO evals (submission_id, king_id, teacher, dataset_hash, seed, sample_count,"
        " margin, score_challenger, score_king, won, finished_at) VALUES (%s, %s, 't', 'd', 'x',"
        " 1, 0.1, 0.2, 0.5, %s, now() - make_interval(mins => %s)) RETURNING id",
        (sid, king_id, won, minutes_ago),
    )["id"]


def test_publisher_keeps_the_site_current(conn, s, spec, tmp_path):
    crown_genesis(conn)
    genesis = db.one(conn, "SELECT id FROM kings")["id"]
    with_artifacts = _eval(conn, submit(conn, hotkey="hk-a", uid=1), genesis, won=False)
    still_uploading = _eval(conn, submit(conn, hotkey="hk-b", uid=2), genesis, won=False)
    never_uploaded = _eval(
        conn, submit(conn, hotkey="hk-c", uid=3), genesis, won=False, minutes_ago=90
    )
    winner_sid = submit(conn, hotkey="hk-d", uid=4)
    winner = _eval(conn, winner_sid, genesis, won=True)
    s3, s = FakeS3(), s.model_copy(update={"artifacts_dir": tmp_path})
    for name, body in (
        ("samples.jsonl", b'{"sample_id": "a", "runs": {}}\n'),
        ("summary.json", b"{}"),
    ):
        s3.put("artifacts", f"evals/hk-a/{with_artifacts}/{name}", body)
    s3.put("artifacts", f"evals/hk-d/{winner}/summary.json", b"{}")
    publisher = Publisher(db.Link(conn), s3, s=s, spec=spec, site="site", artifacts="artifacts")

    def site() -> set[str]:
        return {key for bucket, key in s3.objects if bucket == "site"}

    assert publisher.run_once()
    assert site() == {
        "api/state.json",
        f"api/evals/{with_artifacts}.json",
        f"api/evals/{with_artifacts}/summary.json",
        f"api/evals/{with_artifacts}/samples.jsonl",
        f"api/evals/{never_uploaded}.json",
    }, "a recent eval waits for summary.json; one past the grace goes up without it"
    page = json.loads(s3.body("site", f"api/evals/{with_artifacts}.json"))
    assert page["artifacts"] == ["summary.json", "samples.jsonl"] and len(page["samples"]) == 1
    assert f"api/evals/{winner}.json" not in site(), "a win waits for its coronation"
    assert not publisher.run_once(), "nothing changed, nothing written"

    conn.execute(
        "INSERT INTO kings (submission_id, hotkey, uid, model_uri, family, quant)"
        " VALUES (%s, 'hk-d', 4, 'm', 'qwen38_27b', 'fp8')",
        (winner_sid,),
    )
    assert publisher.run_once()
    assert f"api/evals/{winner}.json" in site()
    assert json.loads(s3.body("site", "api/state.json"))["king"]["uid"] == 4, "the new king"
    assert (
        still_uploading
        not in Publisher(
            db.Link(conn), s3, s=s, spec=spec, site="site", artifacts="artifacts"
        ).published
    ), "a restart re-reads what the bucket holds"


@pytest.mark.usefixtures("fp8_open")
def test_a_committed_model_is_queued_and_published(conn, s, spec):
    """From the chain to the site: r2activate and r2ready, the verified upload, the queue."""
    crown_genesis(conn)
    s3 = FakeS3()
    submit_through(conn, s3, make_deps(s3))
    queued = [(r["hotkey"], r["state"]) for r in state.build(conn, s=s, spec=spec)["queue"]["rows"]]
    assert queued == [(HOTKEY, states.SUBMITTED)]

    assert Publisher(db.Link(conn), s3, s=s, spec=spec, site="site", artifacts="").run_once()
    published = json.loads(s3.body("site", "api/state.json"))
    assert [(r["hotkey"], r["state"]) for r in published["queue"]["rows"]] == queued


def test_static_files_exist():
    root = Path(__file__).resolve().parents[1] / "web"
    assert {p.name for p in root.iterdir()} >= {
        "index.html",
        "app.js",
        "favicon.ico",
        "brand",
        "css",
        "js",
    }
    assert {p.name for p in (root / "css").iterdir()} >= {"ingot.css"}
    assert {p.name for p in (root / "brand").iterdir()} >= {"ingot-white.svg", "ingot-black.svg"}


def test_fails_show_tries_left_and_the_ban(conn, s, spec):
    crown_genesis(conn)
    conn.execute("INSERT INTO workers (name, service) VALUES (%s, %s)", (WORKER, "test"))
    for i in range(validation_config.MAX_INVALID):
        submit(conn, hotkey="hk-1", block=100 + i)
        model_validation.run_once(
            conn,
            worker=WORKER,
            list_files=lambda _uri: FILES,
            load=lambda _family: FAMILY,
            s=s,
            spec=spec,
            fetch=fetch,
            check=lambda *_a, **_k: "no fp8",
            upload=lambda *_a: None,
        )
        fails = state.build(conn, s=s, spec=spec)["fails"]
        left = validation_config.MAX_INVALID - i - 1
        assert {(f["invalid_left"], f["barred"]) for f in fails} == {(left, left == 0)}
    assert fails[0]["barred_by"] == "invalid"
