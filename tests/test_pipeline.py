"""The path every submission takes: submit → validate → eval → crown → weights, through the
real database, with the chain, model stores and GPU faked at each service's boundary."""

from __future__ import annotations

import json
import zlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from chain_reader import signals
from common import chain, db, guard, states
from common.chain import PrivateSignal
from common.config import chain_spec
from eval_service import config as eval_config
from eval_service import main as eval_service
from eval_service import servers
from eval_service.dataset import Dataset, Sample
from eval_service.run import Distance, Duel, Scorer
from model_dedup import config as dedup_config
from model_dedup import main as model_dedup
from model_dedup.bank import AUDIT, BANK, GENESIS
from model_dedup.main import Fingerprint
from model_dedup.verdict import Verdict
from model_validation import config as validation_config
from model_validation import main as model_validation
from model_validation.families.base import Family
from private_store.contracts import activation_signal_payload
from promoter import main as promoter
from tests.conftest import crown_genesis, fingerprint, quantization_error, state, submit
from weight_setter import main as weight_setter

DATA = Dataset(
    samples=[Sample(f"s{i}", [{"role": "user", "content": f"sample {i}"}]) for i in range(10)],
    sha256="d1",
)
FAMILY = Family(
    name="x",
    genesis_config={},
    arch_keys=(),
    genesis_layout={},
    frozen_dtype_prefixes=(),
    genesis_weight_bytes=1,
)
TEACHER = Path("/teacher")
WORKER = "test"


@pytest.fixture(autouse=True)
def rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_config, "SAMPLES_PER_DOMAIN", 4)
    monkeypatch.setattr(eval_config, "KL_MARGIN", 0.05)
    monkeypatch.setattr(eval_config, "DIST_MARGIN", 0.05)
    monkeypatch.setattr(validation_config, "MAX_ATTEMPTS", 2)


@pytest.fixture(autouse=True)
def worker(conn) -> None:
    conn.execute("INSERT INTO workers (name, service) VALUES (%s, %s)", (WORKER, "test"))


def fetch(uri: str, _cache: Path) -> Path:
    """The "model directory" is the uri, so a scorer can tell the upload from the king."""
    return Path(uri)


def valid(_model_dir: Path, **_: object) -> str | None:
    return None


def listing(_uri: str) -> list[str]:
    return [*chain_spec().required_files, "model.safetensors"]


def duel(challenger: float, king: float) -> Scorer:
    """KL and distance both at the given value, so the two margins agree."""

    def score(_teacher: Path, _challenger: Path, _king: Path, samples: Sequence[Sample]) -> Duel:
        return Duel(
            Distance(challenger, challenger),
            Distance(king, king),
            Distance(0.0, 0.0),
            samples=[{"sample_id": sample.id} for sample in samples],
        )

    return score


def no_upload(_key: str, _body: bytes) -> None:
    return None


class FakeArtifacts(dict):
    """The R2 artifacts bucket: key -> uploaded bytes."""

    def __call__(self, key: str, body: bytes) -> str:
        self[key] = body
        return f"s3://artifacts/{key}"


def validate(conn, s, spec, **kwargs) -> bool:
    return model_validation.run_once(
        conn,
        worker=WORKER,
        s=s,
        spec=spec,
        fetch=fetch,
        list_files=listing,
        check=kwargs.get("check", valid),
        load=lambda _family: FAMILY,
        upload=kwargs.get("upload", no_upload),
    )


class FakeBank:
    """The OpenSearch bank at the service boundary: the banked fingerprints and every put."""

    def __init__(self, banked: dict[str, dict] | None = None) -> None:
        self.banked = dict(banked or {})
        self.puts: list[tuple[str, str]] = []

    def shortlist(self, fp: dict, *, hotkey: str, coldkey: str | None) -> dict[str, dict]:
        return dict(self.banked)

    def put(
        self,
        bank_id: str,
        fp: dict,
        *,
        status: str,
        model_uri: str,
        hotkey: str | None = None,
        coldkey: str | None = None,
        verdict: Verdict | None = None,
    ) -> None:
        self.puts.append((bank_id, status))
        if status == BANK:
            self.banked[bank_id] = fp


def own_error(model_dir: Path) -> dict:
    """Every upload quantized independently: an error of its own."""
    return fingerprint(quantization_error(zlib.crc32(str(model_dir).encode())))


def dedup(
    conn, s, bank: FakeBank | None = None, fp: Fingerprint = own_error, upload=no_upload
) -> bool:
    return model_dedup.run_once(
        conn,
        worker=WORKER,
        s=s,
        bank=bank or FakeBank(),
        fingerprint=fp,
        fetch=fetch,
        upload=upload,
    )


def evaluate(conn, s, scorer: Scorer, upload=no_upload) -> bool:
    return eval_service.run_once(
        conn,
        worker=WORKER,
        teacher="org/teacher@main",
        teacher_dir=TEACHER,
        data=DATA,
        s=s,
        fetch=fetch,
        scorer=scorer,
        upload=upload,
    )


class FakeChain:
    def __init__(self, block: int, hotkeys: dict[int, str]) -> None:
        self.current, self.hotkeys = block, hotkeys
        self.calls: list[tuple[list[int], list[float]]] = []

    def block(self) -> int:
        return self.current

    def hotkey_by_uid(self, netuid: int) -> dict[int, str]:
        return self.hotkeys

    def set_weights(
        self, *, netuid: int, uids: list[int], weights: list[float]
    ) -> tuple[bool, str]:
        self.calls.append((uids, weights))
        return True, ""


def test_challenger_beats_king_and_gets_paid(conn, s, spec):
    crown_genesis(conn)
    sid = submit(conn, hotkey="hk-1", uid=7)

    assert validate(conn, s, spec)
    assert state(conn, sid) == states.VALID

    bank = FakeBank()
    assert dedup(conn, s, bank)
    assert state(conn, sid) == states.UNIQUE
    assert bank.puts == [(str(sid), BANK)], "a unique model joins the bank"
    run = db.one(conn, "SELECT duplicate, compared, finished_at FROM dedups")
    assert run["duplicate"] is False and run["compared"] == 0 and run["finished_at"]

    assert evaluate(conn, s, duel(0.1, 0.5))
    assert state(conn, sid) == states.WON
    ev = db.one(conn, "SELECT * FROM evals")
    assert ev["won"] and (ev["score_challenger"], ev["score_king"], ev["sample_count"]) == (
        0.1,
        0.5,
        4,
    )
    assert (ev["teacher"], ev["dataset_hash"]) == ("org/teacher@main", "d1")
    assert (ev["dist_challenger"], ev["dist_king"], ev["noise_kl"]) == (0.1, 0.5, 0.0)
    assert (s.eval_work_dir / "results" / f"{ev['id']}.jsonl").exists()
    assert guard.is_used(conn, "hk-1"), "an evaluated hotkey is spent, win or lose"

    published: list[str] = []

    def publish(_model_dir: Path, repo: str, _card: str) -> str:
        published.append(repo)
        return "url"

    assert promoter.run_once(conn, worker=WORKER, s=s, spec=spec, fetch=fetch, publish=publish)
    assert state(conn, sid) == states.CROWNED
    king = db.one(conn, "SELECT * FROM kings ORDER BY id DESC LIMIT 1")
    assert king["submission_id"] == sid and king["uid"] == 7
    assert published == [king["hf_repo"]] == ["ingot-test/Ingot-I-27B-FP8"]

    chain = FakeChain(block=1000, hotkeys={7: "hk-1"})
    assert weight_setter.run_once(conn, chain, worker=WORKER, s=s)
    assert chain.calls == [([0, 7], [0.8, 0.2])]
    ws = db.one(conn, "SELECT ok, kings FROM weight_sets")
    assert ws["ok"] and ws["kings"] == [king["id"], 1], "newest king first, then genesis"
    chain.current = 1050
    assert not weight_setter.run_once(conn, chain, worker=WORKER, s=s), (
        "rate limited for SET_RATE_BLOCKS"
    )


def test_losing_challenger_leaves_the_king(conn, s, spec):
    crown_genesis(conn)
    sid = submit(conn, hotkey="hk-2", uid=8)
    validate(conn, s, spec)
    dedup(conn, s)

    assert evaluate(conn, s, duel(0.5, 0.1))
    assert state(conn, sid) == states.LOST
    assert db.one(conn, "SELECT count(*) AS n FROM kings")["n"] == 1
    assert not promoter.run_once(
        conn, worker=WORKER, s=s, spec=spec, fetch=fetch, publish=lambda *_: "url"
    )

    chain = FakeChain(block=1000, hotkeys={8: "hk-2"})
    weight_setter.run_once(conn, chain, worker=WORKER, s=s)
    assert chain.calls == [([0], [1.0])], "genesis has no uid, so everything burns"


def test_within_margin_is_not_a_win(conn, s, spec):
    crown_genesis(conn)
    sid = submit(conn)
    validate(conn, s, spec)
    dedup(conn, s)
    evaluate(conn, s, duel(0.46, 0.5))
    assert state(conn, sid) == states.LOST, "0.04 better than the king is inside the 0.05 margin"


def test_invalid_model_is_terminal(conn, s, spec):
    crown_genesis(conn)
    sid = submit(conn)
    validate(conn, s, spec, check=lambda *_a, **_k: "safetensors total 2 bytes, genesis has 1")
    assert state(conn, sid) == states.INVALID
    assert (
        db.one(conn, "SELECT error FROM submissions")["error"]
        == "safetensors total 2 bytes, genesis has 1"
    )
    assert not evaluate(conn, s, duel(0, 1))


def test_crashing_stage_retries_then_fails(conn, s, spec):
    sid = submit(conn)

    def boom(*_a, **_k):
        raise RuntimeError("R2 is down")

    model_validation.run_once(
        conn,
        worker=WORKER,
        s=s,
        spec=spec,
        fetch=boom,
        list_files=listing,
        check=valid,
    )
    row = db.one(conn, "SELECT state, attempts, error FROM submissions")
    assert (row["state"], row["attempts"]) == (states.SUBMITTED, 1) and "R2 is down" in row["error"]

    model_validation.run_once(
        conn,
        worker=WORKER,
        s=s,
        spec=spec,
        fetch=boom,
        list_files=listing,
        check=valid,
    )
    assert state(conn, sid) == states.FAILED, "MAX_ATTEMPTS=2 is spent"


def test_expired_lease_is_reclaimed(conn, s, spec):
    sid = submit(conn)
    conn.execute(
        "UPDATE submissions SET state = 'VALIDATING', attempts = 1,"
        " lease_until = now() - interval '1 second'"
    )

    validate(conn, s, spec)
    assert state(conn, sid) == states.VALID
    events = conn.execute(
        "SELECT state, worker FROM events WHERE submission_id = %s ORDER BY id", (sid,)
    ).fetchall()
    assert [(e["state"], e["worker"]) for e in events] == [
        (states.VALIDATING, WORKER),
        (states.VALID, None),
    ]


def test_evaluator_waits_for_the_promoter(conn, s):
    crown_genesis(conn)
    won = submit(conn, hotkey="hk-won", uid=1)
    conn.execute("UPDATE submissions SET state = 'WON' WHERE id = %s", (won,))
    waiting = submit(conn, hotkey="hk-next", uid=2)
    conn.execute("UPDATE submissions SET state = 'UNIQUE' WHERE id = %s", (waiting,))

    assert not evaluate(conn, s, duel(0, 1))
    assert state(conn, waiting) == states.UNIQUE


def test_the_next_eval_faces_the_new_king(conn, s, spec):
    """A win in a long queue: the following eval waits for the crown, then meets the new king."""
    crown_genesis(conn)
    first, second = submit(conn, hotkey="hk-1", uid=7), submit(conn, hotkey="hk-2", uid=8)
    validate(conn, s, spec)
    validate(conn, s, spec)
    dedup(conn, s)
    dedup(conn, s)

    def score(_teacher: Path, challenger: Path, king: Path, samples: Sequence[Sample]) -> Duel:
        def of(model: Path) -> Distance:
            value = 0.1 if "hk-1" in str(model) else 0.5
            return Distance(value, value)

        records = [{"sample_id": sample.id} for sample in samples]
        return Duel(of(challenger), of(king), Distance(0.0, 0.0), samples=records)

    assert evaluate(conn, s, score) and state(conn, first) == states.WON
    assert not evaluate(conn, s, score), "the queue waits while the winner is crowned"
    assert promoter.run_once(
        conn, worker=WORKER, s=s, spec=spec, fetch=fetch, publish=lambda *_: "url"
    )
    new_king = db.one(conn, "SELECT id, submission_id FROM kings ORDER BY id DESC LIMIT 1")
    assert new_king["submission_id"] == first

    assert evaluate(conn, s, score) and state(conn, second) == states.LOST
    faced = db.one(conn, "SELECT king_id FROM evals WHERE submission_id = %s", (second,))
    assert faced["king_id"] == new_king["id"]


def test_one_eval_at_a_time(conn, s):
    crown_genesis(conn)
    running = submit(conn, hotkey="hk-busy", uid=1)
    conn.execute(
        "UPDATE submissions SET state = 'EVALUATING', lease_until = now() + interval '1 hour'"
        " WHERE id = %s",
        (running,),
    )
    waiting = submit(conn, hotkey="hk-next", uid=2)
    conn.execute("UPDATE submissions SET state = 'UNIQUE' WHERE id = %s", (waiting,))

    assert not evaluate(conn, s, duel(0, 1))
    assert state(conn, waiting) == states.UNIQUE


def test_used_hotkey_cannot_activate_again(conn):
    payload = activation_signal_payload(bytes(32), family="qwen38_27b", quant="fp8")
    fresh_hotkey, spent_hotkey = "5Fresh" + "A" * 42, "5Spent" + "B" * 42  # SS58-shaped
    fresh = PrivateSignal("activate", 97, 10, 3, fresh_hotkey, payload)
    spent = PrivateSignal("activate", 97, 10, 4, spent_hotkey, payload)
    guard.mark_used(conn, spent_hotkey, "eval")

    assert signals.handle(conn, [fresh, spent]) == 1
    rows = conn.execute("SELECT hotkey, state FROM private_registrations").fetchall()
    assert [(r["hotkey"], r["state"]) for r in rows] == [(fresh_hotkey, "ACTIVATED")]

    assert signals.handle(conn, [fresh, spent]) == 0, "the rescan next tick changes nothing"
    seen = conn.execute("SELECT hotkey, block, kind FROM commits ORDER BY hotkey").fetchall()
    assert [(r["hotkey"], r["block"], r["kind"]) for r in seen] == [
        (fresh_hotkey, 10, "activate"),
        (spent_hotkey, 10, "activate"),
    ], "every signal is recorded once, applied or not"


def test_a_draining_worker_claims_nothing(conn, s, spec):
    sid = submit(conn)
    conn.execute("UPDATE workers SET state = 'DRAINING' WHERE name = %s", (WORKER,))
    assert not validate(conn, s, spec)
    assert state(conn, sid) == states.SUBMITTED
    assert db.one(conn, "SELECT queued FROM workers WHERE name = %s", (WORKER,))["queued"] == 1


def test_the_worker_row_follows_the_claim(conn, s, spec):
    sid, _waiting = submit(conn, hotkey="hk-1", uid=1), submit(conn, hotkey="hk-2", uid=2)
    seen: list[object] = []

    def check(_model_dir: Path, **_: object) -> str | None:
        seen.append(db.one(conn, "SELECT busy_with FROM workers WHERE name = %s", (WORKER,)))
        return None

    validate(conn, s, spec, check=check)
    assert seen == [{"busy_with": sid}], "busy while the check runs"
    row = db.one(conn, "SELECT busy_with, queued FROM workers WHERE name = %s", (WORKER,))
    assert (row["busy_with"], row["queued"]) == (None, 2), "idle again, queue counted at claim"


def test_a_restart_reactivates_offline_but_not_draining(conn):
    name = db.register_worker(conn, "svc")
    for before, after in (("OFFLINE", "ACTIVE"), ("DRAINING", "DRAINING")):
        conn.execute("UPDATE workers SET state = %s WHERE name = %s", (before, name))
        assert db.register_worker(conn, "svc") == name
        assert db.one(conn, "SELECT state FROM workers WHERE name = %s", (name,))["state"] == after


def test_every_state_change_is_an_event(conn, s, spec):
    crown_genesis(conn)
    sid = submit(conn, hotkey="hk-1", uid=7)
    db.event(conn, sid, states.SUBMITTED)
    validate(conn, s, spec)
    dedup(conn, s)
    evaluate(conn, s, duel(0.1, 0.5))
    promoter.run_once(conn, worker=WORKER, s=s, spec=spec, fetch=fetch, publish=lambda *_: "url")

    rows = conn.execute(
        "SELECT state, worker FROM events WHERE submission_id = %s ORDER BY id", (sid,)
    ).fetchall()
    assert [r["state"] for r in rows] == [
        states.SUBMITTED,
        states.VALIDATING,
        states.VALID,
        states.DEDUPING,
        states.UNIQUE,
        states.EVALUATING,
        states.WON,
        states.CROWNING,
        states.CROWNED,
    ]
    assert [r["worker"] for r in rows if r["state"].endswith("ING")] == [WORKER] * 4


def test_events_keep_every_error(conn, s, spec):
    sid = submit(conn)

    def boom(*_a, **_k):
        raise RuntimeError("R2 is down")

    for _ in range(2):
        model_validation.run_once(
            conn,
            worker=WORKER,
            s=s,
            spec=spec,
            fetch=boom,
            list_files=listing,
            check=valid,
            load=lambda *_: FAMILY,
        )
    errors = conn.execute(
        "SELECT error FROM events WHERE submission_id = %s AND error IS NOT NULL ORDER BY id",
        (sid,),
    ).fetchall()
    assert [e["error"] for e in errors] == ["validation crashed: R2 is down"] * 2


def test_a_copy_is_rejected_and_spends_the_hotkey(conn, s, spec):
    crown_genesis(conn)
    genesis_error = quantization_error(0)
    bank = FakeBank({GENESIS: fingerprint(genesis_error)})
    sid = submit(conn, hotkey="hk-copy", uid=9)
    validate(conn, s, spec)

    def near_copy_of_genesis(_model_dir: Path) -> dict:
        return fingerprint(genesis_error + 0.05 * quantization_error(1))

    assert dedup(conn, s, bank, near_copy_of_genesis)
    assert state(conn, sid) == states.DUPLICATE
    assert db.one(conn, "SELECT error FROM submissions")["error"].startswith("duplicate:")
    run = db.one(conn, "SELECT duplicate, reason, duplicate_of, corr FROM dedups")
    assert (run["duplicate"], run["reason"], run["duplicate_of"]) == (True, "NEAR-COPY", GENESIS)
    assert run["corr"] >= dedup_config.COPY_CORR
    assert bank.puts == [(str(sid), AUDIT)], "kept for review, never compared against"
    assert guard.is_used(conn, "hk-copy"), "a copy costs the hotkey"
    assert not evaluate(conn, s, duel(0, 1)), "a duplicate never reaches the evaluator"


def test_a_copy_of_an_earlier_submission_is_caught(conn, s, spec):
    first, second = submit(conn, hotkey="hk-1", uid=1), submit(conn, hotkey="hk-2", uid=2)
    validate(conn, s, spec)
    validate(conn, s, spec)
    error = quantization_error(5)
    bank = FakeBank()

    def same_error(_model_dir: Path) -> dict:
        return fingerprint(error)

    dedup(conn, s, bank, same_error)
    dedup(conn, s, bank, same_error)
    assert (state(conn, first), state(conn, second)) == (states.UNIQUE, states.DUPLICATE)
    assert db.one(conn, "SELECT duplicate_of FROM dedups WHERE duplicate")["duplicate_of"] == str(
        first
    )


def test_a_crashing_dedup_retries_then_fails(conn, s, spec):
    sid = submit(conn)
    validate(conn, s, spec)

    def boom(_model_dir: Path) -> dict:
        raise RuntimeError("OpenSearch is down")

    dedup(conn, s, fp=boom)
    row = db.one(conn, "SELECT state, error FROM submissions")
    assert row["state"] == states.VALID and "OpenSearch is down" in row["error"]
    assert "OpenSearch is down" in db.one(conn, "SELECT error FROM dedups")["error"]
    dedup(conn, s, fp=boom)
    dedup(conn, s, fp=boom)
    assert state(conn, sid) == states.FAILED, "MAX_ATTEMPTS spent"


def test_a_dedup_that_keeps_crashing_uploads_its_fail(conn, s, spec):
    sid = submit(conn)
    validate(conn, s, spec)

    def boom(_model_dir: Path) -> dict:
        raise RuntimeError("OpenSearch is down")

    artifacts = FakeArtifacts()
    for _ in range(3):
        dedup(conn, s, fp=boom, upload=artifacts)
    doc = json.loads(artifacts[f"dedup/hk-1/{sid}/dedup_fail.json"])
    assert (doc["state"], doc["reason"], doc["barred"]) == (
        states.FAILED,
        "dedup crashed: OpenSearch is down",
        False,
    )


def test_each_stage_gets_its_own_attempts(conn, s, spec):
    crown_genesis(conn)
    sid = submit(conn)
    validate(conn, s, spec)
    dedup(conn, s)

    def crash(*_a, **_k) -> Duel:
        raise RuntimeError("GPU fell off the bus")

    evaluate(conn, s, crash)
    assert state(conn, sid) == states.UNIQUE, "validation and dedup did not spend eval's attempts"


def test_an_invalid_model_uploads_its_fails(conn, s, spec):
    crown_genesis(conn)
    sid = submit(conn)
    artifacts = FakeArtifacts()
    validate(conn, s, spec, check=lambda *_a, **_k: "no fp8", upload=artifacts)
    doc = json.loads(artifacts[f"validation/hk-1/{sid}/fails.json"])
    assert (doc["state"], doc["reason"], doc["hotkey"]) == (states.INVALID, "no fp8", "hk-1")


def test_a_valid_model_uploads_nothing(conn, s, spec):
    crown_genesis(conn)
    submit(conn)
    artifacts = FakeArtifacts()
    validate(conn, s, spec, upload=artifacts)
    assert artifacts == {}


def test_an_eval_uploads_its_samples_and_summary(conn, s, spec):
    crown_genesis(conn)
    submit(conn)
    validate(conn, s, spec)
    dedup(conn, s)
    artifacts = FakeArtifacts()
    evaluate(conn, s, duel(0.1, 0.5), upload=artifacts)
    eval_id = db.one(conn, "SELECT id FROM evals")["id"]
    assert set(artifacts) == {
        f"evals/hk-1/{eval_id}/samples.jsonl",
        f"evals/hk-1/{eval_id}/summary.json",
    }
    assert json.loads(artifacts[f"evals/hk-1/{eval_id}/summary.json"])["won"] is True


def test_a_crashed_eval_uploads_its_summary(conn, s, spec):
    crown_genesis(conn)
    submit(conn)
    validate(conn, s, spec)
    dedup(conn, s)

    def crash(*_a, **_k) -> Duel:
        raise RuntimeError("GPU fell off the bus")

    artifacts = FakeArtifacts()
    evaluate(conn, s, crash, upload=artifacts)
    eval_id = db.one(conn, "SELECT id FROM evals")["id"]
    assert set(artifacts) == {f"evals/hk-1/{eval_id}/summary.json"}
    assert (
        json.loads(artifacts[f"evals/hk-1/{eval_id}/summary.json"])["error"]
        == "GPU fell off the bus"
    )


def test_a_challenger_that_does_not_load_is_invalid_not_retried(conn, s, spec):
    crown_genesis(conn)
    sid = submit(conn)
    validate(conn, s, spec)
    dedup(conn, s)
    reason = "ValueError: There is no module or parameter named 'x.weight_scale_inv'"

    def exits_on(gpu: int) -> Scorer:
        def score(*_a, **_k) -> Duel:
            raise servers.ServerExited(gpu, reason, Path("/vllm/gpu.log"))

        return score

    evaluate(conn, s, exits_on(s.gpus("king")[0]))
    assert state(conn, sid) == states.UNIQUE, "the king not starting is a crash, retried"
    evaluate(conn, s, exits_on(s.gpus("challenger")[0]))
    assert state(conn, sid) == states.INVALID
    error = db.one(conn, "SELECT error FROM submissions WHERE id = %s", (sid,))["error"]
    assert error == f"the model does not load: {reason}"


def test_a_hotkey_is_barred_after_max_invalid(conn, s, spec):
    crown_genesis(conn)
    for i in range(validation_config.MAX_INVALID):
        assert not guard.is_used(conn, "hk-1"), f"still free after {i} invalid"
        submit(conn, hotkey="hk-1", block=100 + i)
        validate(conn, s, spec, check=lambda *_a, **_k: "no fp8")
    assert guard.is_used(conn, "hk-1")
    assert db.one(conn, "SELECT source FROM used_hotkeys")["source"] == "invalid"


def test_fails_json_says_when_the_hotkey_is_barred(conn, s, spec):
    crown_genesis(conn)
    barred = []
    for i in range(validation_config.MAX_INVALID):
        sid = submit(conn, hotkey="hk-1", block=100 + i)
        artifacts = FakeArtifacts()
        validate(conn, s, spec, check=lambda *_a, **_k: "no fp8", upload=artifacts)
        barred.append(json.loads(artifacts[f"validation/hk-1/{sid}/fails.json"])["barred"])
    assert barred == [False] * (validation_config.MAX_INVALID - 1) + [True]


def test_a_copy_uploads_who_it_copied_without_details(conn, s, spec):
    first, second = submit(conn, hotkey="hk-1", uid=1), submit(conn, hotkey="hk-2", uid=2)
    validate(conn, s, spec)
    validate(conn, s, spec)
    error, bank, artifacts = quantization_error(5), FakeBank(), FakeArtifacts()

    def same_error(_model_dir: Path) -> dict:
        return fingerprint(error)

    dedup(conn, s, bank, same_error, upload=artifacts)
    dedup(conn, s, bank, same_error, upload=artifacts)
    assert list(artifacts) == [f"dedup/hk-2/{second}/dedup_fail.json"]
    doc = json.loads(artifacts[f"dedup/hk-2/{second}/dedup_fail.json"])
    assert doc["duplicate_of"] == {"submission_id": str(first), "hotkey": "hk-1", "uid": 1}
    assert (doc["reason"], doc["barred"]) == ("NEAR-COPY", True)
    assert set(doc) == {
        "submission_id",
        "hotkey",
        "uid",
        "state",
        "reason",
        "corr",
        "copied_frac",
        "duplicate_of",
        "barred",
    }


def test_invalid_submissions_of_other_hotkeys_do_not_count(conn, s, spec):
    crown_genesis(conn)
    for i in range(validation_config.MAX_INVALID):
        submit(conn, hotkey=f"hk-{i}", uid=i)
        validate(conn, s, spec, check=lambda *_a, **_k: "no fp8")
    assert db.one(conn, "SELECT count(*) AS n FROM used_hotkeys")["n"] == 0


class FakeCommitments:
    """Commitments.CommitmentOf as the chain returns it."""

    def __init__(self, entries: dict[str, tuple[int, str]]) -> None:
        self.entries = entries

    def query_map(self, *, module: str, name: str, params: list) -> list:
        assert (module, name, params) == ("Commitments", "CommitmentOf", [97])
        return [
            (hotkey, {"block": block, "info": {"fields": [{"Raw9": "0x" + text.encode().hex()}]}})
            for hotkey, (block, text) in self.entries.items()
        ]


def test_plain_commitments_are_read():
    subtensor = FakeCommitments({"hk-old": (100, "v4|org/model")})
    assert list(chain.iter_committed(subtensor, 97)) == [("hk-old", 100, "v4|org/model")]


def test_backfill_bars_hotkeys_that_committed_before_the_start(conn):
    commits = [("hk-old", 100, "v4|org/model"), ("hk-new", 300, "r2activate:...")]
    assert guard.record_legacy(conn, commits, start_block=200) == 1
    assert guard.is_used(conn, "hk-old")
    assert not guard.is_used(conn, "hk-new")
    assert db.one(conn, "SELECT source, detail FROM used_hotkeys") == {
        "source": "backfill",
        "detail": {"block": 100},
    }
    assert guard.record_legacy(conn, commits, start_block=200) == 0, "a restart bars no one twice"
