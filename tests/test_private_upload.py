"""The private upload path through the real database: activate → credentials → upload → ready
→ verify → SUBMITTED, a bad upload, and the retry after an INVALID verdict. S3, Cloudflare
and the chain are faked; the storage and crypto code is real."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import ClientError
from nacl.signing import SigningKey

from chain_reader import controller, signals
from common import db, states
from common.chain import PrivateSignal
from common.config import R2Settings, chain_spec
from private_store.cloudflare import ParentToken
from private_store.contracts import (
    activation_signal_payload,
    mailbox_object_key,
    mailbox_status_key,
    model_prefix,
    ready_signal_payload,
    registration_id,
)
from private_store.crypto import MailboxCipher, encode_ss58_public_key
from private_store.digests import model_digest_from_inventory
from private_store.storage import MailboxStore, R2UploadController

BUCKET, MAILBOX = "private-models", "mailbox"
SUBMISSION_KEY = SigningKey(b"s" * 32)
HOTKEY = encode_ss58_public_key(bytes(SigningKey(b"m" * 32).verify_key))
RID = registration_id(netuid=97, hotkey=HOTKEY, chain_generation=chain_spec().chain_generation)
FILES = {
    "config.json": b'{"architectures": ["Qwen3"]}',
    "model.safetensors.index.json": b"{}",
    "model.safetensors": b"tensor-bytes",
}
R2 = R2Settings(
    _env_file=None,
    account_id="a" * 32,
    endpoint="https://account.r2.cloudflarestorage.com",
    private_models_bucket_name=BUCKET,
    mailbox_bucket_name=MAILBOX,
    mailbox_signing_key="ab" * 32,
)


class FakeBody:
    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def read(self, amount: int | None = None) -> bytes:
        end = len(self.data) if amount is None else min(len(self.data), self.pos + amount)
        chunk, self.pos = self.data[self.pos : end], end
        return chunk


class FakeS3:
    """Enough of the S3 client for MailboxStore and R2UploadController."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}
        self.multipart: list[dict[str, str]] = []

    def put(self, bucket: str, key: str, data: bytes) -> None:
        self.objects[(bucket, key)] = (data, {"sha256": hashlib.sha256(data).hexdigest()})

    def body(self, bucket: str, key: str) -> bytes:
        return self.objects[(bucket, key)][0]

    def status(self, rid: str) -> dict[str, Any]:
        return json.loads(self.body(MAILBOX, mailbox_status_key(rid)))

    def put_object(
        self, Bucket: str, Key: str, Body: bytes, Metadata: dict | None = None, **_: Any
    ):
        self.objects[(Bucket, Key)] = (Body, dict(Metadata or {}))
        return {}

    def get_object(self, Bucket: str, Key: str, **_: Any):
        if (Bucket, Key) not in self.objects:
            error = {"Error": {"Code": "NoSuchKey"}, "ResponseMetadata": {"HTTPStatusCode": 404}}
            raise ClientError(error, "GetObject")
        data, metadata = self.objects[(Bucket, Key)]
        return {
            "Body": FakeBody(data),
            "Metadata": metadata,
            "ETag": f'"{hashlib.md5(data).hexdigest()}"',
        }

    def head_object(self, Bucket: str, Key: str):
        data, metadata = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(data),
            "Metadata": metadata,
            "ETag": f'"{hashlib.md5(data).hexdigest()}"',
        }

    def list_objects_v2(self, Bucket: str, Prefix: str, **_: Any):
        contents = [
            {"Key": key, "Size": len(data), "LastModified": datetime.now(UTC)}
            for (bucket, key), (data, _meta) in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def list_multipart_uploads(self, Bucket: str, Prefix: str, **_: Any):
        return {
            "Uploads": [u for u in self.multipart if u["Key"].startswith(Prefix)],
            "IsTruncated": False,
        }

    def list_parts(self, Bucket: str, Key: str, UploadId: str, **_: Any):
        upload = next(u for u in self.multipart if u["UploadId"] == UploadId)
        return {"Parts": [{"Size": int(upload.get("PartBytes", 0))}], "IsTruncated": False}

    def abort_multipart_upload(self, Bucket: str, Key: str, UploadId: str):
        self.multipart = [u for u in self.multipart if u["UploadId"] != UploadId]

    def download_file(self, Bucket: str, Key: str, Filename: str, **_: Any) -> None:
        with open(Filename, "wb") as out:
            out.write(self.body(Bucket, Key))

    def delete_objects(self, Bucket: str, Delete: dict):
        for entry in Delete["Objects"]:
            self.objects.pop((Bucket, entry["Key"]), None)
        return {}


class FakeGateway:
    def __init__(self) -> None:
        self.revoked: list[str] = []

    def create_parent_token(self, token_name: str) -> ParentToken:
        return ParentToken("tok-1", "tok-1", "s" * 64)

    def revoke_parent_token(self, token_id: str) -> None:
        self.revoked.append(token_id)


def make_deps(s3: FakeS3, gateway: FakeGateway | None = None) -> controller.Deps:
    spec = chain_spec()
    return controller.Deps(
        r2=R2,
        spec=spec,
        gateway=gateway or FakeGateway(),
        mailbox=MailboxStore(s3, bucket=MAILBOX),
        uploads=R2UploadController(s3, private_model_bucket=BUCKET),
        cipher=MailboxCipher(SigningKey(b"v" * 32)),
    )


def activate(block: int, key: SigningKey = SUBMISSION_KEY, quant: str = "fp8") -> PrivateSignal:
    payload = activation_signal_payload(bytes(key.verify_key), family="qwen38_27b", quant=quant)
    return PrivateSignal("activate", netuid=97, block=block, uid=5, hotkey=HOTKEY, payload=payload)


def ready(digest: str, block: int) -> PrivateSignal:
    payload = ready_signal_payload(digest)
    return PrivateSignal("ready", 97, block, 5, HOTKEY, payload, block_hash=f"0xabc{block}")


def upload(s3: FakeS3, prefix: str, files: dict[str, bytes] = FILES) -> str:
    """What the miner CLI does with its credentials: the files; returns the model digest."""
    for path, data in files.items():
        s3.put(BUCKET, f"{prefix}{path}", data)
    return model_digest_from_inventory(
        [(path, len(data), hashlib.sha256(data).hexdigest()) for path, data in files.items()]
    )


def registration(conn: db.Conn) -> db.Row:
    return db.one(conn, "SELECT * FROM private_registrations WHERE registration_id = %s", (RID,))


def submit_through(conn: db.Conn, s3: FakeS3, deps: controller.Deps) -> str:
    """Drive one upload from activate to SUBMITTED."""
    assert signals.handle(conn, [activate(block=100)]) == 1
    assert controller.tick(conn, deps)
    digest = upload(s3, registration(conn)["model_prefix"])
    assert signals.handle(conn, [ready(digest, block=200)]) == 1
    assert controller.tick(conn, deps)
    assert controller.tick(conn, deps)
    return digest


def test_private_upload_becomes_a_submission(conn):
    s3, gateway = FakeS3(), FakeGateway()
    deps = make_deps(s3, gateway)
    prefix = model_prefix(RID)

    assert signals.handle(conn, [activate(block=100)]) == 1
    assert controller.tick(conn, deps), "ACTIVATED → CREDENTIALED"
    reg = registration(conn)
    assert (reg["state"], reg["model_prefix"], reg["parent_token_id"]) == (
        "CREDENTIALED",
        prefix,
        "tok-1",
    )
    envelope = MailboxCipher.decrypt_for_miner(
        s3.body(MAILBOX, mailbox_object_key(RID, 1)), SUBMISSION_KEY
    )
    assert envelope["allowed_prefix"] == prefix and envelope["hotkey"] == HOTKEY

    digest = upload(s3, prefix)
    assert signals.handle(conn, [ready(digest, block=200)]) == 1
    assert controller.tick(conn, deps), "READY → REVOKED"
    assert registration(conn)["state"] == "REVOKED" and gateway.revoked == ["tok-1"]
    assert (MAILBOX, mailbox_object_key(RID, 1)) not in s3.objects, "mailbox wiped before verifying"

    assert controller.tick(conn, deps), "REVOKED → SUBMITTED"
    reg = registration(conn)
    sub = db.one(conn, "SELECT * FROM submissions")
    assert reg["state"] == "SUBMITTED" and reg["submission_id"] == sub["id"]
    assert sub["model_uri"] == f"s3://{BUCKET}/{prefix.rstrip('/')}@sha256:{digest}"
    assert (sub["hotkey"], sub["uid"], sub["block"], sub["block_hash"]) == (
        HOTKEY,
        5,
        200,
        "0xabc200",
    )
    assert sub["state"] == states.SUBMITTED and sub["model_hash"] == digest
    assert s3.status(RID)["step"].startswith("verified")
    assert not controller.tick(conn, deps)


def test_a_bad_upload_fails_and_keeps_its_bytes(conn):
    s3 = FakeS3()
    deps = make_deps(s3)
    signals.handle(conn, [activate(block=100)])
    controller.tick(conn, deps)
    prefix = registration(conn)["model_prefix"]
    digest = upload(s3, prefix)
    s3.put(BUCKET, f"{prefix}modeling.py", b"import os")
    signals.handle(conn, [ready(digest, block=200)])
    controller.tick(conn, deps)

    controller.tick(conn, deps)
    reg = registration(conn)
    assert reg["state"] == "FAILED" and "files not allowed: modeling.py" in reg["fault_message"]
    assert conn.execute("SELECT count(*) AS n FROM submissions").fetchone()["n"] == 0
    assert any(key.startswith(prefix) for _bucket, key in s3.objects), "only the reaper deletes"
    assert s3.status(RID)["next"].startswith("run `ingot activate` to use upload attempt 2 of 3")


def test_an_invalid_verdict_hands_the_attempt_back(conn):
    s3 = FakeS3()
    deps = make_deps(s3)
    submit_through(conn, s3, deps)
    conn.execute("UPDATE submissions SET state = 'INVALID', error = 'no fp8 tensors'")

    controller.sweep(conn, deps)
    reg = registration(conn)
    assert (reg["state"], reg["submission_id"], reg["attempt_count"]) == ("SUBMITTED", None, 1)
    assert s3.status(RID)["reason"] == "no fp8 tensors"

    assert signals.handle(conn, [activate(block=300)]) == 1, "the miner claims attempt 2"
    assert controller.tick(conn, deps)
    reg = registration(conn)
    assert (reg["state"], reg["attempt_count"], reg["model_prefix"]) == (
        "CREDENTIALED",
        2,
        model_prefix(RID, 2),
    )
    assert any(key.startswith(model_prefix(RID)) for _b, key in s3.objects), (
        "attempt 1's bytes stay"
    )


def test_a_duplicate_is_told_why_and_gets_no_attempt_back(conn):
    s3 = FakeS3()
    deps = make_deps(s3)
    submit_through(conn, s3, deps)
    conn.execute("UPDATE submissions SET state = 'DUPLICATE', error = 'duplicate: of genesis'")

    controller.sweep(conn, deps)
    reg = registration(conn)
    assert (reg["state"], reg["attempt_count"]) == ("SUBMITTED", 1) and reg["submission_id"]
    status = s3.status(RID)
    assert (status["reason"], status["blocked"]) == ("duplicate: of genesis", "dedup")
    assert status["next"] == "hotkey is blocked (dedup) — no retry"


def test_a_spent_hotkey_is_refused_at_verification(conn):
    s3 = FakeS3()
    deps = make_deps(s3)
    signals.handle(conn, [activate(block=100)])
    controller.tick(conn, deps)
    digest = upload(s3, registration(conn)["model_prefix"])
    signals.handle(conn, [ready(digest, block=200)])
    controller.tick(conn, deps)
    conn.execute("INSERT INTO used_hotkeys (hotkey, source) VALUES (%s, 'swap')", (HOTKEY,))

    controller.tick(conn, deps)
    assert registration(conn)["state"] == "FAILED"
    assert s3.status(RID)["blocked"] == "hotkey_used"
    assert conn.execute("SELECT count(*) AS n FROM submissions").fetchone()["n"] == 0


def test_parked_multipart_bytes_over_the_limit_close_and_purge_the_upload(conn):
    s3, gateway = FakeS3(), FakeGateway()
    deps = make_deps(s3, gateway)
    deps.spec = replace(deps.spec, upload_limits={("qwen38_27b", "fp8"): 1000})
    prefix = model_prefix(RID)
    assert signals.handle(conn, [activate(block=100)]) == 1
    assert controller.tick(conn, deps)
    s3.put(BUCKET, prefix + "config.json", b"{}")
    assert controller.sweep_credentialed(conn, deps) == 0, "under the limit"

    s3.multipart.append({"Key": prefix + "big.safetensors", "UploadId": "u1", "PartBytes": "5000"})
    assert controller.sweep_credentialed(conn, deps) == 1
    reg = registration(conn)
    assert reg["state"] == "FAILED" and "multipart" in reg["fault_message"]
    assert gateway.revoked == ["tok-1"]
    assert not s3.multipart and not any(k.startswith(prefix) for _, k in s3.objects)


def test_a_disallowed_file_closes_and_purges_the_upload(conn):
    s3, gateway = FakeS3(), FakeGateway()
    deps = make_deps(s3, gateway)
    prefix = model_prefix(RID)
    assert signals.handle(conn, [activate(block=100)]) == 1
    assert controller.tick(conn, deps)
    upload(s3, prefix)
    assert controller.sweep_credentialed(conn, deps) == 0, "config, index, shards"
    s3.put(BUCKET, prefix + "modeling_qwen.py", b"import os")
    assert controller.sweep_credentialed(conn, deps) == 1
    reg = registration(conn)
    assert reg["state"] == "FAILED" and "file not allowed: modeling_qwen.py" in reg["fault_message"]
    assert not any(k.startswith(prefix) for _, k in s3.objects)


def test_activation_declares_an_allowed_pair_that_reaches_the_submission(conn):
    s3 = FakeS3()
    deps = make_deps(s3)
    assert signals.handle(conn, [activate(block=90, quant="gptq")]) == 0, "unlisted pair refused"
    assert not conn.execute("SELECT 1 FROM private_registrations").fetchone()
    assert signals.handle(conn, [activate(block=100)]) == 1
    assert (registration(conn)["family"], registration(conn)["quant"]) == ("qwen38_27b", "fp8")
    assert controller.tick(conn, deps)
    digest = upload(s3, model_prefix(RID))
    assert signals.handle(conn, [ready(digest, block=200)]) == 1
    assert controller.tick(conn, deps) and controller.tick(conn, deps)
    sub = db.one(conn, "SELECT family, quant FROM submissions")
    assert (sub["family"], sub["quant"]) == ("qwen38_27b", "fp8")


def test_the_committed_digest_is_what_the_validator_rehashes_at_download(tmp_path):
    from miner.submit import model_digest, plan_upload
    from private_store.digests import snapshot_digest

    for path, data in FILES.items():
        (tmp_path / path).write_bytes(data)
    digest = model_digest(plan_upload(str(tmp_path)))
    assert digest == snapshot_digest(tmp_path) == upload(FakeS3(), model_prefix(RID))
    (tmp_path / "model.safetensors").write_bytes(b"swapped")
    assert snapshot_digest(tmp_path) != digest
