"""Miner side of private R2 submissions.

    activate     commit r2activate with a one-time Ed25519 pubkey — asks for credentials
    upload       poll the sealed mailbox, upload the model files
    ready        commit r2ready with the model digest — freezes the upload and triggers
                 verification; the validator re-hashes every byte against it at download

The hotkey authors the two commits; the submission key (generated locally) is what the
credentials are sealed to. `submit_private` runs all three.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from loguru import logger

from common import chain
from common.config import chain_spec
from private_store.contracts import (
    activation_signal_payload,
    mailbox_object_key,
    mailbox_status_key,
    ready_signal_payload,
    registration_id,
)
from private_store.crypto import MailboxCipher
from private_store.digests import model_digest_from_inventory

_UPLOAD_AUTH = "upload-auth.json"
_SUBMISSION_KEY = "submission-key"
_USER_AGENT = {"User-Agent": "ingot-miner/2.0"}


def registration_id_for(hotkey_ss58: str, *, netuid: int) -> str:
    generation = chain_spec().chain_generation
    return registration_id(netuid=netuid, hotkey=hotkey_ss58, chain_generation=generation)


def _mailbox_url() -> str:
    url = chain_spec().mailbox_url
    if not url:
        raise SystemExit("chain.toml has no private_store.mailbox_url")
    return url


def _state_dir(hotkey_ss58: str) -> Path:
    path = (Path.cwd() / ".ingot-miner" / hotkey_ss58).resolve()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)


def submission_key(hotkey_ss58: str) -> Any:
    """Load, or create and persist, the hotkey's submission key. It must survive from activate
    to ready on the same machine: losing it means the credentials cannot be decrypted."""
    from nacl.signing import SigningKey

    path = _state_dir(hotkey_ss58) / _SUBMISSION_KEY
    if path.exists():
        return SigningKey(bytes.fromhex(path.read_text().strip()))
    seed = os.urandom(32)
    _write_private(path, seed.hex())
    return SigningKey(seed)


def _fetch_status(rid: str) -> dict[str, Any] | None:
    url = f"{_mailbox_url()}/{mailbox_status_key(rid)}?t={int(time.time())}"
    try:
        request = urllib.request.Request(url, headers=_USER_AGENT)
        with urllib.request.urlopen(request, timeout=10) as resp:
            status = json.loads(resp.read())
        return status if isinstance(status, dict) else None
    except Exception:
        return None


def _status_line(status: dict[str, Any]) -> str:
    parts = [f"upload attempt {status.get('upload_attempt')}/{status.get('max_upload_attempts')}"]
    parts += [str(status[key]) for key in ("step", "reason") if status.get(key)]
    if status.get("deadline"):
        parts.append(f"window closes {status['deadline']}")
    if status.get("next"):
        parts.append(f"next: {status['next']}")
    return " · ".join(parts)


def _commit(wallet: Any, netuid: int, network: str, payload: str, *, yes: bool) -> None:
    if not yes:
        print(f"about to commit on netuid {netuid} ({network}): {payload}")
        if payload.startswith("r2activate:"):
            status = _fetch_status(registration_id_for(wallet.hotkey.ss58_address, netuid=netuid))
            if status is not None:
                print(f"  current status: {_status_line(status)}")
        if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("aborted — nothing committed")
    chain.reveal_commit(wallet, netuid=netuid, network=network, payload=payload)


def activate(
    wallet: Any, *, netuid: int, network: str, family: str, quant: str, yes: bool = False
) -> str:
    ss58 = wallet.hotkey.ss58_address
    key = submission_key(ss58)
    payload = activation_signal_payload(bytes(key.verify_key), family=family, quant=quant)
    _commit(wallet, netuid, network, payload, yes=yes)
    return registration_id_for(ss58, netuid=netuid)


def fetch_credentials(
    *, hotkey_ss58: str, rid: str, signing_key: Any, timeout_s: float = 900.0
) -> dict[str, Any]:
    """Poll the public mailbox until the validator publishes sealed credentials."""
    base = _mailbox_url()
    key = mailbox_object_key(rid, 1)
    deadline = time.time() + timeout_s
    attempt = 0
    last_line: str | None = None
    logger.info("waiting for the validator to publish upload credentials…")
    while True:
        try:
            # r2.dev 403s the default urllib UA and edge-caches 404s
            request = urllib.request.Request(f"{base}/{key}?attempt={attempt}", headers=_USER_AGENT)
            with urllib.request.urlopen(request, timeout=15) as resp:
                ciphertext = resp.read()
            envelope = MailboxCipher.decrypt_for_miner(ciphertext, signing_key)
            if envelope["hotkey"] != hotkey_ss58 or envelope["allowed_prefix"].split("/")[2] != rid:
                raise SystemExit("mailbox envelope does not match this registration")
            _write_private(_state_dir(hotkey_ss58) / _UPLOAD_AUTH, json.dumps(envelope))
            logger.info("credentials received; prefix {}", envelope["allowed_prefix"])
            return envelope
        except SystemExit:
            raise
        except Exception:
            status = _fetch_status(rid)
            if status is not None and (line := _status_line(status)) != last_line:
                logger.info(line)
                last_line = line
            if time.time() > deadline:
                raise SystemExit(
                    "timed out waiting for credentials — "
                    + (last_line or "the validator has not published a status for this hotkey")
                ) from None
            time.sleep(10)
            attempt += 1


def upload_paths(local_dir: str) -> list[str]:
    """Every relpath the uploader sends; dotfiles and __pycache__ are left out."""
    root = Path(local_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"model path is not a directory: {local_dir}")
    paths = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        hidden = any(part.startswith(".") or part == "__pycache__" for part in rel.parts)
        if hidden:
            continue
        paths.append(rel.as_posix())
    if not paths:
        raise SystemExit("model directory has no files to upload")
    return paths


def plan_upload(local_dir: str) -> list[tuple[str, int, str, Path]]:
    """(relpath, size, sha256, absolute path) for every file to upload."""
    root = Path(local_dir).resolve()
    plan = []
    for rel in upload_paths(local_dir):
        path = root / rel
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        plan.append((rel, path.stat().st_size, digest.hexdigest(), path))
    return plan


def _s3_from_envelope(envelope: dict[str, Any]) -> Any:
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=envelope["r2_endpoint"],
        aws_access_key_id=envelope["access_key_id"],
        aws_secret_access_key=envelope["secret_access_key"],
        aws_session_token=envelope["session_token"],
        region_name="auto",
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
    )


def model_digest(plan: list[Any]) -> str:
    return model_digest_from_inventory([(rel, size, sha) for rel, size, sha, _ in plan])


def upload_objects(client: Any, envelope: dict[str, Any], plan: list[Any]) -> None:
    from boto3.s3.transfer import TransferConfig

    bucket, prefix = envelope["private_model_bucket"], envelope["allowed_prefix"]
    workers = max(1, int(os.environ.get("R2_UPLOAD_FILE_WORKERS", "4")))
    transfer = TransferConfig(multipart_chunksize=64 * 1024 * 1024, max_concurrency=8)

    def put(rel: str, size: int, path: Path) -> None:
        client.upload_file(str(path), bucket, prefix + rel, Config=transfer)
        logger.info("uploaded {} ({} bytes)", rel, size)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(put, rel, size, path) for rel, size, _sha, path in plan]:
            future.result()


def upload(*, hotkey_ss58: str, local_dir: str) -> str:
    auth = _state_dir(hotkey_ss58) / _UPLOAD_AUTH
    if not auth.exists():
        raise SystemExit("no upload credentials on disk; run activate first")
    envelope = json.loads(auth.read_text())
    plan = plan_upload(local_dir)
    digest = model_digest(plan)
    upload_objects(_s3_from_envelope(envelope), envelope, plan)
    logger.info("upload complete; model digest {}", digest)
    return digest


def ready(wallet: Any, *, netuid: int, network: str, model_digest: str, yes: bool = False) -> None:
    _commit(wallet, netuid, network, ready_signal_payload(model_digest), yes=yes)
    (_state_dir(wallet.hotkey.ss58_address) / _UPLOAD_AUTH).unlink(missing_ok=True)
    logger.info("ready committed; upload access will be revoked and the model verified")


def submit_private(
    wallet: Any,
    *,
    netuid: int,
    network: str,
    family: str,
    quant: str,
    local_dir: str,
    yes: bool = False,
) -> str:
    ss58 = wallet.hotkey.ss58_address
    key = submission_key(ss58)
    rid = activate(wallet, netuid=netuid, network=network, family=family, quant=quant, yes=yes)
    fetch_credentials(hotkey_ss58=ss58, rid=rid, signing_key=key)
    digest = upload(hotkey_ss58=ss58, local_dir=local_dir)
    ready(wallet, netuid=netuid, network=network, model_digest=digest, yes=yes)
    logger.info("submitted registration {}", rid)
    return rid
