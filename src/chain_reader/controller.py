"""Access controller: drives private_registrations through its lifecycle.

ACTIVATED -> CREDENTIALED  mint a parent token, publish sealed credentials
READY     -> REVOKED       kill upload access, wipe the mailbox
REVOKED   -> SUBMITTED     verify the frozen prefix, create the submissions row
          -> FAILED        verification failed; bytes retained, the miner may activate again
SUBMITTED + submission_id NULL   retry parked: the submission came back INVALID with attempts
                                 left, so the next r2activate (signals.py) is attempt n+1.

Verification only runs on rows whose parent token was deleted. Uploaded bytes stay on every
terminal path; reap_expired is the only place a model is deleted.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any, Protocol

import boto3
import httpx
from botocore.config import Config
from loguru import logger as log
from nacl.signing import SigningKey

from chain_reader import config
from common import db, guard, states
from common.config import ChainSpec, R2Settings
from common.db import Conn
from private_store.cloudflare import CloudflareR2TokenGateway, ParentToken
from private_store.contracts import mailbox_object_key, mailbox_status_key, model_prefix
from private_store.crypto import MailboxCipher
from private_store.digests import ArtifactIntegrityError
from private_store.r2_credentials import create_local_temporary_credentials
from private_store.storage import MailboxStore, R2UploadController

Row = Mapping[str, Any]
_REAPABLE_STATES = (states.INVALID, states.DUPLICATE, states.LOST)


class TokenGateway(Protocol):
    def create_parent_token(self, token_name: str) -> ParentToken: ...
    def revoke_parent_token(self, token_id: str) -> None: ...


@dataclass
class Deps:
    r2: R2Settings
    spec: ChainSpec
    gateway: TokenGateway
    mailbox: MailboxStore
    uploads: R2UploadController
    cipher: MailboxCipher
    status_cache: dict[str, str] = field(default_factory=dict)


def build_deps(r2: R2Settings, spec: ChainSpec) -> Deps:
    s3 = boto3.client(
        "s3",
        endpoint_url=r2.endpoint,
        aws_access_key_id=r2.access_key_id,
        aws_secret_access_key=r2.secret_access_key,
        region_name="auto",
        config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
    )
    return Deps(
        r2=r2,
        spec=spec,
        gateway=CloudflareR2TokenGateway(
            httpx.Client(timeout=30),
            account_id=r2.account_id,
            management_token=r2.api_token,
            bucket=r2.private_models_bucket_name,
            jurisdiction=r2.jurisdiction,
        ),
        mailbox=MailboxStore(s3, bucket=r2.mailbox_bucket_name),
        uploads=R2UploadController(
            s3,
            private_model_bucket=r2.private_models_bucket_name,
        ),
        cipher=MailboxCipher(SigningKey(r2.mailbox_signing_key_bytes())),
    )


def _prefix_of(row: Row) -> str:
    return row["model_prefix"] or model_prefix(row["registration_id"])


def _publish_status(
    deps: Deps,
    row: Row,
    *,
    step: str,
    reason: str = "",
    deadline: datetime | None = None,
    retryable: bool = False,
    blocked: str = "",
) -> None:
    """Plaintext status the miner's CLI polls. Never raises; an unchanged status is not re-put."""
    rid = row["registration_id"]
    try:
        used = row["attempt_count"]
        maximum = config.MAX_UPLOAD_ATTEMPTS + row["extra_attempts"]
        left = max(0, maximum - used)
        if not retryable:
            nxt = ""
        elif blocked:
            nxt = f"hotkey is blocked ({blocked}) — no retry"
        elif left:
            nxt = f"run `ingot activate` to use upload attempt {used + 1} of {maximum}"
        else:
            nxt = "no upload attempts left for this hotkey"
        body = {
            "protocol_version": 1,
            "registration_id": rid,
            "upload_attempt": used,
            "max_upload_attempts": maximum,
            "upload_attempts_left": left,
            "blocked": blocked,
            "step": step,
            "reason": reason,
            "deadline": deadline.isoformat() if deadline else None,
            "next": nxt,
        }
        fingerprint = json.dumps(body, sort_keys=True)
        if deps.status_cache.get(rid) == fingerprint:
            return
        body["updated_at"] = datetime.now(UTC).isoformat()
        deps.mailbox.put_status(
            mailbox_status_key(rid), json.dumps(body, separators=(",", ":")).encode()
        )
        deps.status_cache[rid] = fingerprint
    except Exception as exc:
        log.warning("status publish failed for {}: {}", rid, exc)


def _credential(conn: Conn, row: Row, deps: Deps) -> None:
    r2 = deps.r2
    rid = row["registration_id"]
    prefix = model_prefix(rid, row["attempt_count"])
    if row["parent_token_id"]:
        deps.gateway.revoke_parent_token(row["parent_token_id"])
    parent = deps.gateway.create_parent_token(f"ingot-registration-{rid}")
    temp = create_local_temporary_credentials(
        endpoint=r2.endpoint,
        account_id=r2.account_id,
        parent_access_key_id=parent.access_key_id,
        parent_secret_access_key=parent.secret_access_key,
        bucket=r2.private_models_bucket_name,
        prefix=prefix,
        ttl_seconds=config.CREDENTIAL_TTL_S,
    )
    ciphertext, _ = deps.cipher.create_ciphertext(
        submission_pubkey=bytes.fromhex(row["submission_pubkey"]),
        hotkey=row["hotkey"],
        netuid=row["netuid"],
        registration_id=rid,
        generation=1,
        endpoint=r2.endpoint,
        private_model_bucket=r2.private_models_bucket_name,
        allowed_prefix=prefix,
        access_key_id=temp.access_key_id,
        secret_access_key=temp.secret_access_key,
        session_token=temp.session_token,
        expires_at=datetime.fromtimestamp(temp.expires_at_unix, tz=UTC),
        chain_generation=deps.spec.chain_generation,
    )
    key = mailbox_object_key(rid, 1)
    deps.mailbox.delete([key])
    deps.mailbox.publish(key, ciphertext)
    conn.execute(
        """
        UPDATE private_registrations
        SET state = 'CREDENTIALED', parent_token_id = %s, credential_expires_at = to_timestamp(%s),
            model_prefix = %s, updated_at = now()
        WHERE id = %s
        """,
        (parent.token_id, temp.expires_at_unix, prefix, row["id"]),
    )
    log.info("credentials published for {}", rid)
    _publish_status(
        deps,
        row,
        step="credentials issued — upload now, then commit ready",
        deadline=datetime.now(UTC) + timedelta(seconds=config.UPLOAD_WINDOW_S),
    )


def _revoke(conn: Conn, row: Row, deps: Deps) -> None:
    rid = row["registration_id"]
    if row["parent_token_id"]:
        deps.gateway.revoke_parent_token(row["parent_token_id"])
    deps.mailbox.delete([mailbox_object_key(rid, 1)])
    conn.execute(
        "UPDATE private_registrations SET state = 'REVOKED', updated_at = now() WHERE id = %s",
        (row["id"],),
    )
    log.info("upload access revoked for {}", rid)
    _publish_status(deps, row, step="upload frozen — verifying your upload")


def _fail(conn: Conn, row: Row, deps: Deps, reason: str, *, blocked: str = "") -> None:
    conn.execute(
        "UPDATE private_registrations SET state = 'FAILED', fault_message = %s, updated_at = now()"
        " WHERE id = %s",
        (reason, row["id"]),
    )
    log.warning("verification FAILED for {}: {}", row["registration_id"], reason)
    _publish_status(
        deps, row, step="failed verification", reason=reason, retryable=True, blocked=blocked
    )


def _verify(conn: Conn, row: Row, deps: Deps) -> None:
    rid = row["registration_id"]
    prefix = _prefix_of(row)
    pair = deps.spec.pair(row["family"], row["quant"])
    try:
        deps.uploads.verify_upload(
            model_prefix=prefix,
            registration_id=rid,
            max_upload_bytes=pair.upload_limit,
            allowed=lambda name: deps.spec.allows_file(name, pair.format),
        )
    except ArtifactIntegrityError as exc:
        deps.uploads.abort_multipart_uploads(prefix)
        _fail(conn, row, deps, str(exc))
        return
    if guard.is_used(conn, row["hotkey"]):
        _fail(conn, row, deps, "hotkey may not submit again", blocked="hotkey_used")
        return
    digest = row["model_digest"]
    bucket = deps.r2.private_models_bucket_name
    submission = conn.execute(
        """
        INSERT INTO submissions
            (hotkey, uid, coldkey, model_uri, model_hash, block, block_hash, family, quant)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_hash) WHERE state NOT IN ('INVALID', 'FAILED') DO NOTHING
        RETURNING id
        """,
        (
            row["hotkey"],
            row["uid"],
            row["coldkey"],
            f"s3://{bucket}/{prefix.rstrip('/')}@sha256:{digest}",
            digest,
            row["ready_block"],
            row["ready_block_hash"],
            row["family"],
            row["quant"],
        ),
    ).fetchone()
    if submission is None:
        # The digest is the miner's claim until validation downloads and re-hashes the bytes.
        # Another row holding it that is not verified yet may be a squatter who copied this
        # digest from the chain: wait for validation to settle it (INVALID frees the digest).
        other = conn.execute(
            "SELECT state FROM submissions WHERE model_hash = %s"
            " AND state NOT IN ('INVALID', 'FAILED')",
            (digest,),
        ).fetchone()
        if other is not None and other["state"] in (states.SUBMITTED, states.VALIDATING):
            log.info("{}: digest held by an unverified submission; verifying later", rid)
            return
        _fail(conn, row, deps, "these exact model bytes were already submitted")
        return
    db.event(conn, submission["id"], states.SUBMITTED)
    conn.execute(
        """
        UPDATE private_registrations
        SET state = 'SUBMITTED', model_digest = %s, submission_id = %s, updated_at = now()
        WHERE id = %s
        """,
        (digest, submission["id"], row["id"]),
    )
    log.info("verified {} -> submission {}", rid, submission["id"])
    _publish_status(deps, row, step="verified — queued for validation and evaluation")


def _close_pair(conn: Conn, row: Row, deps: Deps) -> None:
    """The row's pair closed while it was in flight: no credentials for it, its upload access and
    bytes removed, and its attempt given back (FAILED, so the miner may activate the open pair),
    as db.release_closed does for a closed pair's submissions."""
    if row["parent_token_id"]:
        deps.gateway.revoke_parent_token(row["parent_token_id"])
    deps.mailbox.delete([mailbox_object_key(row["registration_id"], 1)])
    if row["model_prefix"]:
        deps.uploads.abort_multipart_uploads(row["model_prefix"])
        deps.uploads.cleanup_model_prefix(row["model_prefix"])
    conn.execute(
        "UPDATE private_registrations SET state = 'FAILED', fault_message = 'closed',"
        " extra_attempts = extra_attempts + 1, updated_at = now() WHERE id = %s",
        (row["id"],),
    )
    reason = (
        f"{row['family']}.{row['quant']} is closed; activate {deps.spec.family}.{deps.spec.quant}"
    )
    log.warning("registration {} closed — {}", row["registration_id"], reason)
    row = {**row, "extra_attempts": row["extra_attempts"] + 1}  # the status counts it
    _publish_status(deps, row, step="failed — quant closed", reason=reason, retryable=True)


def _close_upload_window(conn: Conn, deps: Deps, row: Row, reason: str) -> None:
    if row["parent_token_id"]:
        deps.gateway.revoke_parent_token(row["parent_token_id"])
    deps.mailbox.delete([mailbox_object_key(row["registration_id"], 1)])
    deps.uploads.abort_multipart_uploads(_prefix_of(row))
    conn.execute(
        "UPDATE private_registrations SET state = 'FAILED', fault_message = %s, updated_at = now()"
        " WHERE id = %s",
        (reason, row["id"]),
    )
    log.warning("upload window closed for {} — {}", row["registration_id"], reason)
    _publish_status(deps, row, step="failed — upload window closed", reason=reason, retryable=True)


def sweep_credentialed(conn: Conn, deps: Deps) -> int:
    """Fail registrations that never commit ready in time or abuse the upload quota."""
    rows = conn.execute(
        """
        SELECT id, registration_id, hotkey, parent_token_id, attempt_count, extra_attempts,
               model_prefix, family, quant, EXTRACT(EPOCH FROM (now() - updated_at)) AS age_s
        FROM private_registrations WHERE state = 'CREDENTIALED'
        """
    ).fetchall()
    swept = 0
    for row in rows:
        if deps.spec.closed(row["family"], row["quant"]):
            _close_pair(conn, row, deps)
            swept += 1
            continue
        breach = None
        if row["age_s"] > config.UPLOAD_WINDOW_S:
            reason = f"upload window expired ({int(row['age_s'])}s with no ready signal)"
        else:
            pair = deps.spec.pair(row["family"], row["quant"])
            breach = deps.uploads.quota_breach(
                _prefix_of(row),
                pair.upload_limit,
                partial(deps.spec.allows_file, fmt=pair.format),
            )
            reason = f"upload quota abused: {breach}" if breach else None
        if reason is None:
            continue
        _close_upload_window(conn, deps, row, reason)
        if breach:
            # R2 cannot cap a prefix, so over-quota bytes are removed as soon as they are seen.
            deps.uploads.cleanup_model_prefix(_prefix_of(row))
        swept += 1
    return swept


def plan_reap(
    retained: Mapping[str, Any], *, cutoff: datetime, protected: Collection[str]
) -> list[str]:
    return sorted(p for p, newest in retained.items() if newest < cutoff and p not in protected)


def _protected_prefixes(conn: Conn) -> set[str]:
    rows = conn.execute(
        """
        SELECT pr.registration_id, pr.attempt_count, pr.model_prefix
        FROM private_registrations pr
        LEFT JOIN submissions s ON s.id = pr.submission_id
        WHERE pr.state <> 'FAILED'
          AND NOT (pr.state = 'SUBMITTED' AND pr.submission_id IS NULL)
          AND COALESCE(s.state, '') <> ALL(%s::text[])
        """,
        (list(_REAPABLE_STATES),),
    ).fetchall()
    return {_prefix_of(row) for row in rows}


def reap_expired(conn: Conn, deps: Deps) -> int:
    """Delete uploads older than the retention window that no live registration needs."""
    cutoff = datetime.now(UTC) - timedelta(hours=config.RETENTION_HOURS)
    retained = deps.uploads.retained_uploads()
    victims = plan_reap(retained, cutoff=cutoff, protected=_protected_prefixes(conn))
    reaped = 0
    for prefix in victims:
        try:
            removed = deps.uploads.cleanup_model_prefix(prefix)
        except Exception as exc:
            log.opt(exception=True).warning("reap failed for {} — will retry: {}", prefix, exc)
            continue
        reaped += 1
        log.warning("reaped {} ({} objects)", prefix, removed)
    return reaped


def _reset_for_retry(conn: Conn, deps: Deps, row: Row) -> None:
    conn.execute(
        """
        UPDATE private_registrations
        SET activation_block = GREATEST(activation_block, COALESCE(ready_block, activation_block)),
            parent_token_id = NULL, credential_expires_at = NULL, model_prefix = NULL,
            ready_block = NULL, ready_block_hash = NULL,
            model_digest = NULL, submission_id = NULL, fault_message = NULL, updated_at = now()
        WHERE id = %s
        """,
        (row["id"],),
    )
    log.info(
        "retry available for {} — attempt {} used", row["registration_id"], row["attempt_count"]
    )
    _publish_status(
        deps,
        row,
        step="last attempt did not pass validation",
        reason=row["error"] or "",
        retryable=True,
    )


def resubmit_invalid(conn: Conn, deps: Deps) -> int:
    """Hand an INVALID submission's attempt back to its miner while attempts remain."""
    rows = conn.execute(
        """
        SELECT pr.id, pr.registration_id, pr.hotkey, pr.attempt_count, pr.extra_attempts, s.error
        FROM private_registrations pr
        JOIN submissions s ON s.id = pr.submission_id
        WHERE pr.state = 'SUBMITTED' AND s.state = %s
          AND pr.attempt_count < %s + pr.extra_attempts
        """,
        (states.INVALID, config.MAX_UPLOAD_ATTEMPTS),
    ).fetchall()
    reset = 0
    for row in rows:
        if guard.is_used(conn, row["hotkey"]):
            _publish_status(
                deps,
                row,
                step="last attempt did not pass validation",
                reason=row["error"] or "",
                retryable=True,
                blocked="hotkey_used",
            )
            continue
        _reset_for_retry(conn, deps, row)
        reset += 1
    return reset


def publish_duplicates(conn: Conn, deps: Deps) -> int:
    """Tell each miner whose submission copied a banked model why; the dedup spent the hotkey."""
    rows = conn.execute(
        """
        SELECT pr.registration_id, pr.attempt_count, pr.extra_attempts, s.error
        FROM private_registrations pr
        JOIN submissions s ON s.id = pr.submission_id
        WHERE s.state = %s
        """,
        (states.DUPLICATE,),
    ).fetchall()
    for row in rows:
        _publish_status(
            deps,
            row,
            step="rejected — copies a model already submitted",
            reason=row["error"] or "",
            retryable=True,
            blocked="dedup",
        )
    return len(rows)


def tick(conn: Conn, deps: Deps) -> bool:
    """Advance one registration by one step. False when there is nothing to do."""
    with conn.transaction():
        row = conn.execute(
            """
            SELECT * FROM private_registrations
            WHERE state IN ('ACTIVATED', 'READY', 'REVOKED')
            ORDER BY updated_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            """
        ).fetchone()
        if row is None:
            return False
        try:
            if deps.spec.closed(row["family"], row["quant"]):
                _close_pair(conn, row, deps)  # never minted, kept or verified for a closed pair
            elif row["state"] == "ACTIVATED":
                _credential(conn, row, deps)
            elif row["state"] == "READY":
                _revoke(conn, row, deps)
            else:
                _verify(conn, row, deps)
        except Exception as exc:
            log.opt(exception=True).warning(
                "{} step failed for {} — will retry: {}", row["state"], row["registration_id"], exc
            )
            conn.execute(
                "UPDATE private_registrations SET updated_at = now() WHERE id = %s", (row["id"],)
            )
        return True


def sweep(conn: Conn, deps: Deps) -> None:
    resubmit_invalid(conn, deps)
    publish_duplicates(conn, deps)
    reap_expired(conn, deps)
