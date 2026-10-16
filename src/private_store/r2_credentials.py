from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from urllib.parse import urlparse

MAX_TEMPORARY_CREDENTIAL_TTL_SECONDS = 604_800
MINER_PREFIX_SCOPE = "object-read-write"


@dataclass(frozen=True, slots=True)
class TemporaryCredentials:
    access_key_id: str
    secret_access_key: str
    session_token: str
    expires_at_unix: int


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def create_local_temporary_credentials(
    *,
    endpoint: str,
    account_id: str,
    parent_access_key_id: str,
    parent_secret_access_key: str,
    bucket: str,
    prefix: str,
    ttl_seconds: int = MAX_TEMPORARY_CREDENTIAL_TTL_SECONDS,
    issued_at_unix: int | None = None,
) -> TemporaryCredentials:
    """Create Cloudflare R2 prefix-scoped temporary credentials.

    The JWT is signed locally with the parent token's derived S3 secret; no
    Cloudflare API call is involved. Cloudflare rejects the documented
    fine-grained ``actions`` claim, so credentials rely on the platform's
    prefix-scoped object-read-write capability.
    """
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path not in ("", "/"):
        raise ValueError("R2 endpoint must be an HTTPS origin without a path")
    if not account_id or not parent_access_key_id or not parent_secret_access_key:
        raise ValueError("account and parent credential fields are required")
    if not bucket:
        raise ValueError("bucket is required")
    if not prefix or not prefix.endswith("/"):
        raise ValueError("temporary credential prefix must be slash-terminated")
    if not 1 <= ttl_seconds <= MAX_TEMPORARY_CREDENTIAL_TTL_SECONDS:
        raise ValueError("temporary credential TTL must be between 1 and 604800 seconds")

    issued_at = int(time.time()) if issued_at_unix is None else issued_at_unix
    expires_at = issued_at + ttl_seconds
    header = {"alg": "HS256", "typ": "JWT"}
    claims = {
        "aud": parsed.netloc,
        "bucket": bucket,
        "exp": expires_at,
        "iat": issued_at,
        "iss": parent_access_key_id,
        "paths": {"objectPaths": [], "prefixPaths": [prefix]},
        "scope": MINER_PREFIX_SCOPE,
        "sub": account_id,
    }
    signing_input = (
        _b64url(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(claims, sort_keys=True, separators=(",", ":")).encode())
    )
    signature = _b64url(
        hmac.new(parent_secret_access_key.encode(), signing_input.encode(), hashlib.sha256).digest()
    )
    jwt = f"{signing_input}.{signature}"
    return TemporaryCredentials(
        access_key_id=parent_access_key_id,
        secret_access_key=hashlib.sha256(jwt.encode()).hexdigest(),
        session_token=base64.b64encode(f"jwt/{jwt}".encode()).decode(),
        expires_at_unix=expires_at,
    )
